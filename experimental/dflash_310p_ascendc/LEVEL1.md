# 310P DFlash MRv1 · 算子上机验证

在 Ascend310P3 / CANN 9.2.1 容器的仓库根目录执行。记录镜像、torch_npu 版本、`git rev-parse HEAD` 和 `git status --short`。

## 1. 构建和 CPU reference

```bash
source /usr/local/Ascend/cann-9.2.1/set_env.sh
op_dir=experimental/dflash_310p_ascendc
cmake -S "$op_dir" -B "$op_dir/build" \
  -DSOC_VERSION=Ascend310P3 -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.1 \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build "$op_dir/build" -j 8
readelf -d "$op_dir/build/libdflash_310p_torch.so" | grep -E 'RUNPATH|NEEDED'
ldd "$op_dir/build/libdflash_310p_torch.so"
python3 -m unittest experimental.dflash_310p_ascendc.test_reference
```

确认 RUNPATH 包含 `$ORIGIN/lib`，且 AscendC kernel 库从本次构建的 `build/lib` 加载。编译失败时回传首个编译错误和构建配置。

## 2. 输入准备算子

```bash
python3 -m experimental.dflash_310p_ascendc.run --stage eager --device 0 --k 7 \
  --output /tmp/dflash-310p-prepare-k7-eager.json
python3 -m experimental.dflash_310p_ascendc.run --stage graph --device 0 --k 7 \
  --output /tmp/dflash-310p-prepare-k7-graph.json
python3 -m experimental.dflash_310p_ascendc.run --stage graph --device 0 --k 1 2 3 4 5 6 \
  --output /tmp/dflash-310p-prepare-k1-6-graph.json
```

回放在固定地址上切换有效请求数、上下文长度、block table 和非法输入。须逐位匹配 CPU 输出及前后各一个 32B 哨兵。用例覆盖两组不同页大小、全拒绝、部分拒绝、context 行数大于 query 行数、跨页、最大长度、padding 和无效页。分别验证从 anchor 和从下一行采样。

## 3. 原生 SplitFuse 能力探针

```bash
python3 -m experimental.dflash_310p_ascendc.probe_attention --stage eager --device 0 --k 7 \
  --output /tmp/dflash-310p-attention-k7-eager.json
python3 -m experimental.dflash_310p_ascendc.probe_attention --stage graph --device 0 --k 7 \
  --output /tmp/dflash-310p-attention-k7-graph.json
```

探针用 35B TP2 的 draft head 形状，在同一图内验证全注意力非因果和滑窗因果两条 mask 路径。滑窗缩短为 32 以便低成本覆盖截断边界。两组拥有各自的固定地址 mask 和 block table，均在回放前刷新。两次 replay 修改有效长度和物理页顺序，逐组与 CPU dense reference 比对。若 eager 或 graph 失败，请回传 JSON 及第一条 CANN 错误，尤其是算子名称、错误码和 tiling 信息。

已通过 LEVEL1 的环境可沿用 `.so`，单跑 K=7 attention graph 命令验证两组同图回放。输入准备的五份原始上机结果仍有效。

输入准备和 SplitFuse 可以在不同空闲设备并行。回传实际设备、编译日志及 JSON。此级使用合成张量，服务接入的下一道验收是 K=7、35B TP2、C1 后 C10 持续补入。

### 输入准备接口约定

`torch.ops.dflash_310p.prepare_` 由调用方持有输入和输出的存储，输入与写入目标不能别名。

| Tensor | 连续 INT32 布局 |
|---|---|
| requests | `[B,8]`，前三项分别为 packed context 起点、止点、拒绝数，第四项是已选 bonus ID。padding 的起点为 -1。有效区间按行递增且互斥 |
| positions | `[C]`，C 是 8 的倍数，包含物理尾部容量 |
| block_tables | `[G,B,P]`，每组同一物理行宽、页面 ID 使用该组物理 KV 编号 |
| group_info | `[G,8]`，前三项分别为页大小、页数、保留的空页 ID（-1 表示禁用） |
| query | `[G,B,6,8]`，六项依次为输入 ID、位置、槽、采样索引、采样位置、请求元数据。元数据前四项为逻辑 query 起点、止点、有效 KV 长度、状态 |
| context | `[G,2,C]`，依次为 packed context 位置和 slot；无效 slot 为 -1 |

一个 group 对应一个核和独有的写区间。请求块写回为 6 个连续 32B 块，context 每次写两个独立 32B 块。首版以设备精度和写入边界为验收。状态值 1 表示非法请求，2 表示非法页或位置；调用者须停止消费报错行。滑窗已淘汰的 context 页以保留空页表示，保留其 RoPE 位置并输出 -1 slot。准备数据的 CPU/GPU 路径负责提供有序且互斥的请求区间。

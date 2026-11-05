# L310P.1 ReplaySSM SD 独立算子上机清单

目标：Ascend310P3 × 2（每次先在其中一张空闲卡做隔离测试），CANN 9.2.1、
torch_npu 2.10.0.post7、vLLM 0.30.0+empty、vllm-ascend v0.30.0 源码。
本级仅验证三个算子；不启用 `vllm serve` 的 ReplaySSM 分派。

## V7 增量：Cast 原子化定位（本轮唯一上机步骤）

V6 的 `fp16_block_cast` 是 FP16→FP32→FP16 的复合路径；其失败尚不能
定位到某一条 Cast 指令。V7 增加独立 `cast_probe` 算子，使用仓内
`recurrent_gated_delta_rule_v310/op_kernel/recurrent_gated_delta_rule_v310.h`
的对齐 `DataCopy` 块宽、`MTE2_V`/`V_MTE3` 同步和 `Cast` 写法。
每个方向单独输出到 GM，另保留原 `io_probe` queue round-trip 作对照。
V7 **不修改 verify/compact 的递推和精度标准**，不执行性能测试。

在仓库根目录全新构建，确认编译退出码为 0、加载成功，然后只运行 Cast 探针：

```bash
git rev-parse HEAD
git status --short
cmake -S experimental/replayssm_sd_310p_ascendc -B experimental/replayssm_sd_310p_ascendc/build \
  -DSOC_VERSION=Ascend310P3 \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.1 \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build experimental/replayssm_sd_310p_ascendc/build -j 8
python3 experimental/replayssm_sd_310p_ascendc/run.py \
  --mtp 3 --stage cast_probe --device 0 \
  --output /tmp/replayssm-310p-v7-cast-probe.json
```

脚本对 count=16/32/64 分别运行以下路径；即使数值失败，仍记录其他 case。
JSON 含输入、预期、实际的数值与 uint16/uint32 十六进制位模式，以及首个错误
索引、GM 地址对 32B 的余数。以 **逐位一致** 为本轮验收标准，测试数据延续
V6 的 `0, 0.25, …`，不放宽容差。

| mode | 独立路径 | 可定位的阶段 |
|---:|---|---|
| 0 | FP16 对齐 GM→UB→GM | FP16 读入/写回及位宽 |
| 1 | FP32 对齐 GM→UB→GM | FP32 读入/写回及位宽 |
| 2 | FP16→FP32 (`CAST_NONE`)，直接写 FP32 | 上行 Cast |
| 3 | FP32→FP16 (`CAST_NONE`)，直接写 FP16 | 下行 Cast |
| 4 | FP32→FP16 (`CAST_RINT`)，直接写 FP16 | V6 的下行模式 |
| 5 | FP16→FP32→FP16 (`NONE/NONE`) | 两条 Cast 串联与同步 |
| 6 | FP16→FP32→FP16 (`NONE/RINT`) | V6 模式在独立 UB 布局下的结果 |

`legacy_queue_roundtrip` 则运行 V6 原始的 VECIN→VECCALC→VECOUT 路径。
若 0/1 已失败，先排查对齐搬运；若 0/1 通过而 2 或 3/4 首次失败，
再分别检查 Cast 方向与舍入模式；若独立 2–6 通过而旧路径失败，重点检查
queue、UB 偏移和同步。**在拿到该结果之前不修改正式 verify**。
请回传完整 JSON、第一条 CCE 编译错误或设备错误、构建日志和设备型号；
本轮不运行 V6 下方的 smoke、sequence、graph 或 perf 步骤。

## V6 增量：区分 FP16/FP32 块宽并检查 UB 内复制

V5 的 status 写回通过，但 verify 在 `pos=0` 仍出现大量 NaN。此时没有有效
history 行被读取，因此不能仅凭 V5 报告把故障归因于读取了邻近 NaN ring 槽。
本轮把 verify/compact 中 UB 内 FP32 `DataCopy` 改为原生 V310 使用的
`Adds(x, 0)` 复制方式；compact 逐项读取 FP32 gate 时先向下对齐地址，再取块内
目标元素。另增加两个原子探针：FP16 的 32B 块有 **16** 个元素，
FP32 的 32B 块有 **8** 个元素。读写的块数均按 `ceil(元素数×sizeof(T)/32)`
计算；仅对有效位置参与计算，不清零测试数据中的未使用 NaN 槽。

完成第 1 节的全新构建后，先运行已知支持的对齐探针，再运行 MTP3/5 smoke：

```bash
python3 experimental/replayssm_sd_310p_ascendc/run.py \
  --mtp 3 --stage aligned_probe --device 0 \
  --output /tmp/replayssm-310p-v6-aligned.json
for mtp in 3 5; do
  python3 experimental/replayssm_sd_310p_ascendc/run.py \
    --mtp "$mtp" --batch 2 --stage smoke --tile-v 32 --device 0 \
    --output "/tmp/replayssm-310p-v6-mtp${mtp}-smoke.json"
done
```

`aligned_probe` 应包含原有 4 项及 `fp16_block_cast`、`fp32_block_copy`，
全部通过后再跑 smoke。任一 smoke 失败即停止，回传首个失败 case 的 JSON、
完整报错、`out` 的 NaN 数量和前 16 个输出值。两组 smoke 均通过后再跑
第 2 节的 `sequence` 和 `graph`；本轮仍为单核，不做性能比较。

## V5 增量：验证 verify/compact 对齐搬运

V3 已证实 4B `DataCopyPad` 读写失败；V4 的 `aligned_probe` 和 MTP3/5
cursor 已通过。本次把 verify/compact 的设备侧 `DataCopyPad` 全部改为
对齐 `DataCopy`。单元素 gate/status 采用 32B 读改写，并将 verify/compact
暂时限制为 **1 个 AIV** 以避免多核更新同一块的竞争。`--cores` 参数此轮
仍被接受，但实际发射 1 核；**此轮时延不可用于性能验收**。完成第 1 节的
**全新构建**后运行：

```bash
for mtp in 3 5; do
  python3 experimental/replayssm_sd_310p_ascendc/run.py \
    --mtp "$mtp" --batch 2 --stage smoke --tile-v 32 --device 0 \
    --output "/tmp/replayssm-310p-v5-mtp${mtp}-smoke.json"
done
```

先只回传 smoke JSON；任一项失败即停止，回传第一条 CCE 错误或运行日志。
两个 smoke 均通过后，再运行下文的 `sequence`、`graph`。下一轮拆分
32B 块的核归属，恢复多核并测试性能。

## 1. 环境、构建、加载

在仓库根目录执行，先确认所见设备确为 `Ascend310P3`：

```bash
git rev-parse HEAD
git status --short
npu-smi info
python3 -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.get_device_name(0))'
python3 -m unittest experimental.replayssm_sd_310p_ascendc.tests.test_reference
cmake -S experimental/replayssm_sd_310p_ascendc -B experimental/replayssm_sd_310p_ascendc/build \
  -DSOC_VERSION=Ascend310P3 \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.1 \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build experimental/replayssm_sd_310p_ascendc/build -j 8
readelf -d experimental/replayssm_sd_310p_ascendc/build/libreplayssm_sd_310p_torch.so | grep -E 'RPATH|RUNPATH'
python3 -c 'import torch, torch_npu; torch.ops.load_library("experimental/replayssm_sd_310p_ascendc/build/libreplayssm_sd_310p_torch.so"); print(torch.ops.replayssm_sd_310p.verify_._schemas, torch.ops.replayssm_sd_310p.compact_._schemas, torch.ops.replayssm_sd_310p.cursor_._schemas)'
```

**构建命令必须以退出码 0 完成**。`merge_m200_obj_text` 即使在 `.so` 出现后失败，
也视为构建失败；符号、RPATH 和加载成功均不能证明设备代码实际执行。不要手工重链
或运行该次产物。回传第一条错误、完整编译日志、
`CMakeCache.txt` 中的 SoC/CANN/Python 配置，以及构建目录下的文件清单。
优先区分编译器 API 不支持、M200 合并失败、UB 预算、link/RPATH、运行时设备错误。

上轮 `1a789b1` 的 MTP3/MTP5 verify 误差与**全零输出**的 CPU 结果逐位一致，
且三个算子的状态均保持初值。若完整构建通过但仍复现，请先以空闲卡运行 cursor 的
独立测试，并回传该次 `msprof` 的任务名、task duration 和首个设备错误；暂不调
FP16 容差或修改 recurrence。

## 2. 精度与状态生命周期

先运行原子化 I/O 探针，再运行单算子写回探针和小 batch。
`run.py` 的输出 JSON 即回传件：

```bash
mkdir -p /tmp/replayssm-310p-l1
python3 experimental/replayssm_sd_310p_ascendc/run.py \
  --mtp 3 --stage io_probe --device 0 \
  --output /tmp/replayssm-310p-l1/io-probe.json
```

这一轮先回传 `io-probe.json`，不要在探针失败后继续运行三个正式算子。
五项均通过时再执行：

```bash
for mtp in 3 5; do
  python3 experimental/replayssm_sd_310p_ascendc/run.py \
    --mtp "$mtp" --batch 1 --stage cursor_probe --device 0 \
    --output "/tmp/replayssm-310p-l1/mtp${mtp}-cursor-probe.json"
  python3 experimental/replayssm_sd_310p_ascendc/run.py \
    --mtp "$mtp" --batch 2 --stage smoke --device 0 \
    --output "/tmp/replayssm-310p-l1/mtp${mtp}-smoke.json"
  python3 experimental/replayssm_sd_310p_ascendc/run.py \
    --mtp "$mtp" --batch 1 --stage sequence --device 0 \
    --output "/tmp/replayssm-310p-l1/mtp${mtp}-sequence.json"
  python3 experimental/replayssm_sd_310p_ascendc/run.py \
    --mtp "$mtp" --batch 2 --stage graph --device 0 \
    --output "/tmp/replayssm-310p-l1/mtp${mtp}-graph.json"
done
```

`io_probe` 会依次执行 6 个独立 case，**即使一个失败也会记录其余结果**：

| case | 单独检验 | 预期 |
|---|---|---|
| `aligned_copy` | 原生 V310 使用的 `DataCopyParams{1,1,0,0}`，32B GM→UB→GM | 8 个输入值原样写回 |
| `scalar_aligned` | `SetValue` 8 个 int32 + 对齐 `DataCopy` | 8 个 0x1234 |
| `pad_read` | 4B `DataCopyPad` 读取 + 对齐写回 | 8 个值均为输入首项 |
| `pad_write` | 4B `DataCopyPad` 写回 | 仅首项为 0x1234 |
| `block_num` | `GetBlockNum` + 对齐写回 | 8 个 1 |
| `aligned_rmw` | 对齐读旧值、修改首元素、对齐写回 | `[4660,-1×7]` |

先只回传 `io-probe.json` 与首次编译错误；如果 aligned 两项未过，不跑后续
Replay 精度。若 aligned 两项通过而 pad 项失败，下一轮按仓内 V310 的
`DataCopyPadCustom`/`DataCopyCustom` 改造小块元数据 I/O，暂不更改 recurrence。

`smoke` 覆盖 history 0/7/12/16、ring 跨尾、checkpoint 与请求 ring 槽号分离、混合 flush、padding、
verify→compact→cursor；`sequence` 覆盖 24 轮接受、折叠和 ring 周转；
`graph` 只检查 Replay 三算子，捕获后更改设备元数据并重放，不依赖 native aclnn。
逐元素 int32 状态码必须精确一致；
FP16 输出和 checkpoint 使用 atol=rtol=2^-9、匹配率至少 99%、
最大绝对误差不超过 0.1 的混合容差，并记录实际最大误差。

## 3. 与 310P 原生算子对比

只有第 2 节全过后才运行性能矩阵。脚本直接调用
`torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310`，与 Replay 使用相同
Q/K/V/g/beta；native 的初始 state 先从同一份 checkpoint 和 history 重建。
该算子是 vllm-ascend 源码中 `csrc/attention/recurrent_gated_delta_rule_v310`
定义的**自定义 ACLNN**，不能只在 CANN 内置 `libopapi.so` 查找。先确认
`vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so`
存在且导出 `aclnnRecurrentGatedDeltaRuleV310`，再运行性能矩阵：

```bash
find vllm_ascend/_cann_ops_custom -name 'libcust_opapi.so' -print
find vllm_ascend/_cann_ops_custom -name 'libcust_opapi.so' \
  -exec sh -c 'nm -D "$1" | grep RecurrentGatedDeltaRuleV310' _ {} \;
```

若缺少自定义库/符号，需检查本机 vllm-ascend 的 `build_aclnn` 产物及
`ASCEND_CUSTOM_OPP_PATH`；此时仍可完成 Replay 的 graph 验证，但暂不做 native 对比。
计时区不含输入准备和状态重置，预热 20 次、图重放 100 次：

```bash
for mtp in 3 5; do
  for batch in 1 16 32 64; do
    python3 experimental/replayssm_sd_310p_ascendc/run.py \
      --mtp "$mtp" --batch "$batch" --stage perf --tile-v 32 \
      --cores 7 --iterations 100 --warmup 20 --device 0 \
      --output "/tmp/replayssm-310p-l1/mtp${mtp}-b${batch}-perf.json"
  done
done
```

每组记录 verify、compact、cursor、三个算子组合、native 的 graph replay
p50/p90/min，场景为 pos0/12/16 与 flush0/1。单算子时延和三个算子总时延
都要报告；此级只建立基线，不以快于 native 为通过条件。`tile_v=64` 仅在
tile32 构建、精度和图测试均通过后补测，compact 始终使用 tile32。

## 4. 回传

请回传：全部 JSON、首个失败 case 的完整 stderr/CCE 日志、
`git rev-parse HEAD`、`git status --short`、设备型号与包版本、
`readelf` RUNPATH 输出。若能同时抓 msprof，附上每个 kernel 的 task duration、
AIV vector/scalar/MTE2/MTE3；这部分用于下一轮优化，不影响本级验收。

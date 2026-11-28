# ReplaySSM SD 在 Ascend 910C 单 die 的算子迭代

目标是 Qwen3.5-9B TP1 B256 与 Qwen3.5-27B TP4 B32/B64，均用 MTP3、BF16 checkpoint、FP16 D/K、FP32 G。Blue2 的 CANN 版本为 9.2.1，编译目标为 `ascend910_9362`。27B TP4 单 die 实验按单 rank 几何执行。

## 构建和候选

- Compact V5 一次批量搬入各元数据数组与两段历史 K/D/G。支持 tile64，在非 flush 只更新 status。V6 额外把 gate 后缀积应用到各 D 行，整块 checkpoint 只缩放一次。V6 改变 FP32 累加顺序，需要单独验证。
- Verify V23 在计算 K 侧点积后，将同一 UB 工作区用于一次 32 KiB BF16 checkpoint 搬入，再 Cast 到独立 FP32 state。Cube 路径的 packed panel 只让 AIV 搬入和迭代八个有用列，Cube 仍按 16 列输出。
- V24 在 V23 输出候选以后复用同一份 state 和历史 K/D/G 做 compact。只在 flush 行折叠，复用 product UB 一次写回 BF16 checkpoint。混合核的 `--mixed-suffix`、`--mixed-qkv` 和 `--mixed-balanced` 用于分别定位投影递推、搬运和切分。
- Cursor 在 B32–1024 且物理槽不超过 1024 时批量读入槽、状态和接收信息，同一 AIV 按实际槽散写，status 整段写出。小批次和超出槽数范围时用原路径。
- 服务对照仍可使用 native、V13、V20。V23、Compact V5/V6 和混合核通过独立 Torch 库做对照，服务 dispatch 不参与算子性能消融。

```bash
source /usr/local/Ascend/cann/set_env.sh
: "${CANN_ROOT:=/usr/local/Ascend/cann-9.2.1}"
: "${ASCEND_RT_VISIBLE_DEVICES:?set this to one idle die}"
export ASCEND_RT_VISIBLE_DEVICES
mkdir -p /tmp/replay-910c
for op in replayssm_sd_ascendc replayssm_sd_compact_ascendc replayssm_sd_cursor_ascendc replayssm_cube_910b/mixed_verify; do
  out="/tmp/replay-910c/${op##*/}"
  cmake -S "experimental/$op" -B "$out" -DSOC_VERSION=ascend910_9362 \
    -DCMAKE_BUILD_TYPE=Release -DASCEND_CANN_PACKAGE_PATH="$CANN_ROOT"
  cmake --build "$out" -j4
done
verify=/tmp/replay-910c/replayssm_sd_ascendc/libreplayssm_sd_torch.so
compact=/tmp/replay-910c/replayssm_sd_compact_ascendc/libreplayssm_sd_compact_torch.so
cursor=/tmp/replay-910c/replayssm_sd_cursor_ascendc/libreplayssm_sd_cursor_torch.so
mix=/tmp/replay-910c/mixed_verify/libreplayssm_sd_mix_torch.so
```

## 门槛与 batch 曲线

先用独立 runner 验证烟测和图回放，覆盖 pos0/7/12/15/16、环形跨尾、padding 和不同长度。Compact 还需检查有 flush、无 flush、混合 flush 与双槽寻址。Cursor 需检查乱序槽、未选槽、reset、零接收和非法输入。V24 用对比脚本的 `fused_v24` 路径同时检查候选输出、ring 和 checkpoint。

```bash
export OMP_NUM_THREADS=4
for scenario in 27b:4 9b:1; do
  geometry=${scenario%:*}
  tp=${scenario#*:}
  for stage in smoke graph; do
    python3 experimental/replayssm_sd_ascendc/run.py --library "$verify" --geometry "$geometry" \
      --tp "$tp" --batch 2 --variant 23 --tile-v 128 --cores 40 --stage "$stage" \
      --output "/tmp/replay-910c/verify-$geometry-$stage.json"
  done
done
for version in 5 6; do
  python3 experimental/replayssm_sd_compact_ascendc/run.py --library "$compact" \
    --batch 2 --variant "$version" --tile-v 64 --stage all \
    --output "/tmp/replay-910c/compact-$version.json"
done
for b in 32 64 256; do
  python3 experimental/replayssm_sd_cursor_ascendc/run.py --library "$cursor" \
    --batch "$b" --stage perf --output "/tmp/replay-910c/cursor-$b.json"
done
```

框架 native 对照需要加载匹配的 CANN vendor 库。Blue2 容器中的路径为 `/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer`，将它加入 `ASCEND_CUSTOM_OPP_PATH`，将其 `op_api/lib` 加入 `LD_LIBRARY_PATH`。确认库中存在 `aclnnAddRmsNormBias`，否则服务可能在预热时停止。

```bash
run=experimental/replayssm_cube_910b/run.py
libs=(--suite mixed --verify-library "$verify" --compact-library "$compact" --mixed-library "$mix")
paths=(native v20 v23 compact_v5 compact_v6 v23_compact_v5 mixed_packed mixed_packed_compact_v5)
for geometry in 27b 9b; do
  if [[ "$geometry" == 27b ]]; then
    batches=1,2,3,4,7,8,9,15,16,17,18,19,31,32,33,47,63,64,65
  else
    batches=1,3,8,15,16,17,18,31,32,33,63,64,65,127,128,129,191,255,256
  fi
  for flush in none mixed; do
    python3 "$run" "${libs[@]}" --geometry "$geometry" --batches "$batches" \
      --pos 15 --length 4 --base 31 --flush "$flush" --path "${paths[@]}" --iterations 25 \
      --output "/tmp/replay-910c/sweep-$geometry-$flush.json"
  done
  python3 "$run" "${libs[@]}" --geometry "$geometry" --batches 15,16,17,18,31,32,33,63,64,65 \
    --pos 15 --length 4 --base 31 --flush-count 2 --path native v23_compact_v5 mixed_packed_compact_v5 \
    --iterations 25 --output "/tmp/replay-910c/fixed-2-flush-$geometry.json"
done
```

`mixed` 固定前 `max(1, floor(B/8))` 行 flush。`--flush-count 2` 用于排除 flush 行数跳变。按 B15/16/17/18、B31/32/33、B63/64/65、B127/128/129 和 B255/256 对比绝对 p50/p90 与相对 native 比值。若 B 增长后相对 native 的时延比额外跳升超过 10%，且重跑后 p90 也有类似变化，单独检查实际任务数、活跃核和 flush 行分布。

27B 在单 die 模拟 48 个 GDN 层，每层使用独立 checkpoint 和 ring。每轮重置在计时区外，`gdn_stack` 记录整张 stack 图的时间。该实验只计算 GDN 的差值，省略注意力、MLP、TP 通信和调度。Cursor 在服务里每轮调用一次，单独计量。

```bash
for flush in none mixed; do
  python3 "$run" "${libs[@]}" --geometry 27b --batches 32,64 --layers 48 \
    --flush "$flush" --path native vector_pair v23_compact_v5 mixed_packed_compact_v5 \
    --iterations 25 --output "/tmp/replay-910c/stack-27b-$flush.json"
done
```

9B TP1 用同一个服务环境顺序跑 native 和候选，记录实际 running、FULL decode 图形状、目标模型 graph forward p50/p90 和拒绝采样接受率。只有真实 running 以及图形状都覆盖目标 B256，才用于目标场景验收。最终比较以同负载的 decode forward 为准。

## Blue2 首轮测量

以下为单 die、MTP3、历史 15、ring base 31、约 1/8 请求 flush 时的设备 graph replay p50。单位为 µs，前三行分别使用两种模型的实际每 rank 几何。

| 场景 | Native | Verify V23 | Packed Cube verify | Compact V1→V5 | Packed Cube + V5 | V24 融合 |
|---|---:|---:|---:|---:|---:|---:|
| 27B TP4 B32 | 205 | 224 | 201 | 106→61 | 263 | 253 |
| 27B TP4 B64 | 375 | 420 | 360 | 171→92 | 450 | 480 |
| 9B TP1 B256 | 3,710 | 4,290 | 3,825 | 1,359→631 | 4,467 | 4,816 |

分项来自独立计时，组合时延直接测量。V24 来自同工作负载的第二轮。独立 Compact V6 在 B32/B64/B256 分别为约 65/91/572 µs，V5 同轮为约 75/100/629 µs。Cursor B32/B64/B256 的普通接收从 71/132/491 µs 降至约 51/51/51 µs，按每轮一次计。

在 48/24 层 GDN stack 上，混合 flush 时 native / 最快 Replay p50 为 27B B32 9.06/11.77 ms、B64 17.77/20.68 ms，9B B256 88.98/107.12 ms。全无 flush 时为 9.02/9.73、17.79/17.49、88.95/93.03 ms。每层独立 cache，计时省略模型其余部分。9B HTTP 短文本对照中，native 与 V20 都完成 256 条请求。Native 实际记录的最大 FULL decode 图是 192 行，V20 发出过 256 行的图。两者未形成相同 B256 的完整 forward 对照。

固定两行 flush 的两次复测中，27B packed 路径在 B15→16 从约 140 µs 升至 169–174 µs，B17 回落至 162–164 µs。20 个 Cube 上，key-head 分组的最长队列由 3 个任务变成 4 个，按 value head 分段的消融用于核对这一归因和修复效果。

同一新二进制的 A/B 复测中，按 value head 均分将 B15→16 的增幅从约 6–13% 降至约 2–7%。在 27B B32/B64，均分后约 263/457 µs，同轮 key-head 切分约 258/448 µs。保留均分作为小批次候选，大批保持 key-head 切分。本轮批量搬入候选 Q/K/V 的收益更稳定。在固定历史 15、约 1/8 flush 下，27B B32/B64 与 9B B256 的 verify 为 203/349/3,543 µs，配 Compact V5 的组合为 255/427/4,180 µs。

## 混合 flush 测量与加权

在相同 batch 下扫描实际 flush 行数，并控制 flush 行在 batch 内的摆放。用于对比的合成样本默认将 flush 行的历史设为 15，非 flush 行设为 7。三个布局分别为前缀、均匀分散和固定种子随机。B32 可先用 F=0/1/2/4/8/16/32，B64 用 F=0/2/4/8/16/32/64，B256 用 F=0/4/8/16/32/64/128/256。用单层图扫比例，在 F=0 和 F≈B/8 用独立 cache stack 校准。

```bash
for f in 0 1 2 4 8 16 32; do
  python3 "$run" "${libs[@]}" --geometry 27b --batches 32 --mixed-qkv \
    --pos 15 --nonflush-pos 7 --base 31 --flush-count "$f" --flush-layout random \
    --path native mixed_packed_compact_v5 mixed_packed_compact_v6 fused_v24 \
    --iterations 25 --output "/tmp/replay-910c/27b-b32-f$f-random.json"
done
python3 experimental/replayssm_cube_910b/model_flush.py /tmp/replay-910c/27b-b32-f*-random.json \
  --batch 32 --layers 48 --output /tmp/replay-910c/27b-b32-flush-curve.json
```

在服务按真实或填充后的 graph batch 归组，记录每次 forward 中有多少有效请求需要 flush，再用 `--histogram` 传入该组观测，例如 JSON `{"0":9,"1":21,"2":25,"4":8}`。模型在实测行数之间做分段插值，按各行数出现的频率加权，同一 F 优先使用整个 stack 实测。曲线输出包含每个 F 的 p50/p90，模型加权的是平均值。估计范围为 GDN，端到端 forward 需要在相同 batch 内另计一次 cursor、输入 pack 和其他算子。不同历史长度、不同调度参数、图 padding 和 flush 位置会造成预测偏差。

Blue2 上机结果中，混合随机摆放、F≈B/8 时，native / packed Cube + 批量 QKV + Compact V6 的单层 p50 分别为 B32 205/216 µs、B64 377/383 µs、B256 3.71/3.44 ms。实测 GDN stack 均值分别为 9.12/10.01、17.74/18.07、89.02/82.50 ms。F≈B/2 的单层时间则为 199/333 µs、376/541 µs、3.71/5.37 ms。

## 9B TP1 的服务对照与 eager 诊断

同一 910C die，MTP3、BF16、同一短文本输入、请求输出上限 64、`max-num-seqs=256`、`max-num-batched-tokens=8192`。候选服务使用 V24 融合加批量 cursor。下面的 decode 时延是目标模型 graph 的 NPU Event 测值。B128 客户端并发为 128 并重复两轮，B256 客户端并发为 256。

| 客户端并发 | 路径 | 成功请求 | 目标 graph batch | Graph p50/p90（ms） | HTTP p50/p90（s） |
|---:|---|---:|---:|---:|---:|
| 128 | Native | 256 | 128 | 98.77/98.86 | 3.579/3.953 |
| 128 | V24 + 批量 cursor | 256 | 128 | 112.15/135.19 | 4.308/4.454 |
| 256 | V20 + Compact V1 | 256 | 256 | 208.66/309.01 | 8.386/8.472 |
| 256 | V24 + 批量 cursor | 256 | 256 | 200.86/230.33 | 7.590/7.726 |

并发 256 时，native 的最大实际 graph 形状为 B192，因此缺少 B256 同形状目标 forward 对照。V24 采到 22 个有效请求数与 graph 都是 B256 的 forward，平均 flush 比例 18.2%，11 轮只有 0–1 行，5 轮达到 125–160 行。F=0 的 8 轮 p50 为 197.59 ms，F=160 的单轮为 243.70 ms。实际 flush 的历史以 pos13 为主。用这份 F 直方图加权合成网格，packed + Compact V6 的估计平均 GDN 时延与 native 接近，结果对实际 position、摆放及服务 pack 成本敏感。

用 `--compare-eager` 可在同输入、同 flush 布局下补采集 eager NPU Event 和从 host 提交到完成的时间。9B B128、F=16、历史 15/7 的单层结果为 native graph/eager 1.873/1.932 ms、V20+V1 2.922/2.945 ms、V24 2.370/2.388 ms。这组数据中 Replay 的额外时延主要已出现在图内。真实非图 forward 还需检查输入 pack、metadata、采样和其他 host 调用。

### Torch Profiler 的稳态 target forward

9B TP1/MTP3，真实 B128、每次 512 token、无混合 prefill。每条路径从第六次同形状 forward 开始抓 5 次，标记 `REPLAY_STEADY_TARGET/<FULL|NONE>/B128/stepN`。Torch NPU Profiler 使用 CPU/NPU activity、Level1 和 PipeUtilization。用 [汇总脚本](../../../scripts/replayssm/summarize_steady_decode_profile.py) 按标记的起止时刻选取设备任务，同时输出设备 span、活跃区间并集和算子组耗时。四组服务均完成 256 条请求，本表用 trace 内五次 p50，单位 ms。

| 路径 | 目标设备 span | GDR 或 V24（24 次） | IndexCopy / IndexSelect |
|---|---:|---:|---:|
| Native FULL graph | 98.97 | 44.87 | 0 |
| Replay V24 FULL graph | 113.20 | 51.39 | 8.42 |
| Native 强制 eager | 99.39 | 44.86 | 0 |
| Replay V24 强制 eager | 114.06 | 51.97 | 8.16 |

FULL graph 与强制 eager 的 Replay 差距约为 14.2 ms 和 14.7 ms。非强制 eager 的服务里，观测到的纯 decode forward 全部走 FULL 图，NONE 出现在含 prefill 的轮次。不同 run 的 flush 分布可变，不能用这 0.5 ms 的差值确定 eager 的额外影响。

Replay FULL 图里的 120 次 InplaceIndexCopy 共 8.12 ms，另有 24 次 IndexSelect。前者来自 24 个 GDN 层各打包 Q、K、V、gate、beta 的 5 次 scatter，后者按同一索引取回有效输出。Native 直接消费 token-major 张量、请求长度和 state slot。两条路径都可能有投影输出拆分时的连续化，这笔额外 scatter 由 Replay verify 的固定四槽输入布局引起。

### 输入打包消融

对照同一个 V24 核、相同初始 checkpoint、ring、位置与 flush 行，令 T=4，分别直接传入现成视图，或执行服务同款的 5 次 zero + IndexCopy 以及输出 IndexSelect。设备 graph replay p50 如下，stack 各层使用独立 cache，单位 ms。

| 场景 | Flush 行 | Native | V24 直接 view | V24 含 pack/unpack | 打包开销 |
|---|---:|---:|---:|---:|---:|
| 27B TP4 B32，48 层 | 4 | 9.05 | 11.78 | 16.70 | 4.92 |
| 27B TP4 B64，48 层 | 8 | 17.70 | 22.00 | 29.73 | 7.73 |
| 9B TP1 B128，24 层 | 0 | 44.82 | 51.94 | 60.41 | 8.47 |
| 9B TP1 B128，24 层 | 16 | 44.78 | 56.80 | 65.47 | 8.67 |
| 9B TP1 B128，24 层 | 80 | 44.82 | 75.53 | 85.28 | 9.74 |
| 9B TP1 B256，24 层 | 32 | 89.02 | 112.34 | 129.50 | 17.15 |

无 flush 的 9B B128 单层，图内 pack/unpack 多 346 µs，同输入 eager 多 474 µs。纯 decode 的真实图仍承担打包。FULL decode 的 T4 恒等映射适合用 view 省去该步骤，服务接入时必须验证 capture 和 replay 都具备相同布局。长度不齐的路径可以用已存在的请求起始偏移定位 token，相关 verify ABI 需单独验证。

```bash
python3 "$run" --suite vector --geometry 9b --batches 128 --layers 24 \
  --pos 15 --nonflush-pos 7 --base 31 --flush-count 16 --flush-layout random \
  --path native fused_v24 fused_v24_with_pack --compare-eager --verify-library "$verify" \
  --iterations 25 --output /tmp/replay-910c/pack-ablation-b128.json
```

### 紧凑 token 布局的 V24

V24 接受可选的 `token_offsets`。传入时，Q/K/V 和输出为 token-major，
gate/beta 为 `[T, HV]`。同一请求使用
`[token_offsets[row], token_offsets[row + 1])`，每个有效请求有 1–4 个
token。padding 行应使用零长度和空 slot。前缀和以零开始，必须与当轮
输入顺序一致，末项可以小于图上的 token 缓冲区容量。下游只应消费
有效前缀。算子按请求读取 checkpoint、ring slot 与 flush。越界或反向
的局部 span 返回状态码 6，跳过该行输出和 cache 写入。全局单调性和
跨张量顺序由 metadata 生产者保证。

算子在已有 UB 队列中用一次 DMA 读相邻两个 offset，替代原来的 length
读取。按有效 token 数读取 gate/beta、写出结果，直接使用原有的 V24
历史修正和 compact。同一个 `fused_` 接口仍接受未传 offset 的固定槽
调用。当前验证入口是独立 Torch binding，服务接入另需覆盖 FULL 图的
padding、混合 prefill/decode 及连续多轮请求交换。

上机时选择空闲的 910C die。Blue2 测试使用 CANN 9.2.1、MTP3 几何、
tile128、40 核、BF16 checkpoint 以及 FP16 D/K 和 FP32 G。以下命令构建
并对比固定槽、旧 pack/unpack、紧凑三个路径。恢复被 flush 的 checkpoint
在计时区间外。

```bash
cmake -S experimental/replayssm_sd_ascendc -B build-verify \
  -DSOC_VERSION=ascend910_9362 -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build-verify -j 8
run=experimental/replayssm_cube_910b/run_token_layout.py
verify=build-verify/libreplayssm_sd_torch.so
python3 "$run" --geometry 27b --batches 4 --layout ragged --flush-fraction 0.25 \
  --verify-library "$verify" --output /tmp/replay-layout-ragged.json
python3 "$run" --geometry 9b --batches 4 --layout padded --flush-fraction 0.25 \
  --verify-library "$verify" --output /tmp/replay-layout-padded.json
python3 "$run" --geometry 27b --batches 32,64 --layers 48 --layout uniform \
  --iterations 25 --verify-library "$verify" --output /tmp/replay-layout-27b.json
python3 "$run" --geometry 9b --batches 128,256 --layers 24 --layout uniform \
  --iterations 25 --verify-library "$verify" --output /tmp/replay-layout-9b.json
```

把最后两个命令的 `--layout` 改为 `ragged`，输出到不同文件，可复测
每行长 1–4 的混合长度。B4 ragged 会在 capture 后变更请求边界，并验证
超界的末行在同一个图中只写错误状态。B4 padded 还检查空行和保留容量，
确认有效前缀以外的输出没有被覆写。比较包含有效输出、D/K/G ring、
checkpoint、两组状态，且 checkpoint 与 ring 的物理槽号不同。

Blue2 图回放 p50，单位 ms。stack 各层使用独立 cache。位置为 flush
行 15、其余有效行 7，ring base 为 31。Native 用现有对照脚本在同一
die 上分别重新测量，各有效 token 的 state slot 随 token 排列。

| 几何 | 布局 | B | Flush 行 | Native | V24 固定槽 | V24 含 pack/unpack | V24 紧凑 |
|---|---|---:|---:|---:|---:|---:|---:|
| 27B TP4，48 层 | T4 | 32 | 4 | 9.07 | 11.90 | 16.68 | 11.85 |
| 27B TP4，48 层 | T4 | 64 | 8 | 17.80 | 22.26 | 29.45 | 22.05 |
| 9B TP1，24 层 | T4 | 128 | 0 | 44.77 | 51.72 | 61.11 | 51.61 |
| 9B TP1，24 层 | T4 | 128 | 16 | 44.77 | 57.15 | 66.16 | 56.97 |
| 9B TP1，24 层 | T4 | 128 | 80 | 44.86 | 76.14 | 84.89 | 75.52 |
| 9B TP1，24 层 | T4 | 256 | 32 | 88.99 | 112.84 | 129.99 | 112.62 |
| 27B TP4，48 层 | 混合 T1–4 | 32 | 4 | — | 9.14 | 13.41 | 9.11 |
| 27B TP4，48 层 | 混合 T1–4 | 64 | 8 | — | 17.46 | 23.10 | 17.31 |
| 9B TP1，24 层 | 混合 T1–4 | 128 | 16 | — | 43.74 | 49.81 | 43.07 |
| 9B TP1，24 层 | 混合 T1–4 | 256 | 32 | — | 85.98 | 97.71 | 84.88 |

同一布局内，紧凑路径和预先排好的固定槽路径相差约 0.2%–1.5%。
每行 T4 已可用零拷贝视图作为对照。紧凑 ABI 的主要收益是允许混合
长度和 padding 也省掉输入 scatter 和输出 gather。混合长度的绝对时延
还受有效 token 数影响。Native 的 T4 测值可用于同形状差距，混合长度
应另测对应 native 下发方式。上述测试只包含 GDN 算子图。

### V25 在 flush 行先折叠 checkpoint

V25 沿用紧凑 V24 的单次 launch、tile128、40 核、双 slot 和候选 ring
地址。有效 flush 行先在同一 FP32 UB tile 折叠已提交的历史，再用
该状态做候选 Q/K 投影，跳过逐候选的历史点积和加权归约。候选输出
后仍然只写回一次 BF16 checkpoint。没有增加 UB buffer 或写入额外
checkpoint。V25 通过 `variant=25` 和 `token_offsets` 显式选择。

构建方式与上一节相同。以下是窄烟测及图回放矩阵，需在空闲 die 上运行。

```bash
run=experimental/replayssm_cube_910b/run_token_layout.py
verify=build-verify/libreplayssm_sd_torch.so
python3 "$run" --geometry 27b --batches 4,32 --layout ragged --flush-fraction .5 \
  --iterations 15 --path compact fold_first --verify-library "$verify" \
  --output /tmp/replay-fold-first-smoke.json
python3 "$run" --geometry 9b --batches 4,32 --layout padded --flush-fraction .5 \
  --iterations 15 --path compact fold_first --verify-library "$verify" \
  --output /tmp/replay-fold-first-padded.json
for fraction in 0 .125 .625 1; do
  python3 "$run" --geometry 27b --batches 32,64 --layers 48 --layout uniform \
    --flush-fraction "$fraction" --path compact fold_first --iterations 35 \
    --verify-library "$verify" --output "/tmp/replay-fold-first-27b-$fraction.json"
  python3 "$run" --geometry 9b --batches 128,256 --layers 24 --layout uniform \
    --flush-fraction "$fraction" --path compact fold_first --iterations 35 \
    --verify-library "$verify" --output "/tmp/replay-fold-first-9b-$fraction.json"
done
```

精度对照以相同初始状态的紧凑 V24 为参照。非 flush 行、已提交的 D
历史、K/G ring、两组状态码与 checkpoint 要求逐元素一致。flush 行
的候选输出和 D 可因浮点计算顺序变化而不同，容差为
`atol=2e-5, rtol=0.008`，JSON 同时记录最大绝对差。混合长度烟测
在图捕获后重写 offset，并检查无效跨度的状态和 cache 写入保护。

Blue2 单 die，CANN 9.2.1。表内每格为 V24 → V25 的 GDN stack
图回放 p50，单位 ms。每个 case 35 次，非 flush 行 pos7，flush 行
pos15，base31，T4，checkpoint 重置在计时区间外。

| 场景 | 0% flush | 12.5% flush | 62.5% flush | 100% flush |
|---|---:|---:|---:|---:|
| 27B TP4，B32，48 层 | 10.57 → 10.64 | 11.87 → 11.47 | 15.34 → 13.92 | 17.61 → 15.29 |
| 27B TP4，B64，48 层 | 20.05 → 20.37 | 22.03 → 21.72 | 28.52 → 26.03 | 34.25 → 29.76 |
| 9B TP1，B128，24 层 | 51.12 → 52.14 | 56.63 → 55.83 | 75.78 → 68.55 | 87.63 → 76.05 |
| 9B TP1，B256，24 层 | 102.05 → 103.29 | 111.69 → 110.31 | 148.01 → 135.24 | 173.99 → 151.77 |

V25 在全非 flush 慢 0.7%–2.0%，约 1/8 flush 时快 1.2%–3.3%，
全 flush 快 12.8%–13.2%。最大观测误差为输出 1.53e-5、候选 D
6.10e-5，checkpoint 与 V24 逐元素一致。服务 dispatch 保持现有
版本，V25 的端到端收益取决于真实 flush 分布、延迟尾部和多轮精度。

GPU 的两个 kernel 分别负责非 flush 和 flush，均能独立完成各自行
的 verify。这样可分别调 tile 和核数，代价是固定图多一个设备 launch，
并重复读路由元数据。V25 在一个 kernel 内复用 UB，但执行资源和任务
切分需要兼容两种行。上述数字比较的是两版合一 kernel 的算法顺序，
双 kernel 的 Ascend 性能需用同算法、同 flush 行分布另测。

### 9B TP1 的紧凑 V24/V25 服务对照

Blue2 910C 单 die、Qwen3.5-9B BF16、MTP3，
`--max-num-seqs 256 --max-num-batched-tokens 8192`，显存利用率
0.85，异步调度关闭。三路用同一份短文本，输出上限 64 token。
Replay 在隔离服务副本中通过实验 binding 调用紧凑算子，保持原有
双槽和批量 cursor，跳过固定槽 pack/unpack。两版紧凑路径都预先
清零输出，以定义图容量中的 padding。HTTP 吞吐为实际输出 token
除以整次客户端测试时长，包含客户端进程和指标读取消耗。

| 客户端并发 | 路径 | 成功请求 | 输出 tok/s | HTTP p50/p90 (s) | 观测峰值 waiting |
|---:|---|---:|---:|---:|---:|
| 128 | Native | 256/256 | 2,160 | 3.625 / 3.917 | 0 |
| 128 | 紧凑 V24 | 256/256 | 2,030 | 3.901 / 4.024 | 0 |
| 128 | 紧凑 V25 | 256/256 | 2,008 | 3.861 / 4.075 | 0 |
| 256 | Native | 512/512 | 2,078 | 5.097 / 9.568 | 89 |
| 256 | 紧凑 V24 | 512/512 | 2,295 | 6.928 / 7.033 | 0 |
| 256 | 紧凑 V25 | 512/512 | 2,317 | 6.895 / 7.015 | 0 |

| 目标 forward 的 graph batch | Native p50/p90 (ms) | 紧凑 V24 (ms) | 紧凑 V25 (ms) |
|---:|---:|---:|---:|
| B128 | 98.62 / 98.68 | 103.50 / 122.79 | 106.92 / 114.50 |
| 最大实测 graph | B192，132.26 / 132.35 | B256，190.89 / 215.87 | B256，191.91 / 207.77 |

目标 forward 以日志记录的 graph batch 归组，计时使用 NPU Event，
覆盖服务完成的整个 target graph。Replay 日志额外记录了 flush 元数据，
位于设备计时区间外。C256 的 Native 和 Replay 目标图实际形状不同。
Native 的估计可用 KV 为 387,873 token，V24/V25 均为 585,728 token。
C256 的 V25 相比 Native 输出吞吐增加 11.5%，HTTP p90 降低 26.7%，
HTTP p50 增加 35.3%。C128 无排队，此时 V25 相比 Native 的目标图
p50 慢 8.4%，输出吞吐低 7.0%。V25 对 V24 在 C256 的输出吞吐约
增加 1%，在 C128 略降，这组服务数据不足以宣布默认切换 V25。
27B TP4 本轮仅覆盖单 die 算子几何，未得到四 die 的服务对照。

### GPU 计算组织的三项消融

用同一批输入对比 native GDR、单 launch V25、以及此前低 flush 的
最快混合基线 `mixed_packed_compact_v6 --mixed-qkv`。混合基线
用 Cube 做 checkpoint 的 Q/K 投影，AIV 分别处理两个 tile64，再调用
Compact V6。新变体复用同一 Cube 投影、核数、切分和缓存。

- `mixed_history_panel` 将历史 D 按后缀衰减加权，一次扩展活跃
  D 行，形成每个历史行的八列乘积，并树归约到投影。辅助对照
  `mixed_suffix_reference` 使用同样的 D 后缀衰减和 QKV 批量加载，
  但仍逐行更新投影。批量归约额外占用 32 KiB AIV UB。
- `mixed_triangular` 仍使用基线历史修正。一次 Gather 将八列
  checkpoint + 历史投影改为列连续。预备衰减系数后在 AIV 展开
  四步前代，减少整块 panel 的更新。候选 D 仍在 FP32 UB 内参与
  后续候选计算，写 ring 时再转 FP16。
- `cube_fold_gemm` 测已经备好的 FP16 `[128,16]×[16,128]`。
  `cube_fold_path` 一并计入 ring 读取、FP32 衰减、转 FP16 和
  布局调整、BMM 结果回到 Vector 路径、checkpoint 写回。
  `mixed_cube_fold` 用该链路替换基线的 Compact V6。该探针
  通过 Torch NPU BMM 连接独立算子，已知 active rows 在计时前准备。
  生产实现若在一个混合核内交接，需要独立测量。

在空闲 910C die 上编译混合核。Compact V6 与 V25 的实验库按前面的
清单构建。使用与服务一致的 vendor 算子库。

```bash
cmake -S experimental/replayssm_cube_910b/mixed_verify -B build-mix \
  -DSOC_VERSION=ascend910_9362 -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build-mix -j 6
run=experimental/replayssm_cube_910b/run.py
python3 "$run" --suite mixed --geometry 9b --batches 256 --layers 24 \
  --pos 16 --nonflush-pos 7 --base 31 --flush-count 32 --flush-layout random \
  --mixed-qkv --mixed-library build-mix/libreplayssm_sd_mix_torch.so \
  --compact-library "$compact" --verify-library "$verify" --iterations 15 \
  --path native fused_v25 mixed_packed_compact_v6 mixed_suffix_reference \
      mixed_history_panel mixed_triangular mixed_cube_fold \
  --output /tmp/replay-mixed-ablations-stack.json
for layout in prefix spread random; do
  python3 "$run" --suite mixed --geometry 9b --batches 256 \
    --pos 16 --nonflush-pos 7 --base 31 --flush-count 128 --flush-layout "$layout" \
    --mixed-qkv --mixed-library build-mix/libreplayssm_sd_mix_torch.so \
    --compact-library "$compact" --verify-library "$verify" --iterations 30 \
    --path native fused_v25 mixed_packed_compact_v6 mixed_suffix_reference \
        mixed_history_panel mixed_triangular compact_v6 cube_fold_gemm cube_fold_path mixed_cube_fold \
    --output "/tmp/replay-mixed-ablations-$layout.json"
done
```

每条混合路径会在 eager 和图回放中核对 Cube 投影、候选输出、
D/K/G 和状态。折叠结果与向量 compact 比较，测试覆盖 T1–T4、
跨 ring 尾及独立 checkpoint/ring 槽。

Blue2 910C 单 die、CANN 9.2.1、torch-npu 2.10.0.post7、BF16 checkpoint
和 FP16 D/K、FP32 G。Cube 投影最多 20 个 AIC，配对 40 个 AIV。
Compact V6 tile64，V25 tile128。以下是随机摆放 12.5% flush 的 stack
图回放 p50，单位 ms。非 flush pos7，flush pos16，base31，T4。
每组回放 15 次，定位到 `c5715bf` 上这一提交的混合核改动。

| 几何和 batch | Native | V25 | Cur best 混合 + V6 | 同衰减逐行参考 | 整窗历史 | 四步前代 | 混合 + Cube 折叠 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 27B TP4 B32，48 层 | 9.09 | 11.49 | 10.15 | 10.12 | 10.02 | 10.51 | 19.69 |
| 27B TP4 B64，48 层 | 17.81 | 22.13 | 17.91 | 17.68 | 17.40 | 18.62 | 28.07 |
| 9B TP1 B128，24 层 | 44.78 | 55.83 | 42.48 | 42.03 | 41.00 | 44.12 | 51.83 |
| 9B TP1 B256，24 层 | 89.01 | 111.00 | 82.84 | 81.85 | 79.92 | 86.36 | 108.49 |

整窗历史修正在四组都快于 Cur best，降幅约 1.2%–3.6%。同样使用
后缀衰减的逐行对照也慢于整窗实现，说明收益既来自后缀加权，也来自
批量扩展和归约。高并发 9B 的收益最大。全非 flush 的 9B B256
单层从 2.770 ms 降到 2.668 ms，native 为 3.711 ms。27B B32
在 12.5% flush 时仍慢于 native 约 10.3%，虽然全非 flush 的整窗
实现为 0.164 ms，快于 native 0.204 ms。这个形状应优先定位低 flush
时的 compact 固定开销和尾部核负载。

四步前代把每候选两次 Gather 改为每窗一次，并通过了 27B 和 9B
的 T1–T4 校验。代表性的 T4 stack 比 Cur best 慢 3.6%–4.3%。
它减少了 Gather，但同时改变了每个候选的向量累加和系数准备方式，
当前数据不能把回退归因到单条指令。暂保留为消融对照。

以下为每组 50% flush 时，prefix、spread、random 三种摆放的单层图
p50 范围，单位 ms。每 case 30 次。各列范围用于展示摆放敏感度。

| 几何和 batch | Native | Cur best | 整窗历史 | V25 | 裸 Cube BMM | AIV Compact V6 | Cube 完整折叠 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 27B TP4 B32 | 0.197–0.205 | 0.293–0.337 | 0.285–0.329 | 0.276–0.295 | 0.034–0.041 | 0.132–0.165 | 0.411–0.434 |
| 27B TP4 B64 | 0.376–0.379 | 0.530–0.595 | 0.511–0.579 | 0.529–0.562 | 0.040–0.057 | 0.227–0.293 | 0.599–0.620 |
| 9B TP1 B128 | 1.872–1.875 | 2.733–2.806 | 2.630–2.707 | 2.730–2.755 | 0.054–0.060 | 1.124–1.187 | 2.949–3.022 |
| 9B TP1 B256 | 3.710–3.712 | 5.437–5.503 | 5.242–5.301 | 5.396–5.472 | 0.151–0.152 | 2.265–2.323 | 6.063–6.737 |

固定 flush 数后，27B B32 的整窗历史仍可因摆放浮动约 44 µs。
9B B256 半数 flush 的最好结果仍比 native 慢约 41%。单独的
FP16 BMM 在小探针中由 `msprof` 确认为 `BatchMatMulV2` 的
`KERNEL_AICORE`，replay 中该任务约 13 µs。9B B256、F128 的
裸 BMM p50 约 0.15 ms，整个折叠链路却要 6.06–6.74 ms，高于 AIV
Compact V6 的 2.27–2.32 ms。该结果说明现有 GM 物化方式的整体
开销大，尚未测量同一个混合 kernel 内的折叠与投影复用。Cube 路径
checkpoint 相对向量参考的最大绝对差为 4.88e-4。

主矩阵 28 组均通过，相关 CPU 单测 15 项通过。整窗历史变体可作为
下一轮低 flush 的候选，仍需接入服务测整体收益。高 flush 应先
定位折叠的数据流和尾部核负载，避免依据裸 Cube 吞吐选型。

27B TP4 B32 的后续探针固定随机 F4、pos16/7、T4，45 次回放。
20 Cube、按 key-head 分组、Compact V6 tile64 仍是该组最快组合，
单层 219.98 µs，native 201.52 µs。只改 Compact 到 tile32 后，
Compact 从 62.36 增至 78.32 µs，组合增至 234.54 µs。开启按 V
head 均分后，tile64 组合为 224.24 µs，进一步减到 16 Cube 为
246.04 µs。无 flush 的单独 Compact 图约 48.44 µs，其中包括
图回放固定开销，不能直接从组合时延中扣减。这组数据不支持继续
用这些 tile、核数或任务均分配置消除剩余差距。

### 混合 verify 内融合 AIV checkpoint 折叠

`mixed_fused_fold` 以整窗历史修正为基线。AIC 仍按 key-head 分组
计算 checkpoint 投影。两个 AIV 各拥有 64 行，收到 READY 并
读完投影后，仅对 flush 行预取自己的 BF16 checkpoint tile。
候选输出完成后，用 UB 内已有的后缀加权 D、历史 K 和总衰减折叠
checkpoint。计算为 FP32，写回为 BF16。折叠复用整窗历史的乘积
缓冲区。非 flush 和 pos0 均跳过 checkpoint 搬入和写回。
实验算子在图内提交一次。Torch schema 标记了 checkpoint mutation，
旧实验 variant 仍可按原接口调用。

用上一节的混合核编译方式，测试时加入 `mixed_fused_fold`。
例如复现 27B B32 的随机轻 flush 对照。

```bash
python3 experimental/replayssm_cube_910b/run.py --suite mixed --geometry 27b \
  --batches 32 --layers 48 --pos 16 --nonflush-pos 7 --base 31 \
  --flush-count 4 --flush-layout random --mixed-qkv \
  --path native mixed_history_panel mixed_fused_fold fused_v25 \
  --mixed-library build-mix/libreplayssm_sd_mix_torch.so \
  --compact-library "$compact" --verify-library "$verify" --iterations 20 \
  --output /tmp/replay-mixed-fused-fold.json
```

Blue2 910C 单 die，CANN 9.2.1，torch-npu 2.10.0.post7。
基于 `d727cf8` 的本次实验。20 AIC、40 AIV，T4、base31，
非 flush pos7，flush pos16。旧整窗组合用 Compact V6 tile64。
以下为随机 12.5% flush 的 GDN stack 图回放 p50，单位 ms。

| 几何和 batch | Native | 整窗历史 + Compact V6 | 整窗历史内融合折叠 | V25 |
|---|---:|---:|---:|---:|
| 27B TP4 B32，48 层 | 9.064 | 9.955 | 9.082 | 11.592 |
| 27B TP4 B64，48 层 | 17.741 | 17.781 | 16.462 | 21.922 |
| 9B TP1 B128，24 层 | 44.790 | 41.458 | 38.849 | 56.078 |
| 9B TP1 B256，24 层 | 88.958 | 80.459 | 75.801 | 110.734 |

27B B32 F4 的独立 msprof 采样用同一形状和布局。末尾 35 次
任务的中位时延中，独立混合 verify 为 149.22 µs，Compact V6 为
54.12 µs。融合的混合任务为 185.32 µs，图中没有独立的 Compact
任务。这个结果与 stack 图回放少约 0.87 ms 的方向一致。

| 几何和 batch | 50% flush 的摆放 | Native | 整窗历史 + V6 | 融合 | V25 |
|---|---|---:|---:|---:|---:|
| 27B TP4 B32 | prefix / random | 204 / 198 | 297 / 332 | 279 / 283 | 288 / 298 |
| 27B TP4 B64 | prefix / random | 376 / 376 | 519 / 542 | 474 / 494 | 531 / 537 |
| 9B TP1 B128 | prefix / random | 1874 / 1874 | 2657 / 2729 | 2365 / 2427 | 2733 / 2772 |
| 9B TP1 B256 | prefix / random | 3710 / 3710 | 5265 / 5323 | 4657 / 4724 | 5420 / 5471 |

半 flush 表为单层图 p50，单位 µs。融合改善了重路径和摆放造成的
波动，仍显著慢于 native。27B 两个相邻 batch 在同样 spread F2 下，
B16/B18 分别为融合 137/146 µs，native 108/125 µs。这个范围仍
比 native 慢，B16 到 B18 的相对差距没有扩大。当前证据支持低
flush 的融合路径，但单 die 的算子图无法验证真实服务收益。

本轮 34 份设备报告和 15 项 CPU 单测通过。B4 精度专项覆盖
T1–T4、pos0 和跨环尾的 pos16。捕获后原地将 flush 清零再回放，
候选输出仍满足同一参考，两个 checkpoint tile 都保持原始值。
专项最大绝对差为 checkpoint 2.44e-4、候选输出 1.53e-5。
服务接入的当前依赖是让混合算子消费紧凑 token offset 和实际
checkpoint page stride，随后复测目标 forward 和 HTTP。

### 混合融合核直接消费服务布局

实验 binding 的可选 `token_offsets` 接收设备端 int32 前缀和。
传入后 Q/K/V 及输出使用 `[tokens, heads, 128]`，G/beta 使用
`[tokens, heads]`。AIC 按行读取实际 Q/K 窗口，分别填充 Cube
投影中未使用的 K/Q 列。AIV 用同一窗口读 V、G、beta 并写回紧凑
输出，保留物理容量中未使用的尾部。故意破坏 offset 时，仅写
该行错误状态，避免对其输出、checkpoint 或 ring 写入。旧的
固定槽调用仍可省略 offset。

Checkpoint 的 slot stride 由 Tensor 传到 AIC 和 AIV。每页内部
仍为连续的 BF16 `[heads, 128, 128]`，请求 ring 继续用独立槽号。
仅 flush 且历史非空时，AIV 折叠并写回自己负责的 64 行。
这条路径可直接使用服务的 checkpoint view 和紧凑 token，省掉
固定槽 scatter/gather 及 checkpoint 的重新物化。

在 Blue2 单 die、CANN 9.2.1 上重建混合实验库，使用上一节
编译命令。如下命令用同一混合算子的固定槽与连续 checkpoint
作为精度对照，同时给出紧凑 V25 的图回放时延对照。

```bash
python3 experimental/replayssm_cube_910b/run_token_layout.py \
  --geometry 27b --batches 4,8 --layout ragged --flush-fraction .5 \
  --path fold_first mixed_fold --mixed-balanced --iterations 12 --verify-library "$verify" \
  --mixed-library build-mix/libreplayssm_sd_mix_torch.so \
  --output /tmp/replay-mixed-packed.json
```

27B TP4 几何及 9B TP1 均运行 B4/B8、ragged/padded、50% flush。
核对候选输出、checkpoint、D/K/G ring 和 status，合计 8 组
混合融合布局对照全部逐元素一致。图捕获后变更 offset，输出
按新边界更新。无效跨度和图容量尾部的写入保护通过。CPU 单测
15 项通过。该结果验证布局变更保持混合核的语义。

9B TP1 MTP3 的真实服务对照使用同一 910C die、同一短文本及
输出上限 64 token。最大请求数 256，最大批处理 token 数 8192，
最大模型长度 4096，显存利用率 0.85，BF16 模型、卷积缓存和
checkpoint，`FULL_DECODE_ONLY`。混合核通过隔离服务副本的
实验 binding 调用，使用 20 AIC、40 AIV、按 V head 均分、每层
预分配投影用 scratch。两条 Replay 路径复用同一批量 cursor 和
紧凑输入处理，图输出的 padding 均预先清零。

| 客户端并发 | 路径 | 成功请求 | 输出 tok/s | HTTP p50 / p90 (s) | 峰值 waiting |
|---:|---|---:|---:|---:|---:|
| 128 | Native | 256/256 | 2179 | 3.598 / 3.906 | 0 |
| 128 | 紧凑 V25 | 256/256 | 2008 | 3.857 / 4.070 | 0 |
| 128 | 紧凑混合融合 | 256/256 | 1965 | 3.992 / 4.154 | 0 |
| 256 | Native | 512/512 | 2105 | 5.115 / 9.554 | 89 |
| 256 | 紧凑 V25 | 512/512 | 2318 | 6.866 / 7.038 | 0 |
| 256 | 紧凑混合融合 | 512/512 | 2298 | 6.981 / 7.087 | 0 |

| 目标图 batch | Native p50 / p90 (ms) | 紧凑 V25 (ms) | 紧凑混合融合 (ms) |
|---:|---:|---:|---:|
| B128 | 98.84 / 98.89 | 106.68 / 114.26 | 122.71 / 130.66 |
| B192 | 132.43 / 132.60 | 140.54 / 150.66 (2 次) | — |
| B256 | — | 191.57 / 208.26 | 222.90 / 236.05 |

目标 forward 使用 NPU Event，设备计时结束后读取本步 flush 行。
HTTP 表包含启动客户端与获取指标的墙钟时间。Native 最大实测
目标图为 B192，不能对 B256 图做同形状比较。KV 容量 Native
387,873 token，Replay 两组分别 586,020 和 585,728。C256 的
混合路径相对 Native 输出吞吐增加 9.2%，HTTP p90 降低 25.8%，
与同轮 V25 比较仍略慢。服务请求通过与 Native 逐 token 等价
是两个门槛，该轮只以请求完成为服务验收，布局的逐元素验收
来自固定槽混合核探针。多模态、长输出和 27B TP4 服务未覆盖。

混合方案的非 flush 优势能在服务中复现。在正好有 128 或 256
个活跃 slot 的图中，按 flush 行比例汇总，单元格为
目标图 p50 毫秒（样本数）。

| Batch | 路径 | 0% | (0%, 12.5%] | (12.5%, 50%] | >50% |
|---:|---|---:|---:|---:|---:|
| B128 | V25 | 103.22 (10) | 105.35 (11) | 111.66 (17) | 114.33 (3) |
| B128 | 混合融合 | 83.88 (12) | 116.25 (8) | 123.18 (19) | 123.35 (3) |
| B256 | V25 | 185.35 (8) | 186.08 (11) | 197.09 (17) | 211.27 (5) |
| B256 | 混合融合 | 143.69 (7) | 189.41 (14) | 223.63 (17) | 243.93 (1) |

相同短 prompt 会产生较集中的 flush，B256 的单步 flush 在
混合核达到 139 行，在 V25 达到 166 行。这个分桶支持优先拆查
混合核的有 flush 步，同时还混有历史长度、候选长度与行摆放
差异。不能从分桶直接估算单个 flush 行代价。V25 保留为当前
紧凑服务对照，混合融合继续留在实验路径。

板上原始报告在 `mixed-packed.FmrhIp/results`，服务日志和 HTTP
结果在 `serve-mixed-packed.GHPMH4/results`。两者位于 Blue2
`/home/lmt_infer/replayssm` 下。混合实验库 SHA-256 为
`c9ca2c7cd3915702a9cf8aba85e1740a33416b2b338cfe136b3a539261ca47d0`。

同一库另测了按 key-head 整组轮转，和允许切开 key-head 的连续
value-head 均分。输入为 ragged T1–T4，随机 flush，pos15/7、
base31，单层 graph p50。均分 35 次，整组与 V25 各 25 次。
以下为固定约 12.5% flush 的结果，单位 µs。

| 场景 | flush 行数 | V25 | 混合均分 | 混合整组轮转 |
|---|---:|---:|---:|---:|
| 27B TP4 B16 | 2 | 120.12 | 183.00 | 106.08 |
| 27B TP4 B18 | 2 | 123.90 | 169.90 | 121.38 |
| 27B TP4 B32 | 4 | 188.60 | 231.00 | 173.64 |
| 27B TP4 B64 | 8 | 350.16 | 348.50 | 305.22 |
| 9B TP1 B128 | 16 | 1750.40 | 2092.90 | 1346.52 |
| 9B TP1 B256 | 32 | 3463.96 | 3283.06 | 2615.42 |

整组轮转相比均分，在 9B B128/B256 降低 35.7%/20.3%。
在 50% flush 的 B128/B256，整组为 2096.92/4054.50 µs，
均分为 2491.06/5010.42 µs。27B B32 半数 flush 仍比 V25
慢 8.6%，说明改切分有帮助，重路径还有差距。

整组轮转对照覆盖 27B 12 个 batch（B1–B64）和 9B 16 个
batch（B1–B256），各测 0%、约 12.5%、50% flush，均通过。
约 12.5% flush 时，27B B30/32/34 为 149.18/173.64/181.70 µs，
其中 B30→B32 同时从 3 行 flush 增到 4 行。9B B126/128/130
为 1304.78/1346.52/1340.22 µs，B254/256
为 2600.62/2615.42 µs。关键边界的相对变化应在服务侧继续看
实际 flush 行摆放及请求长短混合。旧版的固定槽、紧凑 V24/V25
精度探针一并回归通过。以上是同一 die 上的算子图对照。

保持服务代码、实验库、配置和数据相同，只关闭混合核的
`balance_heads` 后再跑 9B TP1。该模式按 key-head 整组轮转，
同一 key head 的 2 个 value head 共享 Q/K 投影输入和点积，
也让同一个 Cube 分到的请求行更分散。服务仍预分配相同 scratch，
使用相同的输出清零和 cursor。

| 客户端并发 | Native tok/s | V25 tok/s | 混合均分 tok/s | 混合整组轮转 tok/s | 整组 HTTP p50 / p90 (s) |
|---:|---:|---:|---:|---:|---:|
| 8 | 331 | 325 | 296 | 331 | 1.379 / 1.425 |
| 128 | 2179 | 2008 | 1965 | 2211 | 3.468 / 3.788 |
| 256 | 2105 | 2318 | 2298 | 2632 | 6.043 / 6.194 |

整组轮转三组请求全部通过，峰值 waiting 为 0，KV 为 585,728
token。C256 的吞吐相对 Native 和 V25 分别增加约 25.0% 和
13.5%。B128 target graph p50/p90 为 88.56/103.35 ms，
Native 为 98.84/98.89 ms。整组轮转的 p50 比 Native 低 10.4%，
p90 仍高 4.5%。B256 target graph p50/p90 为 152.38/182.75 ms，
比 V25 的 191.57/208.26 ms 分别低 20.5% 和 12.2%。Native 的
B256 target graph 未出现，因为该组请求部分排队。

活跃行全部填满图时，三档 flush 区间的 p50 如下，单位 ms。
每档历史和候选长度仍由服务产生。

| Batch | 路径 | 0% | (0%, 12.5%] | (12.5%, 50%] |
|---:|---|---:|---:|---:|
| B128 | 混合均分 | 83.88 | 116.25 | 123.18 |
| B128 | 混合整组轮转 | 81.53 | 86.56 | 94.82 |
| B256 | 混合均分 | 143.69 | 189.41 | 223.63 |
| B256 | 混合整组轮转 | 138.53 | 149.33 | 172.29 |

两种混合切分在这些区间的样本数相同。实测收益主要出现在
flush 步，说明保持 Q/K 复用和改变任务分布比无条件提高
value-head 数量均衡更有效。数据支持整组轮转作为当前
9B 短文本服务候选。更长输出、多模态和 27B 四 die 服务
还需独立确认，生产 dispatch 仍由现有实现决定。

用 Torch NPU Profiler 对相同短文本 C128 抓 steady decode，
两个实验服务各记录实际 B128 的第 6–10 步。设备任务通过
`REPLAY_STEADY_TARGET/FULL/B128/stepN` 对齐。下表每格为
设备 forward span / 24 层 GDN kernel 合计，单位 ms。

| 步 | Flush 行 | V25 | 混合整组轮转 |
|---:|---:|---:|---:|
| 6 | 3 | 104.22 / 51.34 | 82.17 / 29.31 |
| 7 | 0 | 104.23 / 51.42 | 84.51 / 31.70 |
| 8 | 1 | 105.29 / 52.49 | 88.88 / 36.09 |
| 9 | 124 | 123.75 / 69.97 | 123.10 / 69.34 |
| 10 | 3 | 104.30 / 51.45 | 81.40 / 28.54 |

两路每步均有 153 个 matmul 任务和 8 个 gather/scatter 任务，
各类合计分别约 42 和 0.12 ms。轻 flush 步的 target 变化
基本与 GDN 合计变化一致。124/128 行同时 flush 时，两路接近，
整组轮转的轻路径收益基本消失。这个真实重步为后续 flush
折叠提供了针对性回归点。原始 trace 及汇总在同一服务副本的
`profile-steady/graph-{compact-v25,mixed-grouped}` 下。

### 重 flush 的向量乘加消融

`mixed_fold_fma`（`aiv_ablation=5`）沿用整组轮转和 FP32
checkpoint 折叠。只将每个历史行的两条跨行 `Mul`、一次
`Add` 和中间 Vector 屏障替换成两条 `MulAddDst`。
历史 D 广播、输出和 ring 更新顺序相同。对照可用同一个实验库：

```bash
python3 experimental/replayssm_cube_910b/run.py --suite mixed --geometry 9b \
  --batches 128 --length 4 --pos 16 --nonflush-pos 7 --base 31 \
  --flush-count 124 --flush-layout random --iterations 25 \
  --path native fused_v25 mixed_fused_fold mixed_fold_fma \
  --mixed-library "$mix" --verify-library "$verify" --compact-library "$compact" \
  --output /tmp/sd-mixed-fold-fma.json
```

Blue2 Ascend910_9362 die 13，CANN 9.2.1、torch-npu 2.10.0.post7。
单层、MTP3 的 T4、随机 flush，其他行 pos7。两种混合核 B4
精度和图回放均通过。下表为同 die、同输入的 25 次回放 p50，单位 µs。

| 场景 | Native | V25 | 原混合折叠 | 混合乘加 |
|---|---:|---:|---:|---:|
| 27B TP4 B32 F31 | 201.08 | 326.00 | 358.70 | 400.04 |
| 9B TP1 B128 F124 | 1871.70 | 3261.34 | 3347.60 | 3732.22 |
| 9B TP1 B256 F128 | 3710.48 | 5460.20 | 4737.20 | 5144.48 |

乘加变体三个场景比原混合折叠慢 8.6%–11.5%，保留作消融，
服务继续使用原混合折叠。减少 API 数量和一处屏障在这个跨行
步长及累加依赖下未能转化成时延收益。此组是单 die 算子数据，
测试时同卡另一 die 在跑长文本对照。各路径对照在同一 die 内
完成。原始报告在 `mixed-packed.FmrhIp/results/fold-fma-early`。

后续在主测试 die 12 上顺序运行 28 份报告，覆盖两种几何的
ragged/padded、T1–T4、随机 F=0、B/8、B/2 和接近满 flush，
精度与图回放全部通过。在 12 组非零 flush 的性能对照中，
乘加全部慢于目前的混合折叠。27B B32/B64 近满 flush 从
358.2/642.7 增至 399.0/718.3 µs。9B B128/B256 近满
flush 从 3339.4/6595.4 增至 3724.9/7364.5 µs，约慢
11%–12%。0 flush 四组的差距在 ±2% 内。每组 40 次
同图回放。这轮避免了相邻 die 的服务负载，复现了重路径负向证据。
完整矩阵在 `mixed-packed.FmrhIp/results/fold-fma`。

### 9B 长文本服务对照

同一 Blue2 910C die 12，9B TP1、MTP3、CANN 9.2.1，
`FULL_DECODE_ONLY`，BF16 模型及 checkpoint。每组发一轮，
16 个事先校验为 4000 token 的纯文本 prompt 轮流提交，每请求
输出 400 token。最大模型长度 4608，`max-num-batched-tokens=8192`，
`max-num-seqs=256`，显存利用率 0.85。以下是端到端 HTTP 结果。

| 客户端并发 | 路径 | 成功 | 输出 tok/s | 请求 p50 / p90 (s) |
|---:|---|---:|---:|---:|
| 128 | Native | 128/128 | 677.4 | 67.609 / 74.914 |
| 128 | 紧凑 V25 | 128/128 | 711.4 | 69.072 / 71.150 |
| 128 | 混合整组轮转 | 128/128 | 731.1 | 67.293 / 69.420 |
| 256 | Native | 256/256 | 700.5 | 115.845 / 145.176 |
| 256 | 紧凑 V25 | 256/256 | 707.3 | 129.349 / 143.472 |
| 256 | 混合整组轮转 | 256/256 | 725.2 | 126.651 / 139.708 |

混合整组相对 Native，C128/C256 吞吐增加 7.9%/3.5%，
请求 p90 降低 7.3%/3.8%。C256 p50 增加 9.3%。KV 容量为
Native 418,176、V25 615,321、混合整组 615,014 token。
日志采样的峰值运行请求为 93 / 139 / 136。三路的入口峰值
waiting 都为 C128 的 122、C256 的 246。长 prefill 后加入
decode 的实际批形不同，不能用这个峰值判断 decode 容量。

两轮合并后，按图形状统计 NPU Event 的 target graph p50/p90。
B128 为 Native 95.63/96.64 ms、V25 118.65/124.44 ms、
混合整组 105.39/111.32 ms。B192 仅 Replay 出现，
V25 为 150.58/151.91 ms，混合整组为 134.17/137.36 ms，
最高只有 141 个活跃请求。该表按捕获图形状分组，混有不同
活跃行数，不能据此验收同活跃 batch 的 forward。长文本
p50 的回退以及按活跃行数对齐后的 forward 仍需查明。
原始结果位于 Blue2 的 `serve-mixed-packed.GHPMH4/results-long`。

### 首次 flush 错峰的算子探针

有同步重步的根因证据来自 9B 短文本实测的 **124/128 行同时
flush**。实验 cursor 在请求重置时按 ring slot 分配首次折叠
阈值，MTP3 的四档为 20/16/12/8。每档仍保留两个 verify
窗口的原容量约束。首次折叠完成后恢复 20，后续 ring
绕回物理 0 也不会重做错峰。实验使用 cursor 私有状态的高位，
在传入 verify 前只取 flush 位。

Blue2 Ascend910_9362 die 13，CANN 9.2.1。MTP3
B1/B32/B128/B256 每组默认与实验均通过标量或批量读写、状态
校验和图回放。实验组捕获后连续回放 40 个满接受窗口，
全部与 CPU cursor 参考逐元素一致。B32/B128/B256 单步
最多有 8/32/64 行 flush。MTP1、MTP2 B32 的峰值为 4、7 行。
首个窗口之后持续触发重放和环绕，状态位均能清除。
4 项 CPU cursor 单测通过。

B128 的 reset 图时延 p50 为默认 65.20 µs、错峰 69.56 µs，
B256 为 60.30 和 58.02 µs。各 25 次回放，运行中常规
分支约 50–71 µs，样本波动足以覆盖数微秒的差异。
这些图只运行 cursor，未计入服务侧取 flush 位的成本。
结果证明同步满接受场景下可在 cursor 层分散峰值，真实请求
是否仍保持收益取决于接受长度、请求补入和 checkpoint
重舍入。服务独立 A/B 通过同一个实验 cursor 库及同一个
整组轮转 verify，数据存于
`/home/lmt_infer/replayssm/serve-stagger.qZP3qP`。
算子报告位于 `cursor-stagger.o9PMAu/results`。

### 按两条历史记录批量广播的折叠消融

`mixed_fold_pair`（`aiv_ablation=6`）将两条相邻加权 D
记录一次广播到 2 × 64 × 128 元素的 FP32 工作区。两条记录
分别乘对应 K，随后仍按原顺序加到 checkpoint。每对记录
复用原混合核的持久历史 panel 缓冲区，显式分配 64 KiB
（原 32 KiB），每 AIV 核的显式 UB 分配约 166.3 KiB。
奇数历史长度的末条记录用原折叠写法。

Blue2 Ascend910_9362 die 13，CANN 9.2.1、torch-npu 2.10.0.post7，
MTP3。两种几何的 B4/B8 ragged/padded 相对原混合核逐元素一致，
覆盖了 P15 的尾记录、跨 ring 尾和独立 checkpoint/ring 槽。
15 项 CPU 单测和 24 份设备报告通过。性能组使用固定 T4、随机
摆放、flush pos16、其余 pos7、base31，每组 35 次图回放。
近满 flush 的单层 p50 单位为 µs。

| 几何 | Native | V25 | 原混合 | 双行广播 |
|---|---:|---:|---:|---:|
| 27B TP4 B32 F31 | 196.84 | 336.48 | 359.08 | 363.24 |
| 27B TP4 B64 F62 | 376.24 | 633.92 | 647.94 | 651.42 |
| 9B TP1 B128 F124 | 1874.60 | 3260.06 | 3326.84 | 3346.30 |
| 9B TP1 B256 F248 | 3711.50 | 6474.12 | 6587.30 | 6623.90 |

在 F=0、B/8、B/4、B/2 及近满 flush 的 20 个性能点，
双行版均比原混合核慢 0.35%–3.32%。有 flush 的差距
为 0.35%–1.79%。减少广播调用和显式屏障未带来实际收益，
先保留原混合折叠。原始报告在 Blue2 的
`mixed-fold-pair.MTmXuq/results`。

本轮同输入下，原混合在 9B B128 F32 为 1864.68 µs
（Native 1873.12），B256 F64 为 3704.76 µs
（Native 3714.12）。27B 单 rank 的 B32 F8/B64 F16
则为 222.24/399.24 µs（Native 197.34/375.70）。
因此将满 flush 峰值压到约四分之一，对两种几何的单层验收
意义不同。这些点使用固定的历史长度，服务验收仍需记录实际
活跃行、接受数和每步历史长度。性能采样时同卡另一 die
使用同一项目的服务对照，无其它测试混入本组路径。

同形状的 27B TP4 单 rank B32 F31 采集了 `msprof`
`PipeUtilization`。取最后 35 次混合算子任务中位数，
原混合 / 双行版的任务时长为 351.53 / 358.85 µs，AIV
为 307.06 / 313.24 µs，Vector 为 227.34 / 229.06 µs，
Scalar 为 101.08 / 105.05 µs。AIC 的 MAC 时间均为
1.25 µs，约占对应 AIC 时间的 0.4%。计数器区间可能重叠，
不可直接相加。结果支持停止依靠 AIV 屏障与广播批次微调
来消除这组重步差距。若把折叠矩阵乘移入 Cube，必须同时
量化量化转换、排布和 AIC/AIV 同步成本，裸 BMM 的结果
不能替代端到端折叠测量。原始 profile 在同一实验目录的
`profile-mixed_fused_fold` 与 `profile-mixed_fold_pair` 下。

### 缩短待折叠历史

Blue2 第 6 张卡 die 13、Ascend910_9362、CANN 9.2.1、
torch-npu 2.10.0.post7。使用 `9a18a8b` 对应工作区构建的
混合库，原始报告记录完整构建 SHA-256。固定 MTP3 T4、base31、
所有行 flush、每组 35 次图回放，对比 Native、V25 和当前
混合整组轮转。P1/P2/P4/P8 均完成两种几何的两档 batch。
P12 完成 27B，9B 的报告在释放卡时中止，排除该份报告。
同卡另一 die 在部分采样期间运行本项目的服务对照。

P4 单层 p50/p90 如下，单位 µs。27B 是单 rank 算子几何，
表内未包含实际模型 forward、跨卡通信或请求排队。

| 场景 | Native | V25 | 当前混合 |
|---|---:|---:|---:|
| 27B TP4 B32 | 201.48 / 244.14 | 224.52 / 225.98 | 198.26 / 202.02 |
| 27B TP4 B64 | 376.78 / 397.92 | 420.18 / 423.36 | 348.00 / 351.12 |
| 9B TP1 B128 | 1871.88 / 1872.18 | 2142.02 / 2153.40 | 1734.62 / 1738.54 |
| 9B TP1 B256 | 3714.66 / 3715.32 | 4267.10 / 4282.68 | 3446.10 / 3457.84 |

P4 的混合核 p50 在四点比同轮 Native 低 1.6%–7.6%。
P8 四点则慢 17.0%–22.7%。这支持优先验证每步折叠上一个
已接收前缀的策略。MTP3 将 cursor 的逻辑阈值从 20 降到 8
后，首轮无历史，之后最多保留上轮接收的四条历史。物理 ring
仍为 32。全批 P4 提供固定形状证据，实际 P1–P4 交错、填充
行、迟到请求和整网时延需要服务实测。算子原始报告位于
Blue2 的 `mixed-fold-pair.MTmXuq/results-horizon`。

接着用本地 arm64 CPU、Torch 2.10.0 对两种几何各跑 L8/12/16/20
两个合成序列，B1、seed 42、MTP3 T4、120 窗口，beta 采用 BF16
边界量化。常规序列使用已接收计数 3/1/4/2，gate 衰减上限
0.04。保留压力序列使用 4/4/3/4/2/4/1，gate 衰减上限 0.002。
诊断复用 FP16 ring 和 BF16 checkpoint 的 CPU 黄金算子，
与 FP64 连续状态和每个已接收前缀后重存 BF16 的 dense 参考
分别比较。所有 16 份报告在 atol 0.001、rtol 0.03 下均为
零越界步骤。两种序列的 L8 都折叠 119 次，待折叠历史最多 4。
L20 则分别折叠 20/25 次，待折叠历史最多 15/16。

高保留场景的输出最大绝对差如下。每格为 L20 → L8。
较频繁的 BF16 写回提高了相对 FP64 的误差，在这个输入中
相对 BF16 边界参考的误差反而降低。CPU 结果用于筛选策略，
对真实模型的 token 一致性及设备的累加次序没有验证效力。

| 几何 | 相对 FP64 | 相对 dense BF16 边界 |
|---|---:|---:|
| 27B TP4 | 1.221e-4 → 1.831e-4 | 2.441e-4 → 1.831e-4 |
| 9B TP1 | 1.221e-4 → 2.136e-4 | 2.594e-4 → 1.831e-4 |

```bash
python3 scripts/replayssm/diagnose_sd_sequence.py --cpu-reference \
  --geometry 9b --tp 1 --logical-limit 8 --steps 120 \
  --accepted 4,4,3,4,2,4,1 --gate-max 0.002 --beta-native-precision \
  --atol 0.001 --rtol 0.03 --output /tmp/replayssm-l8-cpu.json
```

恢复上卡后可用同一命令将 `--cpu-reference` 换成
`--mixed-library "$mix"`，用实际 packed 混合算子对每个
候选输出、checkpoint 与新 ring 记录做逐步校验。目标服务
仍要比较同活跃行的 p50/p90，并检查 9B 长短文本和多模态
请求的 token 输出、已接收数和吞吐。27B TP4 整网结果仍待
四 die 环境。因第 6 张卡交给其他任务，错峰 A/B、同活跃行
及 L8 服务脚本均已停，当前证据仅包含已完成的上述部分。

### 短历史与紧凑混合算子的服务对照

在 Blue2 使用同一个获准的逻辑 die 串行运行服务和探针。
用 `npu-smi info -m` 核实映射。物理卡 6 的 chip 1 是逻辑
die 13，进程以 `ASCEND_RT_VISIBLE_DEVICES=13` 启动时对应
进程内 `npu:0`。本轮服务只跑 9B TP1。27B TP4 的单 die
探针仍只模拟单 rank 的头数。

使用当前源码版 vLLM Ascend、与它匹配的主扩展，以及按
`-DSOC_VERSION=ascend910_9362` 编译的 mixed verify 实验库。
以 [9B 服务清单](LEVEL3_SD_9B_MULTIMODAL_SERVE.md) 的启动命令
为基础，依次替换 `--additional-config`。以实际构建路径替换
表中的 `/abs/build/libreplayssm_sd_mix_torch.so`。

| 对照 | `--additional-config` | 对比目的 |
|---|---|---|
| Native | `{"replayssm_sd":false}` | 整条原生 MTP3 路径 |
| Vector L20 | `{"replayssm_sd":true,"replayssm_sd_verify_variant":20}` | V20 + compact v1 服务基线 |
| Mixed L20 | `{"replayssm_sd":true,"replayssm_sd_mixed_library":"/abs/build/libreplayssm_sd_mix_torch.so"}` | 紧凑 token、按 key head 分组，verify 内折叠 checkpoint |
| Mixed L8 | `{"replayssm_sd":true,"replayssm_sd_mixed_library":"/abs/build/libreplayssm_sd_mix_torch.so","replayssm_sd_logical_limit":8}` | 同一算子和缓冲布局，仅缩短折叠阈值 |

每组用独立的 `VLLM_CACHE_ROOT`，记录源码 commit 和两份扩展的 hash。
Mixed 路径限 MTP3、BF16、9B TP1 或 27B TP4、`max-num-seqs <= 256`。
未指定阈值时 MTP3 为 20。L8 仍分配 32 项物理 ring，cache dtype 和容量
相同，但会在每个非空 accepted 窗口后安排折叠。启动时检查 logical limit、
mixed dispatch 日志和 FULL 图捕获。先跑单请求图回放并覆盖多轮
fold 与 ring 绕回，再测多请求和 ragged 候选。

端到端测试依次运行短文本与 4000 输入 / 400 输出，C128、C256，
保持模型权重、预算、采样、prompt 分布和图配置一致。按实际
`num_spec_decodes`、候选 token 数和图形状分组，单独报告纯 decode
的 target forward p50/p90。对 Mixed L20 与 L8，检查同一批请求的
MTP 接受率、折叠比例分布、输出及排队情况。若 Native 达不到同一
活跃 batch，汇报可重叠的 batch 区间及各自的 HTTP 吞吐与容量。

#### 设备连续窗口正确性

`b53e1776`，Blue2 逻辑 die 13，Ascend910_9362、CANN 9.2.1、
PyTorch 2.10.0+cpu、torch-npu 2.10.0.post7、vLLM 0.30.0+empty。
使用主扩展 cursor 和整组轮转 mixed verify/compact，mixed binding SHA-256
`d1b9b1e315bd6eb77d811d0be6c5c59764c59a914ba0a2fcc0ab2c309f2a5b77`。
两种几何各用 B1、MTP3、T4，固定 accepted 模式 3/1/4/2，
seed 42、gate 最大绝对值 0.002、BF16 beta 往返。每组跑 120 步，
按实际 BF16 checkpoint 和 FP16 D/K、FP32 G ring 对比本步参考，
容差 `atol=0.001, rtol=0.03`。

| 几何 / 阈值 | 折叠步数 | 最大历史 | 输出对本步参考 max abs | checkpoint 对本步参考 max abs | 输出对 FP64 dense max abs |
|---|---:|---:|---:|---:|---:|
| 9B TP1 / L20 | 20 | 15 | 6.10e-5 | 4.88e-4 | 1.22e-4 |
| 9B TP1 / L8 | 119 | 4 | 3.05e-5 | 4.88e-4 | 2.44e-4 |
| 27B TP4 单 rank / L20 | 20 | 15 | 3.05e-5 | 4.88e-4 | 1.22e-4 |
| 27B TP4 单 rank / L8 | 119 | 4 | 6.10e-5 | 9.77e-4 | 1.98e-4 |

四组均通过 120 步，无 cursor/verify 状态错误或本步精度超差。
D ring 最大绝对差各为 1.22e-4，K/G 一致。设备结果复现了 CPU
探针观察到的 L8 舍入累积增加。该测量针对合成输入连续窗口，
服务端的候选接受率、模型输出与 forward 时延用独立 A/B 验证。
原始报告位于 Blue2 的 `horizon-die13.Qozz2D/sequence.mt9ixy`。

#### 9B TP1 的 4k/400 服务结果

同一逻辑 die 13、TP1、MTP3、BF16 模型和 checkpoint、源码
`b53e1776`、CANN 9.2.1、torch-npu 2.10.0.post7、vLLM 0.30.0+empty。
原生主扩展 SHA-256 为
`d4c77cafd3d282cae488c5774069a25395fec3684555ff9027532a1b251dd280`。
两组 Replay 使用上节同一 mixed binding，并保持物理 ring 32。
16 个纯文本 prompt 轮流使用，每请求确认为 4000 输入和 400 输出。
`max-model-len=4608`、`max-num-seqs=256`、`max-num-batched-tokens=8192`、
显存利用率 0.85、关闭 prefix cache 与 async scheduling、FULL_DECODE_ONLY。
每组运行一轮。Event 只计 target 图设备时延，HTTP 结果来自
启用 Event 同步和 Replay flush 采集的服务，吞吐按整秒 wall time 近似计算。

| 客户端并发 | 路径 | 成功 | token/s | HTTP p50 / p90 (s) | 纯 decode 峰值活跃行 | MTP 草稿接受率 |
|---:|---|---:|---:|---:|---:|---:|
| 128 | Native | 128/128 | 665 | 67.85 / 75.02 | 94 | 84.98% |
| 128 | Mixed L20 | 128/128 | 731 | 67.13 / 69.29 | 128 | 85.12% |
| 128 | Mixed L8 | 128/128 | 731 | 66.97 / 69.04 | 128 | 85.07% |
| 256 | Native | 256/256 | 697 | 116.71 / 145.99 | 94 | 84.87% |
| 256 | Mixed L20 | 256/256 | 731 | 126.49 / 139.70 | 141 | 85.15% |
| 256 | Mixed L8 | 256/256 | 731 | 126.14 / 139.15 | 141 | 85.31% |

Native 的 KV 容量为 418,176 token，两组 Replay 均为 615,628 token
（+47.2%）。C256 的纯 decode 峰值从 94 到 141（+50%）。
L8 和 L20 的服务吞吐接近，C256 中两者的 HTTP p50
都比 Native 高，p90 较低。两侧实际 batch 分布不同，
在 B≥8、实际行数、实际候选数和图形状均相同的切片里，
没有同时达到每侧至少五个样本的 Native/Replay 组。

C128 满 B128、512 候选时，L20 的 target graph p50/p90 为
109.92/112.77 ms（44 样本），L8 为 107.45/107.85 ms（45 样本），
分别减少 2.2% 和 4.4%。C256 有 8 个符合相同形状和样本数门槛
的 L20/L8 切片，实际 B115–141，L8 的 p50 低 1.0%–3.0%。
B141 切片的 p90 从 137.15 ms 增至 138.15 ms，各 7 个样本。
按行统计，L20 的折叠比例约 24%，L8 约 100%。短历史缩短
每次折叠的历史遍历，实测约束比单次 flush 比例更有解释力。

L8/L20 的 B1 短文本冒烟输出 96 token 逐个一致。
相同请求顺序的 C128/C256 长文本分别有 85/128 与 172/256
个请求出现 token 分叉，首分叉位置的中位数为 153/142（从 0
计数）。服务负载下的 batch 路由也会影响舍入，尚无重复同路径或
logprob 对照来归因分叉。保持 L20 默认，并补 C88 固定客户端
并发以取得 Native/Replay 的同批 forward 对照。

原始服务日志、Event 采集 patch、HTTP 报告及启动脚本保存在 Blue2
的 `horizon-die13.Qozz2D/serve.mi4Iyf` 及其父目录。

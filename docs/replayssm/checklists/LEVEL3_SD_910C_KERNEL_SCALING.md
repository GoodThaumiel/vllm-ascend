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

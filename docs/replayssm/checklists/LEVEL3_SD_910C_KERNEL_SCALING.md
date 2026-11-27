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

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

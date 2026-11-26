# 910B4 AIC/AIV 整窗 verify 上机清单

首版对照 Verify V20、Native RGDR，以及 Verify + Compact v1 同图回放。
使用 27B TP4（HK4/HV12）和 9B TP1（HK16/HV32）的单卡单层几何。
使用 BF16 checkpoint/Q/K/V，FP16 D/K ring、FP32 G ring，保持独立 checkpoint/ring 槽。
Benchmark 两组沿用 BS1–128、MTP3/T4、pos15、base31、mixed/full flush。

## Tiling 和同步

- AIC group 的任务是 `(request, key head)`，按 block stride 遍历，blocks 取 `min(BS × HK, 20)`。
  910B4 每个组有 1 个 AIC 和 2 个 AIV。两种几何分别为每个 K head 处理 3 或 2 个 V head。
- AIC 用两次 ND→NZ 整理一份共享 RHS。K 在列 0–3，Q 在列 4–7，列 8–15 补零。
  三次搬运分别写互不重叠的 L1 区域，合用一次 MTE2→MTE1 就绪事件。
  每个 V head 做 `[128,128] × [128,16]` BF16 Mmad，FP32 累加，Fixpipe 以 ND 写出。
  L1 的 A/B 为 32/4 KiB，L0A/B/C 为 32/4/8 KiB。
- 两个 AIV 各读 FP32 `[64,16]` 到自己 UB 后发回确认。投影每个 AIC 使用一个 8 KiB workspace 槽，最多 160 KiB。
  AIC 在两个 AIV 均确认后可覆写这个槽。此后 AIV 仅在 UB 上重放历史与递推，AIC 可进行下一 head 的投影。
  两个 AIV 的 V 切片不重叠。每个 AIV 的共享点积表在 K head 内跨 V head 复用，两侧 AIV 当前各计算一次。
- `READY=6` 从 PIPE_FIX 发布。两个 AIV 在 MTE2 搬完各自的 tile 后各发一次 `CONSUMED=7`。
  910B 模式 2 在两侧都确认后给 AIC 一个计数。每个 head 恰好一轮握手，包括 padding 和错误 metadata。
  没有全核栅栏。G ring 和 status 由下半 V 切片的单个 AIV 写入，K ring 在一个 K head 组结束后写一次。
- 每个 AIV 显式 UB 42,144 B，另计 CCE 临时区。历史最多 16，ring 为 32。短窗只消费有效的 K/Q 列。
  输入来自固定的 NPU 连续 buffer，含同一 launch 内互不重叠的有效 checkpoint/ring 槽。
  这是独立实验库，当前 runner 使用统一的 T、pos、base 合成工作负载。

本轮以 CANNBot 的 API 最佳实践核对 ND2NZ、NZ2ND、LoadData 和 flag 数据可见性。
Tiling 技能用于检查 AIC/AIV 切分和 L0/UB 容量。代码检视重点为事件配对、地址边界、队列生命周期和输出单写者。
[昇腾 CrossCoreSetFlag](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta2/API/ascendcopapi/atlasascendc_api_07_0273.html)
· [Fixpipe V220](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0251.html)

## 构建和冒烟

在 910B4 上使用 CANN 9.2.0、torch-npu 2.10.0.post4，复用已有 V20 / Compact v1 框架扩展。
从仓库根目录执行。首先用 1 个 AIC 走多轮 workspace 复用，然后测默认核数。
每个 case 会分别用 CPU FP32 乘法验证各 workspace 槽最后一张投影，并校验 eager 和 Graph 的输出、D/K/G、status 和 checkpoint。
Graph 检查会把 workspace 和候选写回位置填 NaN，核验捕获图确实写入新结果。

```bash
set -euo pipefail
P=experimental/replayssm_cube_910b
BUILD="$P/mixed_verify/build-history"
R=/tmp/replayssm-mix-910b-history
mkdir -p "$R"
cmake -S "$P/mixed_verify" -B "$BUILD" -DSOC_VERSION=Ascend910B4 \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.0 \
  -DPython3_EXECUTABLE="$(command -v python3)" > "$R/configure.log" 2>&1
cmake --build "$BUILD" -j8 > "$R/build.log" 2>&1
LIB="$BUILD/libreplayssm_sd_mix_torch.so"
```

### 原始投影精度回归

在 `8213d9c` 上，27B TP4 的单核冒烟通过，B1 四个 AIC 的 graph replay 投影有 509 个超差。
随后 9B TP1 的 B1 单核 eager 投影有 511 个超差。
原来的 RHS 整块清零与后续 K/Q 搬入重叠在 L1，现改为 K、Q、padding 各写自己的行区间。
保留上一轮的事件修复。原始容差保持 projection `atol=2e-5, rtol=1e-3`，输出和 ring `atol=0.01, rtol=0.03`。
[CANN 流水同步](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00064.html)

拉取修复后重新编译实验库及其依赖的 kernel .so。下面两种 workspace 使用同一核调度和握手。
`per-core` 复用投影槽，是正式测量配置。`per-head` 给每个 V head 一个独占投影槽，便于区分共享槽复用和投影/递推错误。
每个配置均运行五次未计时的 eager 校验和五次未计时的 graph 校验。
失败时 JSON 和 traceback 会标记阶段及张量名，并按 K、Q、padding 分别列出超差、零值和非有限值数目。
`--mixed-dump` 将首个失败张量的实测值和参考值保存为 `*.mismatch.pt`。

```bash
for workspace in per-head per-core; do
  for case in '27b 20 15' '9b 1 16'; do
    read -r geometry cores pos <<< "$case"
    name="precision-$geometry-$workspace-c$cores-p$pos"
    timeout 120s python3 "$P/run.py" --suite mixed --mixed-library "$LIB" \
      --geometry "$geometry" --batches 1 --pos "$pos" --length 4 --base 31 --flush mixed \
      --mixed-cores "$cores" --mixed-workspace "$workspace" --mixed-check-repeats 5 \
      --mixed-dump --iterations 3 --output "$R/$name.json" 2>&1 | tee "$R/$name.log"
  done
done
```

回传失败 JSON、`*.mismatch.pt` 和 `sha256sum "$LIB" "$BUILD/lib/libreplayssm_sd_mix_kernel.so"`。
如果同一几何的 `per-head` 失败，优先定位实际超差的列组。如果只有 `per-core` 失败，优先检查跨核复用时序。

### AIV 历史搬运优化回归

对照组为已在 910B4 上通过的 `f61c9c91c`。使用其原构建目录的实验库和 kernel 库，
本轮编译单独使用上面的 `build-history`。每次 Python 进程只加载一组库。
如旧目录已覆盖，可在该 commit 的独立 worktree 中按原清单重建，将下面的 OLD 指向该构建。

一个 AIV 读取 K 的完整 128 维行、D 的前或后 64 维。跨 ring 尾部分两段，
两段紧排在同一块 FP16 UB，随后一次转换为 FP32。K 搬入提前到准备 Q/K 之前，
D 搬入提前到等待 READY 之前。队列深度为 1，两次使用的读写由队列管理。
额外显式 UB 为 11,776 B/AIV。pos15/base31 下每个 27B K head 组、每个 AIV 的历史 K/D
搬运次数从 60 降至 8，Cast 从 60 降至 4。实际传输的有效 K/D 字节数相同。
使用投影精度回归检查跨核握手，然后执行以下覆盖正好抵达 ring 尾、跨尾和不跨尾的测试。

```bash
OLD="$P/mixed_verify/build/libreplayssm_sd_mix_torch.so"
for lib in "$OLD" "$LIB"; do
  sha256sum "$lib" "$(dirname "$lib")/lib/libreplayssm_sd_mix_kernel.so"
done
for geometry in 27b 9b; do
  for case in '0 31 1' '1 31 4' '15 17 4' '16 17 4' '16 0 4' '16 31 4'; do
    read -r pos base length <<< "$case"
    timeout 180s python3 "$P/run.py" --suite mixed --mixed-library "$LIB" \
      --geometry "$geometry" --batches 1,2 --pos "$pos" --base "$base" --length "$length" \
      --path mixed_verify mixed_pair --mixed-dump --mixed-check-repeats 5 --iterations 3 \
      --output "$R/history-$geometry-p$pos-base$base-t$length.json"
  done
done
```

先在同卡串行比较 27B B32 三条路径，每次独立进程，固定 pos15/base31/T4。
回传每轮的 graph p50/p90 和目标 kernel 的 op_summary。按同 workload 和 MEASURE 回放
筛选，统计 task_duration、aiv_time、vec/scalar/mte2/mte3 时间与 ratio，以及可用的 AIC 指标。

```bash
for round in 1 2 3; do
  order=(old new native)
  if [[ "$round" == 2 ]]; then order=(new native old); fi
  if [[ "$round" == 3 ]]; then order=(native old new); fi
  for version in "${order[@]}"; do
    op=mixed_verify
    library="$LIB"
    if [[ "$version" == old ]]; then library="$OLD"; fi
    if [[ "$version" == native ]]; then op=native; fi
    msprof --task-time=l0 --output="$R/prof-b32-$version-r$round" \
      python3 "$P/run.py" --suite mixed --geometry 27b --batches 32 --path "$op" \
      --mixed-library "$library" --flush mixed --length 4 --pos 15 --base 31 --iterations 100 \
      --output "$R/b32-$version-r$round.json"
  done
done
```

B32 的上一轮 task p50 为 213.8 µs，native 为 194.2 µs。本轮以同卡 native
的 1.05 倍以内作为“接近”的工程目标（按上轮数据约 204 µs），同时观察
aiv_vec_ratio 与 0.9 的距离，native 上轮为 0.873。按每轮独立对比，
用下方 BS1–128 无 profiler 的结果复核时延。另列 mixed_pair，尤其关注 full flush 下的 compact。
若 vec_ratio 上升而 task 时延增长，性能判定以时延为先。

### 常规冒烟

```bash
for geometry in 27b 9b; do
  for scenario in '0 1' '15 4' '16 4'; do
    read -r pos length <<< "$scenario"
    timeout 120s python3 "$P/run.py" --suite mixed --mixed-library "$LIB" \
      --geometry "$geometry" --batches 1 --length "$length" --pos "$pos" --base 31 \
      --mixed-cores 1 --path mixed_verify --iterations 3 \
      --output "$R/smoke-$geometry-p$pos-t$length.json" \
      2>&1 | tee "$R/smoke-$geometry-p$pos-t$length.log"
  done
  timeout 240s python3 "$P/run.py" --suite mixed --mixed-library "$LIB" \
    --geometry "$geometry" --batches 1,32,128 --pos 15 --base 31 --flush mixed \
    --iterations 3 --output "$R/multicore-$geometry.json" \
    2>&1 | tee "$R/multicore-$geometry.log"
done
```

任一失败记录首个 CCE 错误或设备日志。若投影校验先失败，检查 ND/NZ 布局、Fixpipe 和 ready/consumed。
投影通过后再检查 V64 切分、history 跨尾和候选列。编译和 NPU 测量需上机完成。

## 测 BS1–128、采集 profiler 和出图

```bash
for geometry in 27b 9b; do
  for flush in mixed full; do
    python3 "$P/run.py" --suite mixed --mixed-library "$LIB" \
      --geometry "$geometry" --flush "$flush" --pos 15 --base 31 \
      --iterations 100 --output "$R/$geometry-$flush.json" \
      2>&1 | tee "$R/$geometry-$flush.log"
  done
done
for path in native v20 mixed_verify vector_pair mixed_pair; do
  msprof --task-time=l0 --output="$R/prof-b32-$path" \
    python3 "$P/run.py" --suite mixed --mixed-library "$LIB" \
      --geometry 27b --batches 32 --flush mixed --path "$path" \
      --iterations 100 --output "$R/prof-b32-$path.json"
done
for geometry in 27b 9b; do
  for flush in mixed full; do
    python3 "$P/plot.py" "$R/$geometry-$flush.json" --out "$R/$geometry-$flush.svg" \
      --bandwidth-gbs "$HBM_GBS" --cube-tflops "$CUBE_TFLOPS" \
      --vector-tflops "$VECTOR_TFLOPS" --roof-source "$CALIBRATION_RUN"
  done
done
```

六条默认路径每个场景应有 768 行。Mixed/full 的 verify 使用相同输入。
优先比较 `mixed_verify / v20` 的 p50/p90，以及 `mixed_pair / vector_pair`，然后和 Native RGDR 对照。
Profiler 记录 AIC MAC/FIX 和 AIV Vector/Scalar/MTE，确认设备显示混合任务及实际核数。
数值检查、预热、capture 和测量后的验证也会发算子，按 MEASURE 和回放身份取目标样本。
估算 roofline 计入 FP32 workspace 的逻辑读写。设备 HBM 可能经 L2 命中，实测值接入 `--counters`。
报告附 commit、两份扩展 hash、原始 CSV 和四组 JSON。出图参数沿用 [Vector 基线口径](LEVEL3_SD_VECTOR_BATCH_PROBE.md)。

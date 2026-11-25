# 910B4 Vector 基线 BS1–128 对照

使用已有的 verify V20（tile128、40 AIV）和 compact v1（tile32、40 AIV）。
同口径的 AIC/AIV 整窗投影对照见 [910B4 混合核 verify](LEVEL3_SD_MIXED_VERIFY_910B.md)。
每个 batch 分别测框架原生 RGDR、V20、compact v1，以及 V20 → compact v1 的同图回放。
TP 表示单卡上模拟的每 rank 头部几何。两种模型都用 T4（MTP3），pos15、base31、ring32 和独立 checkpoint/ring 槽。

| 几何 | Q/K heads | V heads | 每个场景的 batch |
| --- | ---: | ---: | --- |
| Qwen3.5-27B TP4 | 4 | 12 | 1 至 128，每个整数 |
| Qwen3.5-9B TP1 | 16 | 32 | 1 至 128，每个整数 |

沿用 mixed 和 full flush 两个场景。Mixed 的前 `max(1, BS // 8)` 行 flush。Full 每行都 flush。
两组 verify 和 native 读取同样的输入与历史。BS1 下两个场景都是一行 flush。
V20 是之前 27B TP4 的性能基线。9B TP1 使用同一实现重新建立基线。

## 上机测量

沿用 CANN 9.2.0、torch-npu 2.10.0.post4，以及已支持 V20 和 9B TP1 的 vllm-ascend 扩展。
在仓库根目录执行。脚本会记录实际加载的扩展路径和 SHA256、每个 batch 的头数、flush 行数、逐次 Event 时间及 p50/p90。
套件会在每个 batch 先校验 native 和 V20 的输出、两条 Replay 路径的 status，以及 compact 的 checkpoint。
每次回放前恢复需要覆盖的状态，恢复在 Event 区间外。

```bash
set -euo pipefail
PROBE=experimental/replayssm_cube_910b/run.py
RESULTS=/tmp/replayssm-vector-bs-sweep
mkdir -p "$RESULTS"
for geometry in 27b 9b; do
  for flush in mixed full; do
    python3 "$PROBE" --suite vector --geometry "$geometry" --flush "$flush" \
      --batches 1,128 --iterations 3 --device 0 \
      --output "$RESULTS/$geometry-$flush-smoke.json"
    python3 "$PROBE" --suite vector --geometry "$geometry" --flush "$flush" \
      --iterations 100 --device 0 --output "$RESULTS/$geometry-$flush.json" \
      2>&1 | tee "$RESULTS/$geometry-$flush.log"
  done
done
```

正式输出每个文件应包含 128 个 batch × 4 条路径，共 512 行。
若有失败，JSON 会记录首个失败 batch 和异常，已完成的 batch 仍可读取。
BS1/32/64/128 可用同一入口仅跑一条路径抓 profiler。以下示例抓 9B TP1 mixed 的组合图，替换 `--path` 可分别抓 `native`、`v20`、`compact_v1`。

```bash
msprof --task-time=l0 --output="$RESULTS/prof-9b-b128-mixed" \
  python3 "$PROBE" --suite vector --geometry 9b --flush mixed --batches 128 \
  --path vector_pair --iterations 100 --device 0 --output "$RESULTS/prof-9b-b128-mixed.json"
```

按 `MEASURE_BEGIN/END`、回放任务及实际算子符号归组。有需要时对照
`REPLAYSSM_SD/<case>/<path>` host range。前置数值检查、预热和 capture 也会发算子。
状态恢复的 copy 在 Event 时间之外，但仍可能出现在 profiler 范围里，汇总时过滤。
组合图每次回放应匹配一个 verify 和一个 compact。

## 出图

每个模型和场景各出一张图，用同一设备校准的 HBM GB/s、Vector/Cube TFLOP/s。
命令中的上限取同卡实测值或注明规格来源。没有设备 FLOP/HBM 计数时，图按估算计算量和逻辑字节绘制。
如拿到每次回放的 FLOP 和 HBM 读写字节，可用 `--counters` 接入相同行集合的 CSV。
计数和延时须来自相同几何、场景及路径。组合图的计数按一次回放聚合两个算子。

```bash
for geometry in 27b 9b; do
  for flush in mixed full; do
    python3 experimental/replayssm_cube_910b/plot.py "$RESULTS/$geometry-$flush.json" \
      --out "$RESULTS/$geometry-$flush.svg" --bandwidth-gbs "$HBM_GBS" \
      --cube-tflops "$CUBE_TFLOPS" --vector-tflops "$VECTOR_TFLOPS" \
      --roof-source "$CALIBRATION_RUN"
  done
done
```

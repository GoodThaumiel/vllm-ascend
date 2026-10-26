# L3 SD verify v8 独立消融上机清单

环境：Ascend910B4、CANN 9.2.0、torch_npu 2.10.0.post4、vLLM 0.30.0+empty、源码编译的 vLLM Ascend。先用目前可见的 4 卡做 TP4；待 8 卡可见再补 TP8。所有版本仍为独立 verify 候选，不切换 `vllm serve` 的 MTP dispatch。

| variant | 基于 v7 的唯一主要改动 | 主要观察 |
| --- | --- | --- |
| 7 | 已通过的 UB 历史修正 | 同容器对照 |
| 8 / v8b | speculative dot 留在 UB；用 8 行批量修正代替逐 draft 行累加 | Vector、Scalar、V→S 次数、总时延；小 T 可能因固定 8 行而变慢 |
| 9 / v8c | history K 合并成最多两段连续读取；history D 用最多两次 2D strided `DataCopyPad` | MTE2、Scalar、总时延；特别检查 ring wrap 与 tile stride |
| 10 / v8d | speculative `Accumulate` 改为 `Duplicate` 系数向量 + `MulAddDst` | Vector、Scalar、总时延；并未减少指令数，性能收益不预设 |

v8b 显式申请 UB 在 tile64 为 114,400 B，v8c/v8d 与 v7 相同，为 112,992 B；均不包含编译器临时空间。v8d 可能改变 FP32 舍入。以上都是消融候选，不应将最快版本直接用于服务，须先完成精度与图回放。

1. 同一容器重编主扩展。若某一候选编译失败，回传首条 `bisheng`/CCE 错误、对应 kernel 名、完整构建日志及设备健康状态；不要用旧扩展继续测试。

```bash
python3 -m pip install --no-build-isolation --no-deps -v -e .
```

2. 按 7→8→9→10 在 TP4/B2/tile64/40 cores 上做 smoke 和 graph。脚本覆盖 pos0/7/12/16、T1/2/3/4、padding、ring wrap；本轮还新增了 `base=0` 的非环绕 case，专门区分 v8c 的两段搬运。每版检查 status 全 0，`out/d_cache/k_cache/g_cache` 相对 FP64 golden 的 max_abs、relative L2，并对 v8d 留意 FMA 舍入。任一版本失败即停在该版本，回传首个失败 JSON 和日志。

```bash
for variant in 7 8 9 10; do
  for stage in smoke graph; do
    python3 experimental/replayssm_sd_ascendc/run.py --backend framework \
      --stage "$stage" --tp 4 --batch 2 --tile-v 64 --cores 40 \
      --variant "$variant" --device 0 \
      --output "/tmp/sd-v8-v${variant}-${stage}.json" || exit 1
  done
done
```

3. 全部正确后先做小范围图回放：B64、tile64、v7/8/9/10；重点比较 pos7/12/16、T4。若某版有明确收益，再扩展 B1/32 和 tile32。计时区外重置缓存，比较同 shape p50/p90，不混用不同容器或 CANN 版本。

```bash
for variant in 7 8 9 10; do
  python3 experimental/replayssm_sd_ascendc/run.py --backend framework \
    --stage perf --tp 4 --batch 64 --tile-v 64 --cores 40 \
    --variant "$variant" --iterations 50 --device 0 \
    --output "/tmp/sd-v8-b64-v${variant}.json" || exit 1
done
```

4. 用同一 `msprof` 抓 native/v7/8/9/10。脚本每组计时 100 次；取 op_summary 后 100 次，并回传 task_duration、AIV、Vector、Scalar、MTE2/MTE3 绝对时间，若有指令级时间线一并提供 V→S、DataCopyPad 和 Vector 等待。各单元时间之和不能直接判断关键路径，需要结合时间线和独立消融的总时延变化。前轮 TP4/tile64/pos12/T4/B64 的 v7≈618 µs、native≈370 µs；≤1.40× native（约 518 µs）只作阶段目标，不是性能预测。

```bash
for variant in 7 8 9 10; do
  python3 scripts/replayssm/profile_sd_operator_compare.py \
    --tp 4 --device 0 --batches 64 --context-lengths 32768 \
    --lengths 4 --positions 12 --ring 32 --tile-v 64 --cores 40 \
    --variant "$variant" --repetitions 100 --warmup 2 \
    --output "/tmp/sd-v8-profile-v${variant}.json" || exit 1
done
```

请附源码 commit、`git status --short`、构建日志、每版 JSON 和 profiler 分解。若 v8b 或 v8d 性能回退但精度通过，保留结果作为消融证据；若 v8c 只在环绕 case 出错，优先检查两段边界、`srcStride` 字节单位和 UB 目的行距。

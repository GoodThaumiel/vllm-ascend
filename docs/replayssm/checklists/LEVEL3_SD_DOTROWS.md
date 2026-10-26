# L3 SD verify v5/v6 上机验证清单

目标：Ascend910B4、CANN 9.2.0、源码安装的 vLLM Ascend、vLLM 0.30.0+empty。先测独立算子，不切换 `vllm serve` 的 MTP dispatch。报告里的容器目前只暴露 4 卡，所以本轮先用 TP4；TP8 待 8 卡可见后再补。

| variant | 与已通过版本的差异 | 要验证的假设 |
| --- | --- | --- |
| 3 | 已通过的共享点积版本 | 同容器对照 |
| 5 | v3 + 多行点积批量归约 | 降低 Vector 小命令及每行 V→S 同步开销 |
| 6 | v5 + 点积计算前预取 checkpoint | 在点积阶段遮蔽 MTE2；验证早预取是否优于 v4 |

1. 重新源码编译主扩展。保存编译日志中 `replayssm_sd_verify_dotrows` 和 `replayssm_sd_verify_dotrows_early` 的资源占用、失败位置；确认 `torch.ops._C_ascend.replayssm_sd_verify_` 可调用。

```bash
python3 -m pip install --no-build-isolation --no-deps -v -e .
```

2. 在空闲卡上按 v3→v5→v6 顺序跑正确性与图回放。先 TP4/B2/tile64，再对通过的版本测 tile32。任何版本失败，停在该版本并回传首个失败 JSON、CCE 日志及 NPU 健康状态。

```bash
for variant in 3 5 6; do
  for stage in smoke graph; do
    python3 experimental/replayssm_sd_ascendc/run.py --backend framework \
      --stage "$stage" --tp 4 --batch 2 --tile-v 64 --cores 40 \
      --variant "$variant" --device 0 \
      --output "/tmp/sd-dotrows-v${variant}-${stage}.json" || exit 1
  done
done
```

3. 全部 PASS 后，对 TP4、B1/32/64、history=0/7/12/16、T=1/4 测 graph replay。相同形状比较 v3、v5、v6 的 p50/p90；reset 在计时区外。重点比较 B32/64、pos12/16、T4。保存 `out/d_cache/k_cache/g_cache` 与 FP64 golden 的最大误差、relative L2 和所有 status 值。

```bash
for batch in 1 32 64; do
  for variant in 3 5 6; do
    python3 experimental/replayssm_sd_ascendc/run.py --backend framework \
      --stage perf --tp 4 --batch "$batch" --tile-v 64 --cores 40 \
      --variant "$variant" --iterations 50 --device 0 \
      --output "/tmp/sd-dotrows-b${batch}-v${variant}.json" || exit 1
  done
done
```

4. 用原生路径与 v3/v5/v6 各跑 100 次，在 `REPLAYSSM_SD/.../native` 和 `/replay` 区间抓 `msprof` 时间线。记录每版的 `task_duration`、AIV Vector/Scalar/MTE2/MTE3 时间和占比；额外看 V→S 等待次数、MTE2 与 Vector 是否实际重叠，以及 `DataCopyPad` 的次数。当前 v3/TP4/tile64/B64 的设备时延参考约 796.5 µs，原生约 369.2 µs；数值只用于同容器回归判断。

```bash
for variant in 3 5 6; do
  python3 scripts/replayssm/profile_sd_operator_compare.py \
    --tp 4 --device 0 --batches 1,32,64 --context-lengths 32768 \
    --lengths 4 --positions 12 --ring 32 --tile-v 64 --cores 40 \
    --variant "$variant" --repetitions 100 --warmup 2 \
    --output "/tmp/sd-dotrows-native-v${variant}.json" || exit 1
done
```

请回传完整 JSON、算子设备时间分解和首次失败日志。若 v5 精度失败，先定位具体 `(position, token, head, V tile)`；若 v6 与 v5 持平或变慢，保留 v5 并从 profiler 时间线决定后续搬运/流水改法。

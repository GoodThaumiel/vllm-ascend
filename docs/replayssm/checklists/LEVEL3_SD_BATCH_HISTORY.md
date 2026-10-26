# L3 SD verify v7 批量历史修正上机清单

环境沿用 CANN 9.2.0 / torch_npu 2.10.0.post4 / vLLM 0.30.0+empty / Ascend910B4。v5 是已验证对照；v7 只替换历史 dot 与 history-D 修正，不切换 `vllm serve` dispatch。当前已知容器只暴露 4 卡，先做 TP4；TP8 待 8 卡可见后补。

v7 的历史 dot 结果留在 UB。每个任务只在 Scalar 侧算一次 16 项 history gate 后缀，系数在 UB 中乘上 dot；按 16 行广播后乘 history-D，再用 4 层树归约。未使用的 history-D 行与系数明确置零。候选 token 的短递归仍沿用 v5。源代码显式申请的 UB：tile32 64,480 B，tile64 112,992 B；请以 CCE 实际资源报告为准。

1. 在同一容器编译主扩展；若编译失败，回传 `replayssm_sd_verify_batch_history` 的第一条 CCE 错误和资源报告。先检查 `Broadcast<float,2,1>`、带 repeat stride 的 `Add`、UB 用量，再运行设备测试。

```bash
python3 -m pip install --no-build-isolation --no-deps -v -e .
```

2. 在空闲卡上按 v5→v7 跑 TP4/B2/tile64 的 smoke 和 graph。v7 若编译或 smoke 失败，停在这一步；设备 trap 后先确认设备健康。保存所有 status 与 `out/d_cache/k_cache/g_cache` 对 FP64 golden 的误差，尤其检查 pos0/7/12/16、T1/4、ring wrap、padding。v7 的 FP32 求和顺序与 v5 不同，应逐 case 比较两版各自的最大绝对误差和相对 L2。

```bash
for variant in 5 7; do
  for stage in smoke graph; do
    python3 experimental/replayssm_sd_ascendc/run.py --backend framework \
      --stage "$stage" --tp 4 --batch 2 --tile-v 64 --cores 40 \
      --variant "$variant" --device 0 \
      --output "/tmp/sd-history-v${variant}-${stage}.json" || exit 1
  done
done
```

3. 正确性与图回放全 PASS 后，小范围比较 B1/32/64、tile32/64 的 v5/v7 graph replay p50/p90；reset 保持在计时区外。重点为 pos12/16、T4。保存编译器报告的 UB 用量、blockDim 和 device memory；若 tile64 的资源受限，先测 tile32。

```bash
for tile in 32 64; do
  for batch in 1 32 64; do
    for variant in 5 7; do
      python3 experimental/replayssm_sd_ascendc/run.py --backend framework \
        --stage perf --tp 4 --batch "$batch" --tile-v "$tile" --cores 40 \
        --variant "$variant" --iterations 50 --device 0 \
        --output "/tmp/sd-history-t${tile}-b${batch}-v${variant}.json" || exit 1
    done
  done
done
```

4. 若 v7 精度与性能均可用，用同一 `msprof` 口径比较 native/v5/v7，100 次计时调用，记录 `task_duration`、AIV Vector/Scalar/MTE2/MTE3 绝对时间、重叠情况。已知 v5 的 TP4/B64/tile64/pos12/T4 纯核约 679 µs，native 约 371 µs；只做同容器、同形状的相对比较。若 v7 没有加速，先确认 Scalar 时间和 Vector 命令数是否下降，再看 Broadcast/树归约是否产生新的瓶颈。

```bash
for variant in 5 7; do
  python3 scripts/replayssm/profile_sd_operator_compare.py \
    --tp 4 --device 0 --batches 1,32,64 --context-lengths 32768 \
    --lengths 4 --positions 12 --ring 32 --tile-v 64 --cores 40 \
    --variant "$variant" --repetitions 100 --warmup 2 \
    --output "/tmp/sd-history-profile-v${variant}.json" || exit 1
done
```

回传首个失败日志或完整结果、所用 commit、编译器 UB 报告及 profiler 分解。完成以上验证前，继续使用 v5 作为已验证参考，不推进 MTP 服务切换。

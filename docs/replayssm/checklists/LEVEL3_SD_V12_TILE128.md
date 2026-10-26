# L3 SD verify v12 tile128 上机验证清单

环境：Ascend910B4、CANN 9.2.0、torch_npu 2.10.0.post4、vLLM 0.30.0+empty、源码编译的 vLLM Ascend。v12 是基于已验证 v9 的独立消融：仅支持 `--variant 12 --tile-v 128`，每个 `(request, value head)` 只遍历一个 V tile。历史 K/D 的批量搬运、候选 Q/K/V 的逐 token 路径、三角修正与缓存语义沿用 v9。v11 的候选 Q/K/V 批量搬运实验与本轮互不依赖；不要在 v11 上机未完成时切换默认 variant。

## UB 预算与风险

按源码 `InitBuffer` 显式申请计算，v9 若直接用 tile128 需 210,016 B，超过 910B4 的 192 KiB（196,608 B）。v12 把通用 `in_` 从 32,768 B 缩至 8,192 B，checkpoint 的 32 KiB BF16 输入按 8 KiB ×4 搬运；history 修正的 `expandedCoeff_` 和 `weightedHistory_` 各 8,192 B 与已完成 MatVec 的 `product_` 分时复用。显式 UB 降为 **169,056 B**，余量 **27,552 B**；余量尚需容纳编译器临时量，**能否编译及实际 UB 占用必须以上机 CCE 报告为准**。tile128 的 16 行树归约最后一级改为单次 `Add`，避免 8-bit row stride 的 256 溢出。该版不使用 double buffer。

1. 拉取本次 commit，重编主扩展。先看 `replayssm_sd_verify_tile128` 的 CCE 编译是否通过、实际 UB 占用及首条警告/错误。若报 UB 不足、`Broadcast`/Vector API 限制或 `DataCopyPad` 错误，停在构建阶段并回传完整日志；不要用旧 `.so` 跑 v12。

```bash
python3 -m pip install --no-build-isolation --no-deps -v -e .
```

2. TP4/B2/40 cores 先跑 v12 smoke，再跑 graph。脚本覆盖 T1/2/3/4、pos0/7/12/16、ring wrap、非环绕与 padding。检查 status 全 0、out 与 d/k/g cache 相对 FP64 golden 的误差、checkpoint 不变，以及 NPUGraph capture/replay。若失败，回传首个 JSON 和日志，不进入性能测试。

```bash
for stage in smoke graph; do
  python3 experimental/replayssm_sd_ascendc/run.py --backend framework \
    --stage "$stage" --tp 4 --batch 2 --tile-v 128 --cores 40 \
    --variant 12 --device 0 \
    --output "/tmp/sd-v12-tp4-${stage}.json" || exit 1
done
```

3. 正确性通过后，在同容器比较 v9/tile64 与 v12/tile128，先 B64、再 B32；同样比较 pos7/12/16、T4 的 graph replay p50/p90。两版每任务的 V tile 数不同，但输出 shape 与算子语义一致。reset 保持在计时区外。tile128 减少一轮 history 修正和候选处理，却把 checkpoint 输入拆成四次搬运，净收益不能预设。

```bash
for batch in 64 32; do
  for variant in 9 12; do
    if [ "$variant" -eq 12 ]; then tile=128; else tile=64; fi
    python3 experimental/replayssm_sd_ascendc/run.py --backend framework \
      --stage perf --tp 4 --batch "$batch" --tile-v "$tile" --cores 40 \
      --variant "$variant" --iterations 50 --device 0 \
      --output "/tmp/sd-v12-b${batch}-v${variant}.json" || exit 1
  done
done
```

4. 用同一 `msprof` 抓 native、v9/tile64、v12/tile128 的 B64、pos12/T4、100 次计时。回传 `task_duration`、AIV、Vector、Scalar、MTE2/MTE3 绝对时间及可能的指令时间线。重点看 checkpoint 四段搬运是否抵消单 tile 收益；执行单元时间可重叠，不可相加归因总时延。v9 的同环境参考为 544.56 µs，native 为 371.42 µs；这只是对照，不是 v12 目标值。

```bash
for variant in 9 12; do
  if [ "$variant" -eq 12 ]; then tile=128; else tile=64; fi
  python3 scripts/replayssm/profile_sd_operator_compare.py \
    --tp 4 --device 0 --batches 64 --context-lengths 32768 \
    --lengths 4 --positions 12 --ring 32 --tile-v "$tile" --cores 40 \
    --variant "$variant" --repetitions 100 --warmup 2 \
    --output "/tmp/sd-v12-profile-v${variant}.json" || exit 1
done
```

5. TP4 全部通过且有收益时，补 TP8 的 smoke/graph 和 B32/B64 性能对照。请连同源码 commit、`git status --short`、CCE UB 报告、全部 JSON、profiler 摘要回传。若 v12 编译或性能不达标，保留 v9 作为已验证基线，v12 不进入默认服务路径。

# L3 SD verify v11 上机验证清单

目标环境：Ascend910B4、CANN 9.2.0、torch_npu 2.10.0.post4、vLLM 0.30.0+empty、源码编译的 vLLM Ascend。v9 是已通过的同容器性能对照；v11 仅在 v9 上把候选 Q/K/V 从逐 token GM→UB 搬运改为多行 `DataCopyPad`。历史 K/D、三角修正、输出与缓存更新逻辑沿用 v9。v11 仍是独立 verify 算子，不接入 `vllm serve` MTP dispatch。

上轮 TP4、B64、tile64、pos12/T4 的 v9 单核 `task_duration` 为 544.56 µs，native 为 371.42 µs。≤518 µs、≤1.40× native 只是阶段目标，不是 v11 的性能预测。上轮 v8 固定 8 行 speculative 修正变慢，v10 `Duplicate` + `MulAddDst` 无明显收益，继续保留为消融反例。

1. 拉取本次 commit，在同一 CANN 9.2.0 容器重新编译主扩展。保存 `replayssm_sd_verify_bulk_qkv` 的首条 CCE 错误和编译器 UB 资源报告。不能用旧 `.so` 运行 v11；若构建失败，在此停止并回传日志。

```bash
python3 -m pip install --no-build-isolation --no-deps -v -e .
```

2. 先在 TP4、B2、40 cores 上验证 v9/v11 的 smoke + graph；覆盖 T1/2/3/4、pos0/7/12/16、ring wrap、padding，以及 tile16/32/64。检查 status 全 0，输出和 d/k/g cache 对 FP64 golden 的误差，以及 NPUGraph capture/replay。短 T 和 tile16 专门验证多行搬运的动态 blockCount 与 UB 对齐。任一失败先停，回传首个失败 JSON 和对应日志。

```bash
for tile in 16 32 64; do
  for variant in 9 11; do
    for stage in smoke graph; do
      python3 experimental/replayssm_sd_ascendc/run.py --backend framework \
        --stage "$stage" --tp 4 --batch 2 --tile-v "$tile" --cores 40 \
        --variant "$variant" --device 0 \
        --output "/tmp/sd-v11-t${tile}-v${variant}-${stage}.json" || exit 1
    done
  done
done
```

3. 正确性通过后做同容器图回放对照。先跑 B64/tile64，确认方向，再扩展 B32/tile32/64；比较 pos7/12/16、T4 的 p50/p90。保持 reset 在计时区外。若 v11 退化，保留结果，不切换默认 variant。

```bash
for batch in 32 64; do
  for tile in 32 64; do
    for variant in 9 11; do
      python3 experimental/replayssm_sd_ascendc/run.py --backend framework \
        --stage perf --tp 4 --batch "$batch" --tile-v "$tile" --cores 40 \
        --variant "$variant" --iterations 50 --device 0 \
        --output "/tmp/sd-v11-b${batch}-t${tile}-v${variant}.json" || exit 1
    done
  done
done
```

4. 对 B64/tile64/pos12/T4 取 native、v9、v11 的同口径 profiler。每组取 100 次计时调用的 `task_duration`、AIV、Vector、Scalar、MTE2/MTE3 绝对时间；有指令时间线时，再数候选 Q/K/V 的 `DataCopyPad` 和等待事件。不同执行单元的时间可以重叠，不能把它们相加当成关键路径。

```bash
for variant in 9 11; do
  python3 scripts/replayssm/profile_sd_operator_compare.py \
    --tp 4 --device 0 --batches 64 --context-lengths 32768 \
    --lengths 4 --positions 12 --ring 32 --tile-v 64 --cores 40 \
    --variant "$variant" --repetitions 100 --warmup 2 \
    --output "/tmp/sd-v11-profile-v${variant}.json" || exit 1
done
```

5. TP4 完整通过后，设备允许时补 TP8 的 smoke/graph，以及 B32/B64、tile64 的 v9/v11 对照。回传源码 commit、`git status --short`、编译日志、JSON 和 profiler 摘要。若出现精度错误，优先核对 Q/K 的 `[B,4,H,128]` 与 V 的 `[B,4,HV,128]` 多行源步长，尤其 T<4 时不能读取下一条请求的 token。

```bash
for variant in 9 11; do
  for stage in smoke graph; do
    python3 experimental/replayssm_sd_ascendc/run.py --backend framework \
      --stage "$stage" --tp 8 --batch 2 --tile-v 64 --cores 40 \
      --variant "$variant" --device 0 \
      --output "/tmp/sd-v11-tp8-v${variant}-${stage}.json" || exit 1
  done
done
```

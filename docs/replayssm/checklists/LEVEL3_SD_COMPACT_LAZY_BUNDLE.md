# L3.5：SD compact 延迟初始化与打包元数据消融

环境：Ascend910B4、CANN 9.2.0、torch_npu 2.10.0.post4、vLLM 0.30.0+empty、源码编译的 vllm-ascend 0.30。仍只测试独立 compact op；没有开启 `vllm serve` 的 SD/MTP dispatch。

L3.4 已上机确认 v1 的非 flush task duration（B32/B64、TP4、c40、tile32）为 23.3/45.5 µs，v0 为 75.9/141.3 µs；v1 的 flush 重路径没有显著退化。因此本轮把 v1 设为默认，v0/v2 保留对照。上一版的 `DataCopyExtParams` 收窄编译错误已回补。

L3.4 报告中把 MTE2 的下降称为“status 写”并不准确：MTE2 主要反映 GM→UB 读取，status 回写更接近 MTE3；v1 同时减少了元数据读取与小 status 写。两种引擎的计时还可能相互重叠。

| variant | 相对 v1 的唯一新增因素 | 非 flush 资源 | 元数据布局 |
|---|---|---|---|
| 1 | 已验证对照 | 全部 UB queues/tiles 在 Init 初始化 | 四个独立 int32[B] |
| 3 | 大块 UB 初始化推迟到本 block 第一次实际 Fold | 仅 32B metadata queue 与整行 status queue | 四个独立 int32[B] |
| 4 | 在 v3 上把四字段改为一次 16B `DataCopyPad` | 同 v3 | 四个列视图共享预打包的 int32[B,4] |

v4 为避免非 flush 时所有 value head 都读取 16B，head0 直接读 bundle，其余 head 先读各自的 flush 字段，仅 flush 时再读 bundle。v4 的打包在单算子计时区间之外，仅用于评估合并读取的内核收益；它**没有**测算打包本身的开销。现有真实框架四个张量独立，若最终采用 v4，必须让 cursor/metadata 生产者直接生成连续 `[B,4]`，或者把打包成本计入完整路径。不能把本轮 v4 单算子 p50 当成端到端收益。

## 上机顺序

1. 拉取本提交并重新源码编译主扩展。先确认 `replayssm_sd_compact` 的 CCE 编译与加载成功；保存首条错误和 UB 资源报告。v3/v4 在设备端首次实际 flush 时才初始化大 buffer，这一调用顺序需 CANN 9.2.0 编译与执行确认。失败即停止该 variant，不用旧 `.so` 测试。
2. 运行 CPU 参考测试和 Meta 测试。再按 v1→v3→v4 顺序做 TP4 B2 smoke、TP4 B2 graph、TP8 B2 graph；覆盖 pos0/13/16、base31 跨尾、无 flush、混合 flush、padding。每组检查 checkpoint、d/k/g ring 与逐元素 status；图重放后修改 metadata，结果必须更新。
3. 精度通过后测 TP4/TP8、B1/32/64、tile16/32、cores40 的 50 次 graph replay p50/p90/min。优先对比 `pos13_base30_flush0`，同时看 `pos13/16_base31_flush1` 是否因首次初始化或 UB 增量退化。再对 B32/B64、TP4、tile32、cores40 的 v1/v3/v4 抓 msprof task duration、Scalar/MTE2/MTE3；graph replay p50 和 task duration 分开报告。
4. 如果 v4 单算子明显优于 v3，再单独 profile 在图内从四个独立 int32[B] 生成 `[B,4]` 的实际成本，或给出将 metadata/cursor 生产者改成原生 interleaved 布局的方案。只有总成本降低才考虑采用 v4。若 v3/v4 不改善设备 task duration，保留已验证的 v1，不因 graph p50 小幅波动切换默认。

```bash
set -e
python3 -m pip install --no-build-isolation --no-deps -v -e .
python3 -m unittest experimental.replayssm_sd_compact_ascendc.tests.test_reference
python3 tests/ut/replayssm/test_framework_meta.py

for variant in 1 3 4; do
  python3 experimental/replayssm_sd_compact_ascendc/run.py \
    --backend framework --stage smoke --tp 4 --batch 2 --tile-v 32 \
    --cores 40 --variant "$variant" --device 0 \
    --output "/tmp/replayssm-compact-v${variant}-tp4-b2-smoke.json"
  python3 experimental/replayssm_sd_compact_ascendc/run.py \
    --backend framework --stage graph --tp 4 --batch 2 --tile-v 32 \
    --cores 40 --variant "$variant" --device 0 \
    --output "/tmp/replayssm-compact-v${variant}-tp4-b2-graph.json"
  python3 experimental/replayssm_sd_compact_ascendc/run.py \
    --backend framework --stage graph --tp 8 --batch 2 --tile-v 32 \
    --cores 40 --variant "$variant" --device 0 \
    --output "/tmp/replayssm-compact-v${variant}-tp8-b2-graph.json"
done

for tp in 4 8; do
  for batch in 1 32 64; do
    for variant in 1 3 4; do
      python3 experimental/replayssm_sd_compact_ascendc/run.py \
        --backend framework --stage perf --tp "$tp" --batch "$batch" \
        --tile-v 32 --cores 40 --variant "$variant" --iterations 50 --device 0 \
        --output "/tmp/replayssm-compact-v${variant}-tp${tp}-b${batch}-perf.json"
    done
  done
done
```

回传 source commit、`git status --short`、构建日志、所有 JSON、首个失败 traceback、NPU/board 健康状态和三版 profiler 摘要。v4 要注明预打包发生在计时区间外。TP4 通过后再扩展 tile16 和 TP8 性能；失败时不要继续对应 variant 的性能测试。

## Fusion 的依赖边界

目前只有独立生命周期对照脚本依次调用 `verify → compact → cursor`；`vllm serve` 的 SD/MTP dispatch 尚未接线，不能声称真实 serving 图已经把两算子紧邻排布。原始 ReplaySSM 生命周期也可表述为“用上一轮 acceptance 更新 cursor → 本轮 verify → 本轮 compact”，与脚本把 cursor 记在本轮尾部仅是迭代边界的表示差异。

非 flush/flush 的**设备端逐请求分支不阻碍融合**：融合核可在每个 `(row, value head, V tile)` 完成候选输出及 ring 写入之后，按 flush 选择是否把旧 `[0, position)` 历史折叠进 checkpoint。必须保证本轮 verify 始终读取折叠前的 checkpoint，不能让已更新的 tile 污染其它 tile 的计算；candidate `[position, position+T)` 与旧 history 在物理 ring 跨尾时也须证明不别名。融合的 UB 生命周期、跨 head 写入顺序、混合 flush、padding、状态码、图 replay 与 48 层调用计数都要重新验证。v3/v4 的结果用于决定融合时是否值得沿用轻量 metadata/UB 布局，不在本轮实现融合。

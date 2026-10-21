# Replay AR 调用次数与时延预算

## 模型与调用数

核对 [Qwen3.6-27B 官方 config](https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json)：
64 层中 48 `linear_attention`、16 `full_attention`，全局 GDN H/HV=16/48，K=V=128。
部署时仍以服务器实际权重 config 为准。`scripts/replayssm/call_budget.py` 可读取本地 config 复算。

纯 decode、一个连续 batch、每层一次当前 AR 算子的情况下：

| TP | 每 rank 的 GDN H/HV | 每 forward 每 rank 调用 | 全机调用总数 | 关键路径串行层数 |
|---|---|---|---|---|
| 4 | 4/12 | 48 | 192 | 48 |
| 8 | 2/6 | 48 | 384 | 48 |

batch=1/8/16 均是每 rank 48 次调用；每次调用处理整个 batch，不能再乘 B。
TP rank 并行执行，不能把全机调用次数当成单条关键路径。
若未来调度将同一层拆成多个 microbatch，则需乘实际 microbatch 数；混合 prefill/decode 不适用上述简化。
flush 已是同一次 kernel 的分支，每 L 步写回 checkpoint 不额外增加调用。
目前没有新增 GDN 层间通信；原有 TP 通信仍然属于整网时延。

## 独立算子实测与粗估

用户回传的 `3ff971b` 独立算子 `graph_perf` 报告均为 Ascend910B4、B=2、capacity=16、
tile_v=32、8个 AI Core、50次迭代。TP4 在 device 0，TP8 在 device 1。
该阶段同时验证了 NPUGraph capture/replay 正确性；下表性能数字来自 Python 算子调用前后的
NPU Event 区间，尚不是 capture 后的48层 graph replay。

| 几何 | Replay 空历史 flush p50 | Native AR 空历史 flush p50 | 单调用节省 | Native/Replay | 耗时降低 |
|---|---:|---:|---:|---:|---:|
| TP4 | 125.61µs | 204.85µs | 79.24µs | 1.631x | 38.69% |
| TP8 | 140.73µs | 224.67µs | 83.94µs | 1.596x | 37.36% |

这是报告中 `replay_ar_empty_history_flush` 与 `native_ar_empty_history_flush` 的同条件对照。
Replay 自身 `perf_posX_flushX` p50 为：TP4 的 pos0/pos8/pos15-flush 分别是
134.17/139.52/174.90µs；TP8 分别是150.04/149.79/150.76µs。

按每 rank 串行48个 GDN 层计算：

| 几何 | Replay 48层预算 | Native 48层预算 | 条件节省 |
|---|---:|---:|---:|
| TP4 | 6.029ms | 9.833ms | 3.804ms |
| TP8 | 6.755ms | 10.784ms | 4.029ms |

若只为观察量级，把上述 B=2 单算子差值与用户报告的整网 graph forward 对照，
3.80–4.03ms 相当于30/40/60ms的12.7–13.4%、9.5–10.1%、6.3–6.7%。
这些比例不是端到端收益预测；batch 与图内调度都不相同，也没有测过 B=1/8/16 的框架算子，
不能假定延迟恒定。6.03–6.76ms 是替换现有 GDN 递推部分的预算，不能简单加到现有
30/40/60ms上。
归一化、gating、conv、投影、其他层、调度、TP 通信和可能的重叠也会影响端到端差值。

现有 `event_timing` 在 eager Python 调用前后记录 NPU Event，区间可能包含 host 发送任务的间隙。
因此上述数字不能直接当成整网 graph 中纯 kernel 时长。框架版新的 graph_perf 会分别输出：

- 原有 eager Event 指标；
- 单算子 capture/replay 指标；
- 可选 `--stack-layers 48`：48套独立 layer 状态在同一 graph 内调用的总时延及摊销值。

48层 microbenchmark 包含真实算子的诊断 status 写入，但不包含完整模型的数据依赖和其他计算。
需要 TP4/TP8 分别测 B=1/8/16，再判断是否优化。这一阶段的性能足以继续正确性接入，暂不承诺收益。

## MTP / SD 的边界

当前 vLLM 0.30 的 `qwen3_5_mtp.py` 用 `layer_type="full_attention"` 构造 MTP 层；
该 drafter head 自身不额外调用 Replay GDN。target 仍有48层 GDN。
后续若实现多 token 融合 SD verify，预计每 rank 每个 verify forward 48次 SD 算子调用，
而不是48乘 draft-token数；每个接受 token 摊销约48/A次，A是该轮实际前进的 token 数。
这些是后续设计计数，当前单 token AR 算子不能直接承担多 token verify 或 rejection/commit。

按这批 B=2 Event 对照，若其余部分不变且没有执行重叠，30ms基线的 Amdahl 量级约为
1.145–1.155x；这仍依赖同 batch、同图模式以及 Native/Replay 覆盖范围完全一致。
先用框架版48层 graph replay 确认实际占比，再做收益判断。

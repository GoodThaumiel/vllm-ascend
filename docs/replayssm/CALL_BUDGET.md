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

## 用户实测数字的粗估

用户报告独立六组测试全部通过，Replay `perf_posX_flushX` 约150µs，原生 AR 约220µs；
准确 batch、TP、tile、cores、case 名称与计时方式待原始 JSON 确认。这些是用户报告的近似值。

```text
Replay: 48 * 150us = 7.20ms
Native: 48 * 220us = 10.56ms
差值:   48 *  70us = 3.36ms
单算子约 1.47x，耗时降低约31.8%（仅限同条件对照）
```

| 用户整网 graph forward | 假设所有48层均为150µs的 Replay 占比 | 假设替换可省3.36ms的比例 |
|---|---|---|
| B=1，30ms | 24% | 11.2% |
| B=8，40ms | 18% | 8.4% |
| B=16，60ms | 12% | 5.6% |

后两列是量级推算，不是端到端收益预测；没有测过 B=8/16 的算子，不能假定其延迟恒定。
7.2ms 是替换现有 GDN 递推部分的预算，不能简单加到现有30/40/60ms上。
归一化、gating、conv、投影、其他层、调度、TP 通信和可能的重叠也会影响端到端差值。

现有 `event_timing` 在 eager Python 调用前后记录 NPU Event，区间可能包含 host 发送任务的间隙。
因此150/220µs不能直接当成整网 graph 中纯 kernel 时长。新的 graph_perf 会分别输出：

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

若 GDN 占原始关键路径约35.2%，且该部分真能达到1.47x、其余不变，Amdahl 粗估整体约1.126x；
该数字同样依赖同 batch、同图模式和无重叠假设。先确认图内实际占比，再做收益判断。

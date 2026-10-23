# ReplaySSM AR/SD：论文与原 vLLM 实现的优化点、Ascend 验收方向

本文件以本地 `replayssm.pdf`（*ReplaySSM: Cache SSM Inputs, Not State*，第 4、5、7 节及附录 B/C/E）和 `ReplaySSM-main` 的 GDN 实现为依据。目标是对齐机制与收益方向，不把 H100/B300、FP32 state、单卡模型的速度倍率当作 Ascend910B4、TP4/TP8、BF16 state 的硬门槛。我们的主模型是 Qwen3.6-27B GDN；AR 功能保留但性能优化暂缓，优先 MTP+SD，DFlash 后置，双精度/低精度 checkpoint 暂不纳入。

## 论文实际优化了什么

| 路径 | 原生瓶颈 | ReplaySSM 的关键变化 | 应观察到的方向 |
| --- | --- | --- | --- |
| AR | 每 token 读、改、写整个 SSM state | 保存一个 checkpoint 和最近 `L` 个输入/修正；非 flush 只从 checkpoint+ring 算输出并追加小记录；满窗口才写 checkpoint | 全量 state **读取仍每步一次**，全量写回由每步一次降为约每 `L` 步一次；在 native kernel 受 HBM 流量约束的并发下，摊销 kernel 和整网 decode 加速 |
| SD/MTP | target 验证 `T` token 要保留逐 token 的完整 state 快照，拒绝后恢复；验证串行地产生中间 state | 候选输入写入环形 ring；验证期间只计算 `T` 个输出，不持久化中间 state；接受后推进 cursor，拒绝后覆盖尾部；flush 只合并此前**已接受**的记录 | 完整 state 快照存储/写流量不再随 `T` 线性增长；rollback 不复制大 state；verify kernel 随 `T` 的成本增长较慢；可承载更高并发，MTP 服务吞吐提高 |

AR 的核心不是缩小持久 state：checkpoint 仍是完整矩阵，额外还有 d/k/g ring。论文附录 C.3 的“约减半”指窗口内 **HBM state I/O**，不是进程显存占用。论文式 (13) 在 FP32 Mamba-2 state 上给出 `L=16` 时流量比约 0.58；GDN 与 BF16 的实际比例必须按本机维度、ring dtype 和 profiler 实测。现有 TP4 同构建显存探针中，native/Replay 物理 cache backing 相同，AR 不应以“节省 1 GiB”验收。

SD 有两种与 AR 不同的收益来源：消除 `T` 份 state 快照对显存容量和写流量的压力；利用 output-only 公式避免为每个草稿 token 生成完整中间 state。GDN 的草稿修正依赖前面的草稿，论文附录 B.3 用一个 `T×T` 上三角求解，再进行因果输出计算；不能把单 token AR kernel 重复 `T` 次当成论文的 SD 实现。原仓库的 GDN spec kernel 将 verify/flush 分开执行，另有图内 cursor commit/reset；它还复用现有 conv/MTP 调度及 accepted-token 结果。

论文在 FP32 state、`L=16`、CUDA Graph、调优过的原生 kernel 下测到：AR 在 B=256 的四个模型上 kernel 为 1.46–1.80×、整网每步为 1.20–1.47×；B=1 时 kernel 反而可能较慢，整网约持平。SD 用 MTP，`T=4`，两种 120B 模型在 B=512 时相对原生 speculative throughput 为 1.85×/2.10×；收益随并发和 `T` 增大。论文还报告 GDN 原生验证 kernel 在 T=8 的成本约为单步的 6.40×，Replay 为 1.11–1.33×（含 flush）。这些是**论文硬件与精度下的参考趋势**，不是本项目数值门槛。论文把低精度双 checkpoint 作为第三项独立优化；本项目当前仅用 BF16 checkpoint，不声称取得该项精度/速度收益。

## 原仓库实现与当前移植的对应关系

| 机制 | `ReplaySSM-main` 源码 | 本项目状态 |
| --- | --- | --- |
| GDN AR checkpoint+ring、重放、flush | `vllm/third_party/flash_linear_attention/ops/fused_recurrent_replayssm.py` | `csrc/kernels/replayssm_ar.cpp` 已有独立 AscendC/torch binding 与框架接入；真实 `vllm serve` 生命周期仍在 L2.3 验证 |
| AR 位置/flush、prefill 后起点和 block 生命周期 | `vllm/v1/attention/backends/gdn_attn.py`，`vllm/v1/worker/gpu_model_runner.py` | Ascend metadata/cache patch 已接入，TP4 capture 和共享 page stride 问题正在上机复验 |
| GDN SD 多 token output-only + causal 修正、ring、flush | `vllm/third_party/flash_linear_attention/ops/gdn_replayssm_spec_decode.py` | 尚未实现；需要新的 AscendC verify/flush 算子，不能复用 AR 单步算子替代 |
| SD accepted-token cursor commit/reset | 同上 `_advance_gdn_spec_cursors_kernel`、`_reset_gdn_replayssm_spec_cursors_kernel`；GDN metadata builder | 尚未实现；需图内固定地址更新并验证拒绝、复用、抢占后恢复 |
| 原论文服务测量 | `benchmarks/replayssm/e2e_decode_speedup.py`、`e2e_spec_decode_throughput.py` | 本项目以 `vllm serve`+HTTP 作为最终口径，保留三模式：native AR、native MTP、Replay MTP |

现有 TP4/BS1 同轮 profiler 里，48 层 native recurrent 合计 0.707 ms，而 Replay AR 合计 2.977 ms；现有 AscendC kernel 每步在 UB 中重建 state tile、串行回放历史。它证明 AR 尚未达到论文所追求的性能方向。应先消除这部分额外 vector/scalar 串行成本，再观察较高并发下流量下降能否转化为速度。早期 B=2 eager Event 单算子结果不能替代图内 48 层或服务端到端结果。

## 最终验收的主次顺序

1. **语义与生命周期先通过。** 同一输入下，native/Replay 的输出在约定 BF16 容差内；跨多个 `L=16` 窗口、flush、slot 复用、prefill→decode、graph capture/replay 正确。SD 另测全部接受、全部拒绝、部分接受、短 draft、flush 边界和恢复后重用；accepted length 与原生 MTP 接近，生成内容/质量无异常。AR1 提速不是 SD 的依赖；L3 SD 从独立算子开始，每一级上机验证后再接下一层。
2. **AR 机制证据。** 逐层记录 full-state read/write 次数和 HBM bytes：非 flush 不写 checkpoint，flush 才写；缓存尾部不应被误读。把 ring 的读写也计入摊销总流量。对 `L=8/16/32`、ring 位置 0/8/15 和 BF16 state 做 profiling；预期存在写流量下降，性能最优 `L` 由 910B4 实测决定。
3. **AR 服务收益。** 在同一源码、CANN、权重、TP、cache 布局和图配置下，比较 `vllm serve` 的 native/Replay；分别报告 GDN 算子摊销 p50/p90、整网 TPOT p50/p95、生成 tokens/s。按 BS=1/8/16/32 与可承载的更高并发，以及 4k/8k/32k 上下文分层。BS1 可接近持平，不要求复制论文的高并发倍率；若高并发依然慢，视为性能目标未达成，不因流量下降就判通过。
4. **SD 机制证据。** 同一个 MTP drafter、`T=2/4/6/8` 和相近 accepted length 下，核对 target 每轮不生成/保存 `T` 份完整 checkpoint；非 flush 约一次 checkpoint 读、零次 full-state 写，flush 增加一次写。记录 snapshot/ring/graph backing、`num_blocks`、最大可服务并发，以及随 `T` 的 verify kernel 时延曲线。只比较真实物理 backing 和实际可用并发，避免把重叠 page 的逻辑尺寸相加。
5. **SD 服务收益。** 同机同配置比较 native AR、native MTP、Replay MTP；主指标是生成 tokens/s、TPOT/尾延迟、最大稳定并发和 MTP 接受长度。论文最核心的方向是：随着 `T` 和并发增大，Replay MTP 相对 native MTP 不因快照流量/容量提前饱和，并且在目标负载上有可复现的吞吐收益。DFlash 在 MTP 通过后复用这套 target-side 验收。

若 910B4/TP/BF16 或模型规格使论文的高并发点不可运行，记录 OOM 或容量限制，并用**两方案共同可运行的最大并发**比较；再单独报告各自最大并发。报告不得跨 CANN/vLLM 版本求加速比，也不能用只标记了 4k/8k/32k 的单步算子脚本推断上下文长度收益：单 token GDN 算子不随上下文长度改变，长上下文的变化来自服务级 attention/KV 与调度。

## 每次上机提交应附的证据

- 提交 SHA、安装的 `vllm`/`vllm-ascend`/CANN/torch_npu 版本、TP、BF16 cache/state、图模式、模型 config 和 profiler 原始文件。
- AR：正确性矩阵、各 ring 位置含 flush 的图内算子分布、native 对照的 full-state/HBM 流量、`vllm serve` 同构建 BS/上下文矩阵。
- SD：上述 AR 基础上增加 `T`/acceptance 矩阵、拒绝与 flush 交叉测试、snapshot/ring/物理显存及最大并发、三模式 `vllm serve` 吞吐。
- 分开写“机制已证实”“整网收益已证实”“尚未达标”；不以单算子通过或理论字节数代替最终性能验收。

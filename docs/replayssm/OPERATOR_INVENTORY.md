# ReplaySSM GDN 算子清单

## 范围与拆分原则

目标是 Qwen3.6-27B 的 Gated DeltaNet（GDN），运行环境为 vLLM 0.30.0、
vLLM Ascend 0.30.0、CANN 9.2.0 和 Ascend910B4。checkpoint、Mamba cache 与模型激活
使用 BF16；Replay d/k ring 使用 FP16，g ring 使用 FP32。

原版 ReplaySSM 的 Triton AR/SD kernel 把 q/k L2 norm、g/beta 预处理和 recurrent core
放在一个 kernel 内。Ascend 0.30 路径已经有经过维护的 causal conv、gating、L2 norm 和
原生 recurrent 算子。移植时保留现有前后处理，只在 recurrent 边界替换 Replay 算子：

```text
projection
  -> causal_conv1d_update                 复用现有 AscendC
  -> split/rearrange q,k,v                复用现有视图/拷贝
  -> fused_gdn_gating(a,b,A_log,dt_bias)  复用现有 Triton-Ascend
  -> l2norm(q), l2norm(k)                 复用现有 Triton-Ascend
  -> recurrent core                       AR/SD Replay 算子替换点
  -> gated norm + output projection       复用现有路径
```

这样不会重新实现已有的官方调优算子，也让 Replay 与 native 的 recurrent 性能对照保持同一边界。

## 公共算子

| 编号 | 算子/阶段 | 0.30 Ascend 实现 | 决策 | 修改性质 |
|---|---|---|---|---|
| C1 | QKVZ/BA projection | vLLM Ascend 现有 GEMM/量化路径 | 原样复用 | 无修改 |
| C2 | decode/spec causal conv | `fla_npu.ops.ascendc.causal_conv1d_update` | 原样复用；该接口已支持 `query_start_loc` 和 `num_accepted_tokens` | 无算子修改 |
| C3 | q/k/v split、layout | `vllm_ascend.ops.gdn._rearrange_decode_qkv` | 原样复用 | 无算子修改 |
| C4 | `g=-exp(A_log)*softplus(a+dt_bias)`、`beta=sigmoid(b)` | `DeviceOperator.fused_gdn_gating`，910B 走现有 Triton-Ascend | 原样复用 | 无算子修改 |
| C5 | q/k L2 norm | `vllm_ascend.ops.triton.fla.l2norm.l2norm_fwd` | 原样复用 | 无算子修改 |
| C6 | prefill GDN | FLA AscendC fused prefill 或现有 Triton chunk pipeline | Replay 只改变 decode；prefill 继续生成 checkpoint | 无算子修改 |
| C7 | 输出 gated norm 和 out projection | `self.norm(core_attn_out, z)` 及现有 projection | 原样复用 | 无算子修改 |

如果后续端到端 profile 显示 C4/C5 的独立 launch 成为瓶颈，再考虑与 Replay core 融合；
它不属于首版正确性接入。

## ReplaySSM + AR

| 编号 | 算子 | 设计来源 | 当前状态 | 还需工作 | 修改性质 |
|---|---|---|---|---|---|
| AR1 | 单 token Replay recurrent step | 新 AscendC `replayssm_ar_step_`；计算边界对应 native `recurrent_gated_delta_rule` | framework F1-F3 已由用户确认通过 | 在真实模型 profile 后决定是否优化 TP4/B16 | 新增插件算子，低侵入 |
| AR2 | AR cursor/flush 元数据 | 原版 ReplaySSM builder 在 host 计算 `position % L` 和 flush | 已实现，等待 L2.3 上机验证 | 验证 graph padding、恢复、slot 复用和 align block 边界 | 框架元数据修改，中侵入 |
| AR3 | Replay ring/checkpoint 分配 | 原版在 GDN state tuple 后追加 d/k/g | 已实现五块 cache 契约，等待 L2.3 上机验证 | 验证真实 TP4/TP8 分配和 block copy/reuse | cache 布局修改，高侵入 |
| AR4 | GDN decode dispatch | 用 AR1 替换 non-spec decode 的 native recurrent op | pure/mixed non-spec decode 已接入，等待 L2.3 上机验证 | `vllm serve` 对照原生 AR；spec 路径仍主动拒绝 | 模型 dispatch 修改，中侵入 |

AR1 当前接口直接接收已经完成 L2 norm 的 BF16 q/k/v，以及 FP32 g/beta。它完成：

- 从 BF16 checkpoint 和 FP16 d/k、FP32 g 历史重建状态；
- 计算当前 delta-rule 更新和输出；
- non-flush 时追加 d/k/g；
- flush 时将更新后的 BF16 state 写回 checkpoint；
- padding、物理 slot、position/flush 校验与 NPUGraph 固定地址调用。

它没有重新实现 C1-C7。原 Triton Replay AR 把 C4/C5 融在 recurrent kernel 内，
所以两者的 Python 签名不同，但 recurrent 数学边界一致。

当前 AR1 是通过独立 AscendC 实现完成的，不是对 FLA 原生 RGDR 源文件的逐行补丁。
早期 B=2 eager Event 对照不能代表图内模型性能。用户后续在同容器开关 Replay AR 的服务测试
报告 baseline 快约 8%；roofline 与 batch 扫描表明本机 native recurrent 算子偏 vector compute
bound，Replay AR 算子各 batch 均较慢且差距随 batch 增大。AR1 性能优化暂缓，保留功能和独立
测试作为对照；不把 AR1 提速设为进入 SD 的前提。若未来恢复 AR 优化，应重新评估输出型公式
与 FLA 原生 tiling，而不是以减少 state 写流量必然换来提速作为假设。

## ReplaySSM + SD（MTP 优先）

Qwen3.6 的 MTP draft layer 是 full attention；Replay 算子运行在 target 的48个 GDN 层。
现有 0.30 Ascend speculative causal conv 已根据上一轮 `num_accepted_tokens` 更新 conv state，
因此 C2 不需要重写。缺失项如下：

| 编号 | 算子 | 最接近的已有实现 | 实现决策 | 状态 | 修改性质 |
|---|---|---|---|---|---|
| SD1 | 多 token Replay verify recurrent | FLA AscendC `recurrent_gated_delta_rule`；原论文仓 Triton `gdn_replayssm_spec_circular_kernel` | AscendC `T≤4` 输出与 d/k/g 写入；扩展到 history≤16、history+T≤20；主扩展已注册 | L3.1 用户回传旧非 flush 窗口 18 组 graph/perf 全过；新窗口待上机 | 新增插件算子，高工作量 |
| SD2 | flush/compact | 原论文仓 SD1 同源的 flush specialization | 首版先执行 SD1，再用独立 AscendC compact 折叠旧已接受历史；graph 中固定两次 launch，按 per-row flush 路由 | 独立算子、CPU golden 与主扩展注册已完成，待 CANN/NPU | 新增插件算子，中高工作量 |
| SD3 | accepted-token cursor commit | 原论文仓 `_advance_gdn_spec_cursors_kernel` | AscendC 更新 block-keyed `write_pos/cache_base/is_flush`；与 reset 共用一个入口 | 独立算子、CPU golden 与主扩展注册已完成，待 CANN/NPU | 新增插件算子，低工作量 |
| SD4 | 新请求/恢复时 cursor reset | 原论文仓 `_reset_gdn_replayssm_spec_cursors_kernel` | SD3 算子的 `do_reset` 分支；reset 覆盖旧 accepted | CPU golden 已完成，待 CANN/NPU | 同 SD3，无额外 kernel |
| SD5 | spec output merge | 0.30 Ascend 的 `index_copy_`/现有 GDN merge | 原样复用，profile 后再决定是否融合 | 未接 Replay，但无需新算子 | 框架调用修改，低侵入 |
| SD6 | MTP accepted-count/scheduler | vLLM 0.30 MTP/spec scheduler 和 Ascend metadata builder | 原样复用 accepted count；只把它接到 SD3 和现有 conv | 未接 Replay | metadata 接线，中侵入 |

SD1/SD2 是同一数学主体的两个执行分支，不应误算成两套独立算法。对应原版 Triton 实现的工作包括：

- packed variable-length verify window；
- checkpoint 加已接受历史的 circular d/k/g replay；
- 一次输出整个 MTP verify window，不持久化每个候选 token 的完整 state；
- spec window 内的因果 delta-rule/三角修正；
- 将候选 d/k/g 写入 ring，拒绝 token 只通过 cursor 丢弃；
- flush 时把旧的已接受历史压入 checkpoint，同时保留本轮可能接受的候选 ring；
- device-side per-row routing 和 NPUGraph 固定 launch。

DFlash 后续复用 SD1-SD4 的 target GDN 数学算子。预期新增工作是把 DFlash 的 packed verify
metadata 适配到相同接口，而不是再写一套 recurrent kernel；只有实际 shape/acceptance 语义不兼容时
才新增 adapter。

## 六组测试实际完成的范围

用户确认通过的是 commit `3ff971b` 的独立 AR1 AscendC 算子，加服务器端 CMake/RPATH 修复。
测试程序用单卡模拟 TP4/TP8 的本地 head 几何，不是4/8进程 HCCL tensor parallel。

已覆盖：

- position 0、中间历史和满窗口 flush；未提交 cache tail 使用 NaN poison；
- checkpoint、d/k/g mutation 与 FP64 CPU closed-form golden；
- padding、slot 重排、slot 复用、超过两个窗口的 lifecycle；
- 非法 slot/position/flush 的状态码；
- current stream、NPUGraph capture/replay 和不同 device-side cursor 值；
- TP4/TP8 本地 head 几何、B=2、L=16、tile_v=32、cores=8；
- 与 FLA native recurrent op 的同边界空历史 flush 性能对照。

framework F1-F3 已在 commit `2808394` 对应源码上由用户确认通过。尚未覆盖：

- C2+C4+C5+AR1 的整条 Qwen GDN decode 模型调用；
- AR2-AR4 在真实调度中的缓存分配、metadata、dispatch 和 block 生命周期；
- 真实 TP4/TP8 多进程、B=1/8/16、48层 capture；
- `vllm serve` 的 Replay AR 输出和长序列生命周期；
- SD1-SD4 在主扩展中的 CANN/NPUGraph 验证、MTP accepted/rejected token、提前 flush 和 rollback；
- prefix cache、preemption/resume、KV connector 和 cache block copy 对新增 ring 的处理；
- DFlash。

因此，F1-F3 可以判定“AR recurrent 算子已完成 framework 打包且图内可运行”，不能判定
“ReplaySSM + AR 性能目标已完成”。用户已在同容器测试发现 AR 相对 baseline 的回退，并确认
32 并发、约 32k 输入时因显存出现 waiting；项目优先转向 SD 容量与 MTP 吞吐。SD1 的独立
AscendC 非 flush 核已获 L3.1 用户上机通过报告；flush compact 与 accepted-token cursor 的
独立实现及框架注册已写入源码，但尚无 CANN/NPU 编译和图捕获验证，因此暂不接入 `vllm serve`
的 SD dispatch。compact 与原论文单一 flush specialization 的计算分解不同，flush 时先 verify
再折叠旧历史，必须以设备 profile 确认额外读写和 launch 的成本。

## 工作量与主要不确定性

| 工作项 | 相对工作量 | 是否依赖上机 | 主要不确定性 |
|---|---:|---|---|
| AR1 framework F1-F3 | 已完成 | 是 | TP4/B16 性能优化推迟到真实模型 profile 后 |
| AR2-AR4 + AR serve | 中 | 是 | 0.30 cache 生命周期、graph metadata 固定地址、block 复用 |
| SD1/SD2 独立 AscendC | 大 | 是，多轮探针 | BS×BS 三角计算的 UB/tiling、BF16/FP32误差、flush 分支性能 |
| SD3/SD4 cursor | 小 | 是 | 同一物理 block 唯一性、图内更新顺序 |
| SD framework + MTP serve | 大 | 是 | accepted-token 时序、rollback、混合 spec/non-spec batch、preemption |
| DFlash adapter | 中，后置 | 是 | DFlash packed metadata 与 MTP 接口差异 |

需要外部协助的点主要是 Ascend910B4 编译/运行、CANN 9.2 profiling，以及在真实模型上验证
MTP acceptance、preemption 和 prefix-cache 生命周期。若 FLA 打包版本的 native RGDR 源码与
公开源码不一致，还需要确认商发包对应 source revision，避免在错误基线上做增量优化。

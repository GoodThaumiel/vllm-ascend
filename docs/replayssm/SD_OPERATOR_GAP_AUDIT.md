# ReplaySSM SD 单算子边界与缺口审计

本文件核对原始 ReplaySSM 的 GDN speculative 路径、当前 vLLM Ascend 0.30 路径和已完成的 SD verify pilot。目标模型为 Qwen3.6-27B，MTP 优先；算子几何为 K=V=128，TP4/TP8 每卡分别 H=4/2、HV=12/6，checkpoint 和 Mamba cache 基线 BF16。本文只界定单算子范围，不代表 `vllm serve` 已接入。

## 算子矩阵

| 边界 | 当前实现 | 本级动作 | 性能对照中计数 |
|---|---|---|---|
| speculative causal conv | `fla_npu.ops.ascendc.causal_conv1d_update` 已接收 `num_accepted_tokens` | 复用，不重写 | 两侧公共前处理；单算子 recurrent 对照均排除 |
| q/k/v 拆分、g/beta、q/k L2 norm | Ascend GDN 现有视图/拷贝、`fused_gdn_gating`、`l2norm_fwd` | 复用，不重写 | 两侧公共前处理 |
| SD1 非 flush 多 token recurrent verify | `_C_ascend.replayssm_sd_verify_`，AscendC，支持 T≤4、padded `[B,4]`、page-strided ring | 已打包；等待 L3.2 框架上机 | Replay verify 一次；原生 `recurrent_gated_delta_rule` 一次 |
| SD2 flush/compact recurrent verify | 原论文同一 Triton 数学核的 flush specialization | 现实现先用 SD1 对所有有效行算输出/候选，再用独立 AscendC compact 对 flush 行折叠旧历史；候选不提前提交 | 每层固定计入 verify+compact 两次 launch，包括非 flush 的 compact 空路径 |
| SD3 accepted-token cursor commit | 原论文 `_advance_gdn_spec_cursors_kernel` | 新图内小算子，按物理 block 更新 `write_pos/cache_base/is_flush`，拒绝的候选仅由 cursor 排除 | 每批一次，不按 GDN 层数乘 48 |
| SD4 first-decode/resume cursor reset | 原论文 `_reset_gdn_replayssm_spec_cursors_kernel` | 与 SD3 共用 AscendC kernel，以 `do_reset` 分支覆盖旧 acceptance | 每批一次，不按 GDN 层数乘 48 |
| spec/non-spec 输出 merge | Ascend GDN 的 `index_copy_` 路径 | 复用现有实现 | 端到端包括；recurrent 单算子对照排除 |

SD1/SD2 是同一个 delta-rule 的非 flush/flush 执行路径，不能按两个不同数学算法验收。现有 SD1 独立上机报告没有覆盖 SD2–SD4；新算子虽已落盘，尚未 CANN/NPU 编译，也未验证 packed-token serving 接线。

## 必须保持的缓存和 cursor 语义

原版把 speculative cursor 放在物理 GDN block 上，而不是 batch row 上；同一个请求跨 step 改变行号时仍必须取同一个 block 的 cursor。以 B 为 Replay 已接受历史目标长度、T 为最大 verify 长度，逻辑 flush 阈值 `L=B+T`，物理 ring 容量 `R=next_pow2(L)`。B=16、T=4 时 L=20、R=32；当前独立 pilot 的 position ≤16、ring=32 覆盖了 verify 算子范围，但不能代替完整 cursor 生命周期。

每个 step 先以**上一 step** 的 `num_accepted_tokens` 执行 SD3，再对 first-decode/resume 的 row 执行 SD4，最后运行本 step 的各 GDN 层 verify/flush。原版 commit 规则如下，`accepted` 已包含 bonus token：

```text
flush_now = accepted > 0 && is_flush
new_base = flush_now ? (cache_base + write_pos) % R : cache_base
new_write_pos = is_flush ? accepted : write_pos + accepted
next_is_flush = new_write_pos + 2*T > L
```

reset 令 `write_pos=0`、`cache_base=0`、`is_flush=(2*T>L)`。原版由两次 specialization launch 在设备端按 `is_flush` 路由；当前实现固定执行 verify，再让 compact 按 per-row `flush` 选择有效行，图捕获不读取 host 上的 flag。flush 时 checkpoint 吸收**本轮之前**的已提交历史；本轮候选依然写入 ring，待下一 step 的接受数决定前缀，拒绝部分不进入 checkpoint。padding/null block 不可读写。

## 单算子完成之后仍需的框架适配

Ascend 0.30 的 speculative metadata 持有 packed token、`query_start_loc` 和每候选独立 state slot `spec_state_indices_tensor[B,T]`；原生 recurrent 用 flatten 的 candidate slot。Replay 使用每请求一个物理 block id 和对应的 block-keyed cursor，而当前 pilot 接口是 padded `[B,4,...]`、`slots[B]`、`lengths[B]`。因此接入 `vllm serve` 前仍需图内固定地址的 pack/scatter 或扩展 kernel 输入协议，按 request 取 block id，并处理混合 spec/non-spec batch、图 padding 和空 spec step。它们是 metadata/dispatch 适配工作，不是新增 recurrent 数学算子。

当前 Ascend AR cache 只按 `replayssm_buffer_len` 分配 d/k/g ring；SD 需要按 `R=next_pow2(B+T)` 修改 GDN cache 形状和状态复制/重用契约。用户指定的 BF16 checkpoint/conv state 与 FP16 d/k、FP32 g ring dtype 保持不变。SD1–SD4 算子通过并不能自动放开现有 speculative guard；必须上机验证缓存分配、prefill→decode、preemption/resume、block 复用及 NPUGraph capture/replay。

## 小范围性能对照口径

先在 TP4/TP8、B=1/16/32、T=1/4、history=0/7/12 复测单层，同一个 q/k/v/g/beta 和等价初始 BF16 状态对比原生 recurrent 与 Replay。现有 `profile_sd_operator_compare.py` 只比较 **native recurrent 一次** 对 **Replay 非 flush verify 一次**；它只能归因 SD1，不能作为 SD 完整路径加速比。

SD2–SD4 补齐后再增加完整单层/一步对照：Replay 计入 verify 与 flush 两次固定 launch；cursor commit/reset 单独记录，并按一次 batch step 分摊，不能误乘 GDN 48 层。分别量测 `is_flush=0/1`、accepted=1/2/4、拒绝尾部、环形跨尾和无效/padding row。报告设备 kernel p50/p90、HBM/UB bytes、Vector/Scalar/Cube 利用率及内存 backing；必须区分算子设备时延与 host dispatch 开销。最后才用 `vllm serve` 测 32k 上下文、32/64 并发和 waiting/吞吐收益。

参考源码：`ReplaySSM-main/vllm/third_party/flash_linear_attention/ops/gdn_replayssm_spec_decode.py`、`ReplaySSM-main/vllm/v1/attention/backends/gdn_attn.py`、`vllm_ascend/ops/gdn.py`、`vllm_ascend/ops/gdn_attn_builder.py` 和 `vllm_ascend/ops/replayssm/cache.py`。

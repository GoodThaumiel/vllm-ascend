# Level 2：ReplaySSM AR算子设计

## 边界与侵入性

本阶段只在vLLM Ascend plugin新增GDN ReplaySSM AR算子、数据契约和单算子验证工具。
不修改vLLM 0.30.0源码，不patch Qwen模型，不接cache allocator、metadata builder或scheduler。
因此当前改动为**低侵入性、可旁路**；原生AR/MTP和`vllm serve`不会调用新增路径。

第三级接入模型与引擎时需要中高侵入性的plugin patch，至少涉及：

- Qwen GDN linear attention的decode dispatch；
- GDN cache shape/dtype扩展与slot生命周期；
- `write_pos`/`is_flush`图输入和请求reset；
- TP worker初始化及配置门槛。

这些接入点在第二级算子通过前不实现。

## 数据契约

Qwen目标几何默认按全局key heads=16、value heads=48、K=V=128；TP4/TP8本地几何分别为
`H/HV=(4,12)`和`(2,6)`。实际模型config若不同，上机前必须按config修正验证几何，不能只靠模型名称推定。

| 张量 | shape | dtype | 说明 |
|---|---|---|---|
| mixed_qkv | `[B, 2*H*K + HV*V]` | BF16 | 当前token的q/k/v |
| a, b | `[B, HV]` | BF16 | decay和beta输入 |
| checkpoint | `[slots, HV, V, K]` | BF16 | 持久SSM state；slot 0保留给padding |
| d_cache | `[slots, HV, L, V]` | FP16 | delta update历史 |
| k_cache | `[slots, H, L, K]` | FP16 | L2-normalized key历史 |
| g_cache | `[slots, HV, L]` | FP32 | log decay历史 |
| state_indices | `[B]` | INT32 | `<=0`为padding，正值为物理slot |
| write_pos | `[B]` | INT32 | 当前逻辑历史长度/写位置 |
| is_flush | `[B]` | INT8 | 当前token是否折叠进checkpoint |
| out | `[B,1,HV,V]`或`[B,HV,V]` | BF16 | 当前token GDN输出 |

Mamba conv cache和checkpoint遵循用户BF16部署基线。Replay `d/k`缓存使用原实现的FP16存储，
因为BF16激活下FP16具有更多尾数位；`g`保持FP32以控制长乘积误差。所有状态重建、归约和递推在kernel内升FP32。

## 数学与生命周期

对每个value head，checkpoint为`S0[V,K]`。历史token保存`(d_i,k_i,g_i)`：

```text
S = exp(sum_i g_i) * S0
  + sum_i exp(sum_{j>i} g_j) * outer(d_i, k_i)

alpha = exp(-exp(A_log) * softplus(a + dt_bias))
d_cur = sigmoid(b) * (v - alpha * S @ k)
S_cur = alpha * S + outer(d_cur, k)
o = S_cur @ (q * scale)
```

非flush token把`d_cur/k/g`写入`write_pos`，checkpoint不变。flush token把`S_cur`写回BF16 checkpoint，
不写历史；调用方随后把`write_pos`重置为0。逻辑位置`L-1`必须flush，也允许为对齐或生命周期边界提前flush。
request slot复用时，调用方必须重置checkpoint和cursor；旧历史数据无需清零，因为`write_pos=0`时不可见。

## Triton-Ascend映射

实现保留论文源码的K分片重建：`BV=64`，K按`nk=2`分成`BKT=64`，历史长度默认L=16并pad到
`BC=16`。`d_scaled[BV,BC] @ k_cache[BC,BKT]`使用`tl.dot`，其余递推和归约使用FP32。
CUDA参考的三维grid被展平为一维`NV * B * HV`，满足Triton-Ascend启动约束。

当前是**正确性候选**，每个输出V tile启动一个program，尚未按AI Core数量做persistent任务循环。
原因是先验证`tl.cumsum + tl.dot + runtime flush branch`在CANN 9.2.0/triton-ascend 3.2.2上的编译和图行为。
通过后再根据实测决定：

1. Triton路径增加核数钳制和任务循环；
2. recurrent reconstruction/flush若成为热点，迁移为AscendC；
3. 保留当前Triton实现作图模式正确性基线与fallback。

## 已知不确定性和外部验证点

| 项目 | 当前判断 | 需要的外部验证 |
|---|---|---|
| Triton编译 | 中风险 | 910B4上确认runtime分支、`tl.cumsum`和二维`tl.dot`能编译 |
| checkpoint二维写回 | 中风险 | Qwen K/V=128时mask为完整tile；仍需检查flush后的全量state误差 |
| NPUGraph | 中风险 | capture后改变输入和flush flag，确认三次replay不是旧输出 |
| 精度 | 中风险 | 与模拟BF16/FP16存储的参考比较，并单独记录相对直接FP32递推的漂移 |
| 性能 | 未知 | 第二级先记录延迟，不设收益门槛；若显著慢再做核数钳制/AscendC |
| 实际Qwen3.6 config | 待服务器确认 | 核对全局H/HV/K/V；默认几何来自当前Qwen3.5同构基线 |

L2.1上机步骤与通过条件见[第二级清单](checklists/LEVEL2.md)。

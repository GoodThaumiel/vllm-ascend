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
第三版用于验证`tl.cumsum + 显式remask后的tl.dot + runtime flush branch`在
CANN 9.2.0/triton-ascend 3.2.2上的融合编译、精度和图行为。
通过后再根据实测决定：

1. Triton路径增加核数钳制和任务循环；
2. recurrent reconstruction/flush若成为热点，迁移为AscendC；
3. 保留当前Triton实现作图模式正确性基线与fallback。

## 已知不确定性和外部验证点

| 项目 | 当前判断 | 需要的外部验证 |
|---|---|---|
| Triton编译 | 中风险 | 隔离构造已确认；仍需910B4上确认第三版融合kernel能编译和执行 |
| checkpoint二维写回 | 中风险 | Qwen K/V=128时mask为完整tile；仍需检查flush后的全量state误差 |
| NPUGraph | 中风险 | capture后改变输入和flush flag，确认三次replay不是旧输出 |
| 精度 | 中风险 | 与模拟BF16/FP16存储的参考比较，并单独记录相对直接FP32递推的漂移 |
| 性能 | 未知 | 第二级先记录延迟，不设收益门槛；若显著慢再做核数钳制/AscendC |
| 实际Qwen3.6 config | 待服务器确认 | 核对全局H/HV/K/V；默认几何来自当前Qwen3.5同构基线 |

L2.1上机步骤与通过条件见[第二级清单](checklists/LEVEL2.md)。

## L2.1硬件诊断与第三版设计

commit `26dd021293bf52d2743329ddb1c6f3c2afe48688`的首轮TP4验证先暴露了
Triton-Ascend 3.2.2的链式布尔AST限制。将三操作数`and`拆成两个二操作数节点后kernel能够编译，
但在step 0第一次同步时稳定触发`507014` AI Core timeout，同时报告AICore、AIVector和MTE错误。
`write_pos=0`时历史cache读取均被mask，这使历史数据越界成为较弱解释；故障发生在进入精度比较之前。

commit `16107d811`的六个隔离探针改变了首轮判断：`tl.cumsum`、串行prefix、普通dot、运行时分支和
masked二维I/O均能独立运行；只有`masked_zero_dot`错误。第二版采用的串行prefix虽然独立通过，却在
完整融合kernel中降低为不支持的`hivm.hir.copy`，使TP4在编译阶段失败。第三版因此恢复首版已能融合编译的
`tl.cumsum`。隔离通过不等于已证明它在融合执行时无问题，但现有证据不再支持把首轮timeout直接归因于它。

`masked_zero_dot`进一步由commit `4c003b4ac`的六路探针定界为方向性缺陷：只有归约维masked load作为
`tl.dot`左操作数时失败；右操作数mask及四种单kernel显式清零方案均通过。由于`BC`可能向上取整并超过
物理cache capacity，第三版先按物理容量mask安全加载`d_cache`，再以`tl.where`显式清零逻辑无效历史和V尾行，
随后才将其作为dot左操作数。这对应已通过的`remask_after_masked_load`路径，保留原矩阵方向和单kernel结构，
不增加中间物化存储。

第三版也纳入用户发现的探针runner参数修复：父进程为子进程补传argparse要求的`--output`，实际子结果仍写入
`--child-report`。以上修改均不改变接口、dtype、缓存布局和数学口径，仍属于**低侵入性算子修复**；没有进入
模型、cache allocator、metadata builder或scheduler。

为避免下一次只能从整核超时反推，新增六个单构造探针，并将每个探针绑定到独立逻辑卡、并行执行：

| 探针 | 隔离的问题 |
|---|---|
| `cumsum` | 原生`tl.cumsum`在长度16时是否直接trap或错算 |
| `serial_prefix` | `get_element + tl.where`替代路径；隔离通过但融合编译失败 |
| `dot` | Replay实际tile `[64,16] @ [16,64]` |
| `masked_zero_dot` | step 0历史长度为0时，全mask输入进入同shape `tl.dot` |
| `runtime_branch` | device标量控制的运行时if两个分支 |
| `masked_2d_io` | `[64,64]` tile的尾部masked二维load/store |

这些探针是**诊断代码**，不会进入生产调用路径。第三版仍需重新完成TP4/TP8整核编译、精度、flush/reset和
NPUGraph验证；探针通过不能单独证明融合kernel已修复，也不能证明首轮`507014`只有一个根因。

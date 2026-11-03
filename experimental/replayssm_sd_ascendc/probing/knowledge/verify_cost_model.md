# Verify cost model：待实测校准

本文件是模型模板，无新实测结果。
令N=B×12，C=min(cores,N)，p=history，T=candidates；v20 tile128。
源码结构可写为每task固定工作 F(T)，p>0启用的固定16行correction J(T)，
随p变化的dot/suffix/DMA工作 L(p,T)，以及随p减少的D尾部补零 Z(16−p)。
关系用于设计实验，不预设各项时延可加：不同pipe重叠，依赖决定关键路径。

对同case分别观察task/vec/scalar/mte2/mte3：

- D01=t(p1)−t(p0)：固定路径启用与一行history工作、补零减少的混合净变化。
- D715=t(p15)−t(p7)：有效行增量、repeat变化和相应等待的混合净变化。
- Dprobe=t(control)−t(probe)：包含删除引起的排布/重叠/资源变化，不是阶段独占时间。

拟合每种pipe的经验趋势 a+b·I(p>0)+c·p（必要时添加跨尾项）；不把拟合系数称作硬件API延迟。
用p0/1/7/15拟合，p2/4/8/12留作预测检验；记录预测区间、残差、每轮漂移。
同一模型不混合T1/T4、B32/B64或不同base；拟合不等于证明线性模型成立。
微基准斜率与编译指令差异仅约束解释范围，不能将ceil(N/C)乘单task时延当精确预测。

| Top假设 | 证据等级 | 当前依据 | 证伪方式 |
| --- | --- | --- | --- |
| p>0固定history归约导致跳变 | inferred | 源码有固定16行路径 | A2保依赖control对照，p0负对照 |
| dot/suffix/搬运导致p7→15增量 | unverified | 三者均随p变化 | A1、按需A3、U1/U2；检查非目标删除 |
| 全kernel依赖/调度放大局部成本 | unverified | 尚无同口径局部实测 | L1重复斜率及task/core缩放 |

归因验收：至少两类成本有可复现实验和编译证据；无法解释的残差明确保留。

# V20 源码操作地图与调用数审计

依据 commit `5b0a2b0` 的 `csrc/kernels/replayssm_sd_verify.cpp`；以下为源码计数，不是实测指令数。
入口 `replayssm_sd_verify_v20_chunk8k_gate_batch.cpp` → `ReplaySDVerify<20>::Process` → `ProcessSharedHeads`。
binding `csrc/replayssm_sd.cpp` → `replayssm_sd_verify_`；V20 可传独立 `ring_slots`，省略时与checkpoint slots共用编号。

定义：p=position=有效历史条数，不是物理写下标；base=最早有效历史的环形下标。
历史位置 `(base+i)%32, 0<=i<p`；候选写 `(base+p+s)%32, 0<=s<T`。
checkpoint slot、request ring slot、batch row是三个不同的索引，实验不得混用。

单task=(row,value_head)，tile128时只有一个V tile。N=B×12，blockDim=min(40,N)，
每核grid-stride分配。B32为384task（每核9/10），B64为768（19/20）；不是384/768个硬件核。

| 阶段 / 源码锚点（约行） | 每task工作 | pos趋势与待验证点 |
| --- | --- | --- |
| metadata `ProcessSharedHeads` 341–400 | 5次ReadInt、校验及offset | 固定工作，非五条机器指令 |
| gate/beta 406–420 | history G整ring32读取/Exp；candidate G、beta各4×12读取 | pos0也执行；不能把所有G工作计作新增history |
| history product/suffix 421–453 | p次GetValue/乘，p次SetValue/乘；S→V同步 | O(p) scalar，另含固定同步 |
| history K load 430–442 | 0、1或2段搬运/Cast，逻辑字节256p | base0且p<=16为一段；跨尾最多两段 |
| candidate Q/K 454–480 | 各一次多行DataCopyPad/Cast，T次Q缩放 | 按head重复，固定T时与p无关 |
| history dots 513–524 / `DotRowsToUB`1394 | p>0时2T次；每次2个Mul API + ReduceK128 + barriers | 算术规模2T×p×128，repeat随p变化 |
| speculative dots 537–543 | T+(T−1)=2T−1次DotRows；T²个标量结果 | T4为7次、16结果，v20仍会Vector→Scalar；不是v8批量spec tree |
| checkpoint 565–582 | 4次8KiB BF16 Load/Cast，FP32 state 64KiB | 每task32KiB GM；与p无关 |
| history D 583–609 | 0/1/2段load/Cast，256p字节；清零(16−p)×128 FP32 | 清零成本随p下降，pos0仍清零全部D UB |
| checkpoint MatVec 621–633 | 2T次128×128 MatVec，再2T个scale | 大向量工作固定T时不随p变 |
| history correction 634–641 / `AccumulateHistory`1403 | p>0每token一次suffix scale + Q/K两次correction | 2T次；每次coeff清零16，Mul有效p，Broadcast16×128，Mul2048，4层tree，Add128 |
| speculative correction 662–674 | 两路各T(T−1)/2次Accumulate | T4两路合计12次，固定T时与p无关 |
| candidate/update/output 675–706 | delta/result、T次out/D写，每组三head首head写K | candidate数据不同会影响数值，保持输入不变 |
| gate/status 707–720 / `WriteGateBatch`1239 | gate一次或跨尾两次，status写，T<4补零out | 与候选跨尾相关；需固定base控制混杂 |

每次 history correction 的4层tree在tile128下包含7次显式Add API调用（前三层各2，末层1），
加上最终dest Add为8次；实际机器指令展开需编译核验。p从1到15时tree宽度仍为16，
但dot repeat、coeff Mul长度、suffix、加载量和D补零长度变化，不能称“history工作完全固定”。

GM逻辑请求量（非HBM实测）：checkpoint每task32768B，history K/D各256pB，history G固定128B。
K因三value heads重复读取，硬件L2命中/合并可能改变HBM流量；这些字节不能直接除HBM带宽推时延。
FP32 UB里history weighted Mul固定2048元素、tree固定四层。UB使用约188448B，仅为显式分配估计。

## 103 次调用解释与验证办法

`profile_sd_operator_compare.py:190–230` 对每case：

1. 一次native/replay正确性调用；
2. 默认warmup=2，两路径各2次；
3. PROFILE_BEGIN/END内各100次测量。

因此每个Replay symbol预期103次；12case就是1236，不是1224。
这解释源码预期，不自动证明历史CSV确实来自该脚本/参数。需回传命令、日志、CSV核验。
按单case/单进程、精确symbol、设备、stream时间顺序识别，前1+2与测量100分别标注。
若profiler有额外重放、warmup或过滤导致数量不是103，不允许强行取最后100；必须从标记/任务关联排查。
旧结果按耗时排序切成B32/B64两半的方法不可靠，作废该分组后重算。

## 当前runner的限制

旧compare以`make_case`+`clone_page_strided_case`产生页步长数据，但未传独立ring_slots；
它是tied-index代理性能样本，不是正式独立request ring性能验收。
`check_sd_independent_ring.py`可先确认双槽正确性，不能据此推定两个布局时延相同。
后续probe runner必须显式构造不同checkpoint/ring索引与独立分配，并记录shape/stride/offset与映射。
旧compare返回host_range_seconds，不含Graph Event计时；Graph另用run.py的结果，不能相互替代。

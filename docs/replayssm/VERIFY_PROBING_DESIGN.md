# SD verify 性能 probing：第一轮设计

状态：设计交付；Phase 0 可用现有脚本执行，A1/A2、局部采样和热点 API 新探针尚未实现。
源码锚点：`5b0a2b0`；任务来源：workspace 的 `ReplaySSM_Verify_Performance_Probing_TODO.md`。
本提交不修改 kernel、binding、服务默认值或精度，不把设计完成当作 TODO 中实机实验的 DoD 完成。
执行入口：[上机清单](checklists/LEVEL3_SD_VERIFY_PROBING.md)。

## 1. 需要回答的问题

H1：pos 从 0 到 1 的跳变，有多少来自开启固定 16 行 history correction？
H2：pos 从 1 到 15 的增量，有多少来自 history dot、suffix 标量循环和 K/D 搬运？
H3：同一局部片段在全 kernel 中是否因依赖和多任务排布增加等待？

均为待证伪假设。现有混合 workload 的 pipe 中位数、模拟 BAR 占比不足以分别回答它们。
主基线 v20：27B TP4 几何 / MTP3 / tile128 / 40 AIV，K/V=128，Hk=4、Hv=12；
BF16 checkpoint、FP16 D/K、FP32 G、FP32 UB；ring=32、有效 history 上限16、候选上限4。
不改变 compact/cursor，也不增加 fusion、dtype、模型适配工作。

## 2. 实验顺序与交付边界

| 阶段 | 本次状态 | 内容 | 退出条件 |
| --- | --- | --- | --- |
| P0 | 可运行 | 同轮基线、调用数审计、历史长度曲线、已有 metadata 微基准 | 数据可分组、漂移可评估，无设备/数值错误 |
| P1 | 设计完成，待实现 | A2、A1，必要时 A3/A4 | 至少两项有效消融，编译差异可核验 |
| P2 | 设计完成，待实现 | L1 单 task 和 U1/U2 热点 API | 至少一个局部样本和一个 API 有实机与编译证据 |
| P3 | 等待 P1/P2 结果 | 成本模型、一次正确优化候选 | 预测与实测对照，成功/失败均入库 |

P0 回传用于确定 P1 的优先 case。测试方这轮只执行清单 P0；未提供的 probe 不能以旧 variant 或 Python 替代来声称完成。
已有 metadata 微基准可校准五次小读取的总成本，但不代替 history 热点微基准。

## 3. 隔离实现方案（下一次代码交付）

新增 `experimental/replayssm_sd_ascendc/probing/experiments/`，独立 CMake + torch library
`replayssm_sd_probing`，专用 symbol `replayssm_probe_v20_<probe_id>`；不向生产 variant 表增加可选项。
复用 v20 数学及布局实现，但用独立编译模板参数选择 probe；生产实例默认参数为 off，
需对生产 v20 编译结果做回归，若变化则改用冻结的实验副本并记录原文件 SHA256。
运行器必须显式 `--profiling-only`；结果标记 `profiling_only=true`、`numerical_equivalence_required=false`。
探针和 probe-control 均固定 188448 B 的显式 UB 规划；实际 UB/栈/溢出以编译产物为准。
独立 sink 使用 GM，每个 task 32B 对齐，不能复用 status、有效输出或其他 task 的存储。
如增加 UB scratch，记录大小并给 probe-control 同样分配，禁止偷偷扩过192KiB。

生产 v20 是性能参照；含 sink 的 probe-control 是消融的直接对照。两者的差值单独报告。
不得在纯 Python 中做张量替代来冒充 AscendC 局部删除。

### A2：去掉 history D 加权归约（先做）

位置：`ProcessSharedHeads` 内两次 `AccumulateHistory`；只保留原 prevQ/prevK，不添加历史贡献。
保留 history K dot、D load/Cast/缺失行清零、scaledSuffix、checkpoint、candidate 和写回。
为防止上游被删除，将完整活动 dot、D、suffix 的校验归约输出到 sink；匹配 control 也执行相同 sink，
且放在原阶段之后，不用 sink 修改候选值。校验归约开销不得从总时间简单减掉。
同时可编译 minimal 版本，观察不保依赖时删除范围；只作为删除上界，不能归因为单独 correction。

核验：目标 Broadcast、2048元素 Mul、4层树和最终 Add 消失；dot、D DMA/Cast 及 candidate 工作保留。
pos0 是负对照：不执行此区域，差异超出同轮噪声时先查编译布局/资源，不宣称算法收益。

### A1：替换 history K dot（第二个）

每 candidate 的两次 history `DotRowsToUB` 换成确定的有限非零数据（例如从只读设备 probe 输入加载），
写足后续会读取的 pos 个系数；不能留下未初始化尾部或使用 NaN。
保留 Q/K candidate、history K DMA/Cast、history D correction 与所有原写回。
为保 K load/Cast，匹配 control 与 probe 均将完整已加载 K 数据归约到 sink；sink 的资源/同步影响单报。
核验被删的是 2T 次 history dot，speculative dot 和 checkpoint MatVec 必须仍在。
零常数替代可作为另一个 minimal 版本，但不能排除后续优化器常量传播，必须单独命名、单独分析。

### A3/A4：只在 A1/A2 留下明显残差时补

A3 保留 G load/Exp，把 historyAlpha 读取、historyProduct 与 suffix 标量链替换为设备输入的
precomputed product/suffix；保留完整 G sink。它额外引入 DMA，测量的是替代方案净变化，不是纯 scalar 成本。
A4 分别仅旁路 Q 或 K 的 `AccumulateHistory`，使用 A2 同样 sink control，检查非对称依赖。
它们都从 v20 独立派生，禁止把 A1+A2 相加解释成总 history 成本。
A5 暂缓：checkpoint/metadata 已有大量先验，先定位 pos 曲线。

## 4. 局部采样与 API 设计

L1：实验 launcher 显式选择一个有效 `(row,head,tile)`，blockDim=1，task_count=1；
不能把 B1 或 cores1 等同单 task（B1仍有12个task）。复用原始 v20 阶段顺序及 UB layout，
pos=0/1/7/15、T=4，重复数 R=1/16/64。每次迭代输入固定，候选区域与有效历史不重叠，
不推进 cursor；保留可观察输出和每轮必要依赖，检查编译器没有将循环提升/删除。
首轮/末轮 canary、状态和有限性检查在计时外；重复数增加不能无限增长输出或触发 overflow。
记录总 task 时延关于 R 的斜率和截距，不把斜率直接乘 B×12/40 预测服务。

L2（需要时）：单片段 history dot、suffix、weighted reduction，尺寸/对齐/屏障与 v20 一致，
区分输入装载+计算与热 UB 计算。先做 L1 再确定片段，避免预先遍历所有 API。
规模恢复：同一实验代码 task_count=1/12/96/384/768，cores=1/4/8/40（有效核数取 min）；
pos1/15、T4。先1/384/768与1/40，再按非线性结果补点；任务输出严格不重叠。

U1 优先：同一32KiB BF16 checkpoint，4×8KiB DMA+Cast，比较单段与原队列/同步组合；
再用8×4KiB作对照，FP32 UB目标及总字节一致。测的是完整 API 展开及依赖，不是纯 DMA带宽。
U2：history coefficient Broadcast+Mul+tree，保持16×128 FP32、128维tree输出；
活动rows=1/2/4/8/16，其余为有限零值。比较依赖链与独立buffer吞吐，使用受预算约束的两组buffer，
注明其与原kernel不同的UB布局；不把“依赖链vs独立”同时混入删除式消融。
根据 A1/A2 数据可将 U2 换为 `DotRowsToUB`（K128，rows同上），但必须先更新 manifest。

API 每条记录分 `documented / measured / inferred`；首次实测前 latency、throughput 填 null。
一条 API 不等于一条机器指令。设备真实 counters、编译指令和 simulator trace 分栏保存。
编译器输出 unavailable 时标记 `instruction_evidence_missing`，不进入具体指令根因结论。

## 5. 矩阵、测量与 DCE 门槛

最小发现集：B32/64、T4、pos0/1/7/15、base0；每变体 baseline-before → probe → baseline-after，
至少3轮，第二轮交换 probe 顺序。扩展：T4 pos0/1/2/4/7/8/12/15、T1 pos0/1/7/15。
ring 物理跨尾作为第二因素：固定pos15，base0/16/31；不能在pos变化时同时改变base。
固定所有原始tensor的seed和内容，改变pos只改变有效长度；相同case两侧 `.so` 和输入hash都要可追踪。

每次 profiler 进程只测一个 `(B,T,pos,base,variant)`；精确标记 warmup/check/measure。
按实际 kernel symbol、设备、进程/stream/task关联和顺序切分；禁止按时延中位数或timestamp gap猜batch。
保存原始CSV，不以聚合中位数替代逐调用数据。不要并发运行性能case。
Graph Event p50/p90/min与host wall time分开，不拿旧比较脚本的host_range_seconds冒充graph延迟。

同一case三轮分别计算配对差值，附基线前后漂移和调用内IQR；建议漂移>3%时标为受污染并复跑，
收益小于基线漂移/重复间波动时结论为 inconclusive。3%是本轮工程门槛，不是芯片规格。
每个删除式probe必须有源码diff、源文件/二进制hash、资源报告、反汇编或编译IR差异；
逐项列出“目标消失”“上游保留”“额外sink”“非目标变化”。代码大小变化本身不能证明DCE正确。

## 6. 安全与验收

所有 probe 检查元数据边界、独立ring索引、padding、T1/4、pos0/16、跨尾；先B2后大batch。
检查所有读buffer先定义、DMA/Vector/Scalar事件成对、sink对齐及唯一写者、canary未变、NPU无fault。
输出不与golden相等是预期，但NaN/Inf、status错误、越界、hang都失败。冻结输入，不能多轮累计错误状态。
强制超时和失败停止由runner负责；停本轮进程，不自行reset共享卡。

仅优化候选恢复golden+graph+独立ring+序列数值+完整性能/serve回归。
最终DoD仍是两项消融、一项局部样本、一项API的实测包，以及两类可复现成本来源；
若数据不支持定位，诚实记录未完成，不为了交付填一个确定根因。

## 7. 改动侵入性、工作量与外部依赖

| 内容 | 侵入性 | 粗估开发量（不含上机等待） | 风险/外部协助 |
| --- | --- | --- | --- |
| 本次方案与P0清单 | 文档，无生产改动 | 本次完成 | 测试方执行采样与回传 |
| 实验runner、manifest/解析、独立ring数据构造 | 仅实验设施 | 1–2天 | 核验本机CSV字段/标记，不能猜工具格式 |
| A1/A2与control/sink | 实验kernel；共享模板若改需回归生产实例 | 2–4天 | CCE编译、DCE与UB实测，sink可能掩盖收益 |
| L1 + U1/U2 | 实验kernel与binding | 2–3天 | CANN接口与重复循环优化行为 |
| 模型/归因与一次候选 | 先文档，候选另行审查 | 1–3天 | 无保证能定位/提速；可能需编译器团队 |

不要求测试方取得当前不支持的片上指令timeline；已有task/pipe数据先行。
编译产物/真实指令计数若不可得，可补 simulator，但结论标记模拟证据而非真机流水。

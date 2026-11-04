# P0报告查验（2026-11-03）

来源：用户回传run `replayssm-verify-probing-20261103T093039Z`，不是本地直接读取服务器原始CSV。
报告中身份、6项原版验证、103次审计和T4漂移足以推进P1/P2；原始数据仍待结果包支持独立复算。

保留的测量事实：B32/T4 p0→1为+50.3us，p1→15为+10.7us；B64分别+86.9us、+18.8us
（报告差分四舍五入写87.0、18.7，与列出的单点数有0.1us差异）。
主要表现为p>0路径启用的跳变，后续行数增长较缓；这支持优先A2，不证明A2就是全部根因。

需收紧表述：

- §4.4“H1证实”与§8的inferred矛盾，以inferred为准。H2来源同样尚未被消融分离。
- 28组合×3个进程是before/v13/after一轮夹测，并非三次独立夹测，不能据此给置信区间。
- ordered_single_case可以作为审计办法，timestamp gap仅作辅助，不能单独认定阶段身份。
- B32/T1/p0的4.59%污染结果不作因果定量依据；Graph表该case也有回退，全case更快的表述不成立。
- Graph与profiler输入不同，差值不能直接归为launch成本；MTE2时间不能称checkpoint独占耗时。
- v20组合相对v13的MTE3变化支持写回优化有效，尚不能证明chunk8k和gate_batch存在协同效应。

P1采用新独立ring fixture、固定完整history数据；必须重测实验baseline，不能直接减P0数值。
执行见`checklists/LEVEL3_SD_VERIFY_P1_P2.md`。实际编译资源、DCE及NPU语义留给本轮上机检验。

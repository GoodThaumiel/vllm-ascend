# ReplaySSM verify probing

P1/P2实验已实现，入口 `run.py`，独立CMake构建，不修改生产.so。
执行与结果口径以[本轮清单](../../../docs/replayssm/checklists/LEVEL3_SD_VERIFY_P1_P2.md)为准。
冻结源身份见snapshot.json；experiments/verify_snapshot.h包含显式实验改动，并非原文件逐字副本。

[方案](../../../docs/replayssm/VERIFY_PROBING_DESIGN.md) ·
[源码地图](../../../docs/replayssm_verify_probe_map.md) ·
[可执行 P0 清单](../../../docs/replayssm/checklists/LEVEL3_SD_VERIFY_PROBING.md)

P0设计已推进为独立实验代码；设备编译和验证待上机。
P0复用已提交脚本。不要把结果的存在当作完整TODO已完成。

后续实现runner时约定命令接口（**设计接口，现在不可运行**）：
`python3 scripts/manifest.py --phase 1 --output results/<run_id>/manifest.json`，
`python3 scripts/run.py --manifest ... --profiling-only`，
`python3 scripts/analyze.py --results ...`。不采用隐式自动试跑/自动启用生产variant。

结果包：

```text
<run_id>/
  manifest.json             # 实际命令数组、case_id、阶段、seed、shape/stride、布局、对照关系
  environment/              # commit/dirty patch、版本、设备/驱动/空闲状态、工具help与真实命令
  compiler/                 # 源码diff/hash、so hash、编译flags/日志、资源/指令差异或缺失原因
  <case_id>/
    stdout.log
    result.json             # 状态、是否profiling_only、数值/安全检查、计时口径
    profiler_raw/           # 原始CSV，不覆盖或只留汇总
    selected_calls.csv      # 原row/task/stream ID、阶段标签、单位、被排除理由
  observations.md           # 可直接从原始数据复算的事实
  interpretation.md         # 假设、置信度、替代解释、下一步反证
```

manifest每case必须标注 baseline/probe/control、B/T/pos/base、逻辑/物理ring长度、TP是几何模拟、
设备号、tile/core/requested/effective/tasks、cache dtype、重复数、warmup/check次数、
输入hash、buffer reset策略、是否计入packing/sink、source/binary hash、profiling_mode。
缺失字段用null+reason；禁止补猜的NPU计数。大CSV/二进制外存，git仅保存摘要及可校验hash。
Mac收到结果先核对identity、数量、DCE与漂移，再做归因；缺少指令证据时只报告源码级/总耗时关联。
经验追加到knowledge；新优化前先检索负结果。模拟器和实机结果必须分目录、标记来源。

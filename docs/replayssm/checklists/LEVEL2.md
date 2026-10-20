# 第二级上机验证清单

状态：**L2.1 Replay AR Triton候选实现待上机，尚未允许开始第三级。**

本级验证Replay AR的BF16 checkpoint、FP16 `d/k`与FP32 `g`历史缓存、flush/reset语义、Triton-Ascend实现和图兼容。
测试基线固定为[开发基线](../BASELINE.md)。每个实现提交会在本清单追加对应命令与通过条件；
用户确认整个第二级之前，`validation_state.json`的`next_level_allowed`保持false。

## 当前开发分段

1. checkpoint/ring的数据布局与CPU参考递推；
2. Replay AR Triton-Ascend算子，先正确性和NPUGraph，再测性能；
3. 影响性能的recurrent/flush路径评估AscendC实现；
4. TP4/TP8单算子矩阵与长序列flush/reset回归。

## 固定验收项

- [ ] BF16 checkpoint、FP16 `d/k`和FP32 `g`缓存的shape、stride、slot映射符合契约。
- [ ] 无历史、部分ring、ring边界、强制flush、自动flush和reset均与FP32参考对齐。
- [ ] TP4与TP8对应的local head数量均覆盖，包含多request slot和无效/padded row。
- [ ] eager多轮输入变化后输出与参考对齐。
- [ ] NPUGraph capture后修改静态输入，至少replay三次且结果与新输入参考对齐。
- [ ] 无越界写、设备异常或跨request状态污染；失败后能定位首个case。
- [ ] 记录内核编译、首次运行和稳态延迟；本级不设强制加速比。

## 回传材料

- 源码commit、`git status --short`、完整版本和模块/扩展加载路径；
- TP4与TP8结果JSON及首个失败case的完整traceback；
- 图capture/replay证据；
- 每个shape的最大绝对/相对误差以及flush/reset后的状态误差；
- 测试后进程、端口和显存残留检查。

## L2.1 Replay AR Triton上机命令

这次改动为**低侵入性**：只新增plugin内的算子、缓存契约和独立验证脚本，尚未patch vLLM模型、
metadata builder、cache allocator或scheduler。若本级算子不通过，不影响原生`vllm serve`路径。

开发方本地已运行34个ReplaySSM CPU测试、`compileall`、Ruff检查/格式检查和`git diff --check`，全部通过。
本机没有torch、Triton-Ascend和NPU，未导入或编译kernel；`bash format.sh ci`因本机缺少`pre-commit`未完成，
不记为通过。NPU编译、精度和图行为必须以下述上机结果为准。

拉取本清单对应commit后，确认工作区无未记录修改，再在CANN 9.2.0容器中执行：

```bash
git switch releases/v0.30.0
git pull --ff-only origin releases/v0.30.0
git rev-parse HEAD
git status --short

TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/replayssm/run_level2_ar.py \
    --tp 4 --capacity 16 --output /tmp/replayssm-level2-ar-tp4-r1

TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/replayssm/run_level2_ar.py \
    --tp 8 --capacity 16 --output /tmp/replayssm-level2-ar-tp8-r1
```

脚本按TP几何分别覆盖local key/value heads `(4,12)`和`(2,6)`，并在每张逻辑卡的独立子进程中运行。
每卡执行35个AR step，覆盖空历史、部分历史、自动flush、提前flush、padded slot以及三次图重放；
图重放的flush值为`[0,1,0]`，用于验证静态图内的运行时分支和输入变化。

通过条件：

- TP4四张卡、TP8八张卡的`deviceN.json`均为`PASSED`，summary为`AUTOMATED_CHECKS_PASSED`；
- output和checkpoint相对存储精度参考满足`atol=3e-2, rtol=3e-2`，`g`满足`5e-4`；
- 三次图重放均使用修改后的输入，padded slot输出严格为零；
- 没有编译错误、UB overflow、MTE异常、AIVEC异常、设备上下文错误或进程残留。

请先返回**第一个失败设备**的`deviceN.json`和`deviceN.log`；两组都通过时返回两个`summary.json`。
本提交只验收AR算子候选，不运行`vllm serve`，也不代表第二级整体完成。

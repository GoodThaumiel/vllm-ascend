# 第二级上机验证清单

状态：**开发中，尚未允许开始第三级。**

本级验证Replay AR的FP32 checkpoint、BF16 ring cache、flush/reset语义、Triton-Ascend实现和图兼容。
测试基线固定为[开发基线](../BASELINE.md)。每个实现提交会在本清单追加对应命令与通过条件；
用户确认整个第二级之前，`validation_state.json`的`next_level_allowed`保持false。

## 当前开发分段

1. checkpoint/ring的数据布局与CPU参考递推；
2. Replay AR Triton-Ascend算子，先正确性和NPUGraph，再测性能；
3. 影响性能的recurrent/flush路径评估AscendC实现；
4. TP4/TP8单算子矩阵与长序列flush/reset回归。

## 固定验收项

- [ ] FP32 checkpoint与BF16 `d/k` ring的shape、stride、slot映射符合契约。
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

具体执行命令将在Replay AR算子提交时补充。本次文档与版本门槛提交不新增NPU代码，
无需单独重跑模型矩阵；继续使用已验收的一级结果作为环境基线。

# 第二级上机验证清单

状态：**L2.1首版发生可复现AI Core超时；移除`tl.cumsum`的第二版及并行诊断探针待上机，尚未允许开始第三级。**

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

## L2.1首轮结果

commit `26dd021293bf52d2743329ddb1c6f3c2afe48688`在CANN 9.2.0、triton-ascend 3.2.2上：

- 原始源码因三操作数链式`and`触发编译器`visit_BoolOp`错误；拆成二操作数节点后编译成功；
- TP4 device 0在step 0第一次launch后同步时稳定触发`507014` AI Core timeout，复跑与`MAX_JOBS=128`均不改变结果；
- 故障发生在任何精度比较或NPUGraph capture之前，TP8未运行；
- 本级保持`next_level_allowed=false`。

第二版移除正式内核中的`tl.cumsum`，采用固定上界串行prefix，并新增可并行运行的隔离探针。
详细依据见[Level 2 AR设计](../LEVEL2_AR_DESIGN.md#l21首轮故障与第二版设计)。

## L2.1第二版：先运行并行诊断探针

在8卡均空闲时，以下命令会同时启动六个独立子进程，分别绑定逻辑卡0到5。每个探针使用独立
`TRITON_CACHE_DIR`，一个设备发生trap不会阻止其他设备生成报告：

```bash
git switch releases/v0.30.0
git pull --ff-only origin releases/v0.30.0
git rev-parse HEAD
git status --short

TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/replayssm/run_level2_triton_probes.py \
    --devices 0,1,2,3,4,5 \
    --output /tmp/replayssm-level2-triton-probes-r1
```

若当前只开放四张卡，分两轮运行；同一轮内设备ID必须互不相同：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/replayssm/run_level2_triton_probes.py \
    --probes cumsum,serial_prefix,dot,masked_zero_dot \
    --devices 0,1,2,3 \
    --output /tmp/replayssm-level2-triton-probes-r1a

TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/replayssm/run_level2_triton_probes.py \
    --probes runtime_branch,masked_2d_io \
    --devices 0,1 \
    --output /tmp/replayssm-level2-triton-probes-r1b
```

每个构造会连续launch多次，以覆盖已知的重复launch trap。探针通过口径：`serial_prefix`、两个dot、运行时分支和
masked二维I/O必须通过。`cumsum`用于确认后端特征；
它失败或timeout是有效诊断结果，不阻止测试已移除该构造的第二版整核。返回`summary.json`以及所有失败探针的
JSON/log；探针造成设备错误后，确认所有卡恢复空闲和健康，再运行整核。

### commit 16107d8探针结果

CANN 9.2.0 / triton-ascend 3.2.2的六卡并行结果如下：

- `cumsum`、`serial_prefix`、普通`dot`、`runtime_branch`和`masked_2d_io`均通过多次launch；
- `masked_zero_dot`失败，4096个元素中608个不匹配，最大绝对误差`21.66828155517578`；
- 因此当前最强故障证据是：归约维动态mask全false时，`tl.load(..., other=0)`的结果直接进入
  `tl.dot`没有保持零值语义。正式kernel的step 0恰好命中该构造；
- Cannbot约束库也明确记录masked load的`other`在相关Ascend路径不可靠，建议在参与后续计算前用
  `tl.where(mask, x, 0.0)`显式清零；本轮探针同时验证该建议及其他保底写法；
- 本轮源码状态包含`M scripts/replayssm/run_level2_triton_probes.py`，所以结果只作为诊断证据，
  不作为干净源码验收。需保留并回传该文件的`git diff`，后续新探针使用独立脚本，不覆盖此修改。

暂不运行第二版整核。先在独立逻辑卡上并行比较六种规避路径：load后`where`、load后乘mask、
masked load后再次select、只mask左/右操作数，以及两kernel物化后再dot。每种路径覆盖有效长度
`0/7/16`并重复launch四次：

```bash
git switch releases/v0.30.0
git pull --ff-only origin releases/v0.30.0
git rev-parse HEAD
git status --short

TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/replayssm/run_level2_masked_dot_probes.py \
    --devices 0,1,2,3,4,5 \
    --output /tmp/replayssm-level2-masked-dot-probes-r1
```

返回`summary.json`和所有失败探针的JSON/log。选择规则依次为：通过全部shape的单kernel路径、改动最小、
UB/指令开销最低；只有单kernel路径全部失败时，才采用`materialize_then_dot`两kernel保底。选定后再修改正式kernel，
并重新执行TP4/TP8整核清单。

## L2.1第二版：Replay AR Triton整核

这次改动为**低侵入性**：只新增plugin内的算子、缓存契约和独立验证脚本，尚未patch vLLM模型、
metadata builder、cache allocator或scheduler。若本级算子不通过，不影响原生`vllm serve`路径。

开发方本地已运行36个ReplaySSM CPU测试、`compileall`、Ruff检查/格式检查和`git diff --check`，全部通过。
本机没有torch、Triton-Ascend和NPU，未导入或编译kernel；`bash format.sh ci`因本机缺少`pre-commit`未完成，
不记为通过。NPU编译、精度和图行为必须以下述上机结果为准。

完成探针并确认设备健康后，在同一CANN 9.2.0容器中执行：

```bash
git switch releases/v0.30.0
git pull --ff-only origin releases/v0.30.0
git rev-parse HEAD
git status --short

TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/replayssm/run_level2_ar.py \
    --tp 4 --capacity 16 --device-timeout 1200 \
    --output /tmp/replayssm-level2-ar-tp4-r2

TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/replayssm/run_level2_ar.py \
    --tp 8 --capacity 16 --device-timeout 1200 \
    --output /tmp/replayssm-level2-ar-tp8-r2
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

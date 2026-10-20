# ReplaySSM 开发基线与版本身份

本项目从第二级开始固定使用以下服务器环境。版本检查同时记录包版本、源码commit和实际模块路径，
避免editable安装的元数据落后于checkout时误判运行源码。

| 组件 | 开发基线 | 身份规则 |
|---|---|---|
| 硬件 | 8 x Ascend 910B4 32GB，TP4/TP8 | 记录 `npu-smi` 与可见逻辑卡；精确型号由服务器资产确认 |
| CANN | 9.2.0 | 必须从当前 `ASCEND_HOME_PATH` 的版本文件核对；后续不使用9.1门槛覆盖 |
| vLLM | 0.30.0，当前构建为 `0.30.0+empty` | public release必须是0.30.0，允许local build后缀 |
| vLLM Ascend | `releases/v0.30.0` | 精确git commit和模块路径为开发态主身份；允许0.30.0的pre/dev/local构建后缀 |
| torch | 2.10.0，当前构建为 `2.10.0+cpu` | public release必须是2.10.0，允许local build后缀 |
| torch-npu | 2.10.0.post4 | 精确匹配 |
| triton-ascend | 3.2.2 | 精确匹配 |
| FLA | `flash-linear-attention-npu-a2 26.9.1+deva4a7958` | 记录distribution、模块路径及三个依赖API签名 |
| 模型 | Qwen3.6-27B结构；一级以同构Qwen3.5-27B权重验证 | 最终模型级验收仍需Qwen3.6-27B权重 |

## 开发态版本规则

`vllm`、`vllm-ascend`和`torch`允许PEP 440的构建后缀，但不能跨public release：

- local build：`0.30.0+empty`、`2.10.0+cpu`；
- prerelease/dev source build：`0.30.0rc2.dev171+g62cc09b17`；
- 拒绝其他release和post release，例如`0.30.0.post1`；
- `torch-npu=2.10.0.post4`、`triton-ascend=3.2.2`与CANN 9.2.0保持严格匹配。

editable安装时，包元数据中的git后缀可能仍指向上次编译commit。因此每份上机报告必须同时记录：

1. `git rev-parse HEAD`和`git status --short`；
2. 完整package version；
3. `vllm_ascend.__file__`、FLA模块路径及编译扩展来源；
4. CANN版本文件绝对路径和内容；
5. 设备、TP、模型config和缓存dtype。

Python改动由editable checkout直接生效。C++、AscendC、CMake或生成扩展改动必须重新构建安装，
不能只用模块路径证明二进制已经更新。商发包形成后恢复正式版本和构建产物的严格门槛。

## 固定推理口径

- 入口为`vllm serve`和HTTP请求；离线`LLM.generate`只保留为历史诊断。
- 原生激活、Mamba conv cache和SSM state均为BF16。
- Replay checkpoint为FP32；本期不实现论文的低精度checkpoint档位。
- MTP优先，DFlash在MTP验收完成后开始。
- 图模式必须给出实际capture和replay日志，配置字段本身不算证据。

算子探针按一张逻辑卡一个子进程运行。单进程先后在多个NPU上创建NPUGraph会混合设备图上下文，
曾表现为旧输出、MTE invalid GM address和`model not in current context`。生产TP本身也是每卡worker进程，
因此跨卡聚合只在父进程做，单卡进程内完成eager、capture和replay。

## 一级验收依据

2026-10-19/20的`vllm serve + HTTP`轮次基于commit
`ee9401dc0b6e3f51e14180a8b698da6818ea9f75`，CANN 9.2.0无门槛覆盖。
TP4和TP8的environment、operators、AR eager/graph、MTP eager/graph均为
`AUTOMATED_CHECKS_PASSED`；两种图路径均有capture/replay证据，MTP draft计数增量均为78。
该结果关闭第一级并允许开始第二级，不构成Replay功能或性能结论。

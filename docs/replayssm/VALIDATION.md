# ReplaySSM：五级人工验收流程

当前交付：**第一级已验收，第二级开发中。**

基线固定为 vLLM 0.30.0 + 本仓 `releases/v0.30.0`、CANN **9.2.0**、910B4、TP4/TP8。
原生AR/MTP推理显式使用 `dtype=bfloat16`、`mamba_cache_dtype=bfloat16` 和
`mamba_ssm_cache_dtype=bfloat16`，对应BF16激活、卷积缓存与SSM state。
本仓分支起点为 `ff97be249d179636c1b538b85115541d2b63ce43`。源码快照、pip版本和实际加载路径都需记录，不能只看版本号。
完整开发环境与版本后缀规则见[开发基线](BASELINE.md)。

## 五级范围

| 级别 | 交付与用户验证内容 | 推进门槛 |
|---|---|---|
| 1 | 环境与依赖、现有Triton算子eager/图重放、原生AR/MTP的TP4/TP8基线 | 用户返回两份结果包，修复本级问题并明确确认通过 |
| 2 | checkpoint/ring契约、Replay AR Triton/AscendC算子、flush/reset和参考误差 | 用户确认算子正确性及图兼容；不据CPU结果代签 |
| 3 | Replay AR插件/引擎集成、TP4/TP8、图重放与请求生命周期 | 用户确认AR模型结果、调度边界和稳定性 |
| 4 | Replay SD/MTP的output-only verify、accepted-prefix提交、卷积/KV一致性、图模式 | 用户确认MTP SD结果、接受率和状态边界 |
| 5 | AR与MTP SD性能、AscendC热点优化、长稳、回退与商发报告 | 用户完成最终验收；按实际结果报告收益与限制 |

**每级只实现当前级。** 用户测试 → 返回日志 → 修复同一级 → 用户明确确认 → 开始下一级。
脚本只汇报自动检查，不自动把状态推进到下一级，不代用户生成验收确认。
发现前一级回归时先修复，不绕过门槛。DFlash在MTP验证成功后另开同样的验证循环，当前不实现。

## 每次提交的验证交付约定

开发方先执行本地可运行且与变更相关的测试，记录命令、结果和未覆盖项。
每次提交必须在提交说明中指向仓库内的上机验证清单，给出适用级别、服务器命令、通过条件和回传材料。
提交代码不代表上机验收通过；不将未运行、dry-run或CPU模拟记为NPU通过。
纯文档修改需标明无需新增硬件测试，当前级别尚未完成的硬件清单仍保留。
同一级修复先重跑失败项与受影响回归项；跨级推进仍需用户明确确认。

当前清单：[L2.3 Replay AR 模型接入验证](checklists/LEVEL2_MODEL_AR.md)。一级验收以commit `ee9401dc0`的服务轮次为准；旧版离线报告仅保留为历史证据。
第二级AR算子的数据布局、数学、侵入性和不确定性见[Level 2 AR设计](LEVEL2_AR_DESIGN.md)。
L2.1首版commit `26dd021293bf52d2743329ddb1c6f3c2afe48688`在拆分链式布尔表达式后能够编译，
但step 0稳定触发`507014` AI Core timeout。commit `16107d811`的隔离探针证明`tl.cumsum`本身通过，
而串行prefix在融合kernel中触发`hivm.hir.copy`编译错误；同时发现`tl.dot`左操作数的归约维masked load
不保持零值语义。commit `4c003b4ac`的六路探针确认load后显式`tl.where`清零可规避该方向性缺陷。
第三版恢复`tl.cumsum`并采用该单kernel规避后仍在TP4 device 0触发`507014`。项目随后转向
AscendC：独立算子、framework F1-F3、TP4/TP8 B=1/8/16及48层图重放已经通过用户上机验证。
当前实现已接入Qwen GDN的五块cache、host cursor和non-spec decode dispatch，等待L2.3真实模型
`vllm serve`验证；spec/MTP会被显式拒绝，直到accepted-token commit/rollback接入。

## 实现档位

- PyTorch eager只作参考、定位与显式fallback。
- 新增通用Replay计算采用Triton-Ascend，有基础则增量开发；以图模式为目标。
- recurrent/verify/flush等热点优先考虑AscendC，在匹配FLA源码上增量开发；不以语言保证性能。
- 复用现有AscendC卷积、prefill及Triton控制kernel，不为了档位纯度重写所有算子。
- Replay checkpoint沿用BF16 SSM state存储，重建和递推在算子内部升FP32计算；本期不做低于BF16的checkpoint档位。
  原生推理的BF16 SSM/卷积缓存是用户部署基线；BF16模型的Replay `d/k`历史缓存按原实现使用FP16，`g`使用FP32。

## 第一级做什么

`scripts/replayssm/run_level1.py` 依次运行6个独立case。四个模型case均启动独立的 **vllm serve**，
通过HTTP验证后关闭服务，再运行下一项；环境与单算子探针继续独立执行。
服务使用当前Python环境调用 `python -m vllm.entrypoints.cli.main serve`，与 `vllm serve` 的console entrypoint一致，
不使用离线 `LLM.generate` 进行模型验收。

1. `environment`：核查包版本、可见NPU数、CANN版本文件、FLA API与实际模块路径。
2. `operators`：在目标TP对应的每张逻辑卡上检查现有GDN gating和L2Norm；eager与torch参考比较，再capture/replay。capture之后修改静态输入，排除旧输出造成的伪通过。
3. `ar_eager`：原生AR，BF16 SSM与卷积缓存，无Replay。
4. `ar_graph`：原生AR，请求 `FULL_DECODE_ONLY`。
5. `mtp_eager`：原生MTP，默认S=3，检查请求前后实际draft计数增量大于0。
6. `mtp_graph`：target请求 `FULL_DECODE_ONLY`；drafter保持eager，和本仓Qwen教程一致。

每个服务先通过 `/health` 和带本次唯一模型别名的 `/v1/models` 核对，再向 `/v1/completions`
发送两条合成prompt的预热与正式请求，核对返回token IDs、batch完整性及usage。
MTP从 `/metrics` 的 `vllm:spec_decode_num_drafts_total` 检查请求后的正增量，最多等待30秒供指标刷新。
这一级使用非流式completion接口；chat模板、流式接口及生产负载另行验证。

第一级不是Replay实现，也不是精度或性能结论。短文本只证明原生路径能运行；性能数字仅供诊断。
现有算子图探针不等于完整模型图证明；模型case记录 `REQUESTED_NOT_INSTRUMENTED`，需结合日志确认capture/replay实际发生。
不因为配置了graph就自动声明全模型已在图上运行。MTP drafter完整图支持留在后续验证。

## 服务器准备

使用已安装vLLM/Ascend的Linux NPU环境，source实际CANN 9.2.0安装目录的 `set_env.sh`。
不要使用开发方macOS的 `.venv`，也不要在服务器上直接安装整个requirements覆盖现有环境。
脚本将此checkout放在子进程模块搜索路径的前面；请使用已按现有部署流程构建/安装此源码及NPU扩展的环境。
只有clone而没有对应编译产物时，可能在原生自定义op导入阶段失败，此时返回环境日志，不自动改用其他wheel冒充该源码。
第一级不改驱动、固件、CANN、torch或FLA，不下载模型、不修改site-packages、不调用外部推理服务。

期望包版本为vLLM/Ascend 0.30.0（允许rc/local build）、torch 2.10.0、torch_npu 2.10.0.post4、triton-ascend 3.2.2。
任何版本差异先如实反馈，不能为了让脚本绿灯而自行放宽。FLA版本与函数签名记录在结果中，需核对实际wheel/ABI。
CANN版本从 `ASCEND_HOME_PATH` 下compiler/version.info等文件读取，查不到视为未验证，不能人工传个版本字符串伪造通过。
设备名称仅是观测值，不据npu-smi的通用Chip Name推定编译SoC；910B4精确型号由用户确认。

## 拉取和运行

在你已授权的fork中拉取普通快进提交，不要reset覆盖服务器本地修改：

```bash
git switch releases/v0.30.0
git pull --ff-only origin releases/v0.30.0
git rev-parse HEAD
```

先检查命令计划。将模型路径替换成服务器**已有**的Qwen3.6-27B目录：

```bash
python scripts/replayssm/run_level1.py \
    --model /path/to/Qwen3.6-27B \
    --tp 4 --runner v1 \
    --output /tmp/replayssm-level1-serve-bf16-tp4 \
    --dry-run
```

dry-run会展示worker命令和实际serve命令模板，不检查硬件、不创建通过报告。确认路径后运行：

```bash
python scripts/replayssm/run_level1.py \
    --model /path/to/Qwen3.6-27B \
    --tp 4 --runner v1 \
    --output /tmp/replayssm-level1-serve-bf16-tp4
```

TP4完成后，在同样环境中运行TP8：

```bash
python scripts/replayssm/run_level1.py \
    --model /path/to/Qwen3.6-27B \
    --tp 8 --runner v1 \
    --output /tmp/replayssm-level1-serve-bf16-tp8
```

仅需使用部分卡时，在启动前按已有部署规范设置可见设备；`--tp`使用进程看到的逻辑卡。
AR与MTP四个模型case均显式使用BF16 SSM与卷积缓存，结果JSON的 `engine_kwargs` 记录这两个参数。
默认显存使用比例0.85、上下文4096、batch最多2、生成64 tokens。可用 `--memory-utilization` 降低比例。
OOM、通信或算子错误应返回失败包，不要不记录地反复修改参数。
每case默认超时1800秒，可通过 `--timeout` 调整。卡数不足会在环境步骤停止。
服务仅绑定 `127.0.0.1`，默认端口18080，端口被占用时拒绝运行；可通过 `--port` 指定空闲端口。
默认服务就绪超时1200秒（`--startup-timeout`），每个HTTP请求超时180秒（`--request-timeout`），
外层 `--timeout` 限制整个case，最先触发的超时生效。图编译耗时长时需相应调整并记录。
完成、失败或外层超时后清理本次worker进程组中的服务/TP子进程；请核对无残留再开始下一轮。
脚本不会连接已有部署服务，不复用未知服务的健康检查或指标。
结果目录和同名tar包必须不存在；重试用新目录，保留失败证据。

当前先验证v1路径。若实际商发明确使用v2，可另开结果目录用 `--runner v2` 跑同一矩阵；不将v1通过当作v2通过。

## 反馈与验收

脚本在每case后更新 `summary.json`，失败即停止，退出码非0。输出目录含：

- `manifest.json`：源码commit、运行参数和模型路径。
- `environment.json`：版本、CANN证据、FLA签名和实际加载路径。
- `operators.json`：各卡现有算子eager/图重放结果。
- 各模型case的JSON：包含 `validation_mode=vllm_serve_http`、实际serve命令、engine参数、HTTP耗时和draft增量。
- `*.server.log` 服务完整日志、`*.request.json` 请求、`*.warmup.json` 与 `*.generation.json` HTTP响应。
- `*.metrics-before.log`、`*.metrics-after.log` 保存原始Prometheus指标；未执行的case不会伪造记录。
- 完成或失败后生成的同名 `.tar.gz`，可直接作为反馈包。

返回TP4/TP8的结果包，说明精确910B4型号、单卡HBM、是否实际看到模型capture/replay，以及任何手动环境调整。
日志可能含服务器路径、设备信息与公开合成prompt；发出前可审阅，脚本不会自动上传任何结果。

`AUTOMATED_CHECKS_PASSED`只表示6个自动检查完成；同时仍有`user_confirmation=PENDING`、`next_level_allowed=false`。
缺项、异常退出、只有dry-run、单卡探针通过或模型输出可读，均不代表第一级验收通过。
本级通过后将你的确认与对应commit记录在本文件；当前不预填确认。

## 开发方本地检查

无NPU环境只运行以下CPU检查，绝不把mock或macOS结果记作NPU通过：

```bash
.venv/bin/python -m unittest discover -s tests/ut/replayssm -v
.venv/bin/python -m compileall -q scripts/replayssm tests/ut/replayssm
.venv/bin/ruff check scripts/replayssm tests/ut/replayssm
git diff --check
```

本工作使用AI辅助。基础源码未改动；当前只新增用户侧验证工具、CPU门槛测试与文档。

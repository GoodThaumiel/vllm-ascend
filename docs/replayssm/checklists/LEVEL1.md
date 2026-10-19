# 第一级上机验证清单

状态：**待用户上机验证，第二级未开始。**
适用于更新为BF16缓存基线后的第一级验证工具。源码commit以服务器实际 `git rev-parse HEAD` 和结果包manifest为准。
本次修改原生模型case的缓存配置，需验证新的BF16基线；当前尚未收到首次硬件测试结果。
本级不修改生产推理路径，不启用ReplaySSM。

## 本次基线精度更新

原生AR/MTP的激活、SSM state及卷积缓存均采用BF16：
`dtype=bfloat16`、`mamba_ssm_cache_dtype=bfloat16`、`mamba_cache_dtype=bfloat16`。
这与后续Replay checkpoint的存储精度分别记录，不启用论文低精度Replay checkpoint功能。
旧commit的FP32 SSM结果不能替代本次BF16基线结果。

本次提交上机需以最新commit运行下方TP4/TP8矩阵；逐一核对四个模型case的JSON中
`engine_kwargs.mamba_ssm_cache_dtype` 与 `engine_kwargs.mamba_cache_dtype` 均为 `bfloat16`，
并保留加载日志以核对实际缓存配置。若实现拒绝BF16或日志显示强制改写dtype，反馈为本级待修复问题。
现有gating/L2Norm探针的FP32参考计算不变，它不表示持久SSM缓存为FP32。

## 本地检查

开发方在macOS、Python 3.12环境运行以下命令，NPU检查交给用户服务器：

```bash
.venv/bin/python -m unittest discover -s tests/ut/replayssm -v
.venv/bin/python -m compileall -q scripts/replayssm tests/ut/replayssm
.venv/bin/ruff check scripts/replayssm tests/ut/replayssm
.venv/bin/ruff format --check scripts/replayssm tests/ut/replayssm
git diff --check
```

CPU测试覆盖报告完整性、版本门槛、失败/超时处理、dry-run与人工确认门槛；不覆盖NPU算子、通信、模型精度或性能。

本次本地执行结果：13个CPU测试通过；compileall、Ruff检查与格式检查、`git diff --check`通过。
仓库全量入口 `bash format.sh ci` 已尝试，但本机缺少 `pre-commit`，未完成全量hook检查，不记为通过。
所有NPU检查尚未运行。

## 上机前核对

- [ ] 拉取 `releases/v0.30.0`，记录commit与本地修改；不覆盖服务器已有改动。
- [ ] 使用服务器已有安装流程构建/安装当前源码及NPU扩展，确认运行环境加载的是此checkout。
- [ ] 确认硬件为910B4，记录精确型号、单卡HBM、驱动/固件、可见设备及通信配置。
- [ ] source实际CANN **9.2.0** 的环境脚本，记录torch/torch_npu、Triton-Ascend、FLA版本。
- [ ] 准备本地Qwen3.6-27B权重及其原生MTP配置；记录权重版本和config，不下载或改写模型。
- [ ] TP4与TP8顺序执行，分别确保4/8张可用卡；输出目录和同名tar包必须尚不存在。

具体软件版本与安装限制见[验证流程](../VALIDATION.md)。依赖不匹配时返回证据，先修复第一级环境。

## 执行命令

在已配置好的服务器Python环境、仓库根目录执行。模型路径替换成实际目录；每次重试使用新结果目录。

```bash
git switch releases/v0.30.0
git pull --ff-only origin releases/v0.30.0
git rev-parse HEAD
git status --short

python scripts/replayssm/run_level1.py \
    --model /path/to/Qwen3.6-27B \
    --tp 4 --runner v1 \
    --output /tmp/replayssm-level1-bf16-tp4-r1 --dry-run

python scripts/replayssm/run_level1.py \
    --model /path/to/Qwen3.6-27B \
    --tp 4 --runner v1 \
    --output /tmp/replayssm-level1-bf16-tp4-r1

python scripts/replayssm/run_level1.py \
    --model /path/to/Qwen3.6-27B \
    --tp 8 --runner v1 \
    --output /tmp/replayssm-level1-bf16-tp8-r1
```

脚本失败即停止并保留结果包。先回传失败证据，我们修复本级后再重试；不要把未执行项勾为通过。
显存不足或通信失败也按失败项反馈，记录所有手动调整。默认MTP speculative tokens=3；改为1需单独记录。
若商发采用v2 runner，需另开目录以 `--runner v2` 运行相同矩阵，v1结果不替代v2验收。

## 检查矩阵和通过条件

每个勾选项都需对应JSON与日志证据。图模式模型项除自动状态外，还需检查实际capture/replay证据。

| 检查项 | TP4 | TP8 | 通过条件 |
|---|---|---|---|
| environment | [ ] | [ ] | 版本、CANN证据、FLA API/加载路径匹配，可见卡数满足TP |
| operators | [ ] | [ ] | 每张逻辑卡现有gating/L2Norm的eager与参考对齐，修改输入后的图重放对齐 |
| ar_eager | [ ] | [ ] | BF16双缓存配置核对，原生AR完成warmup和生成，无异常，输出非空 |
| ar_graph | [ ] | [ ] | BF16双缓存配置核对，原生AR生成通过，模型图capture/replay有实际证据 |
| mtp_eager | [ ] | [ ] | BF16双缓存配置核对，原生MTP生成通过，实际draft计数大于0 |
| mtp_graph | [ ] | [ ] | BF16双缓存配置核对，MTP生成及draft计数通过，target图执行有证据；drafter为eager |

算子自动容差为 `atol=5e-3, rtol=1e-2`，完整模型质量与Replay精度不在本级结论内。
模型图状态目前记录为 `REQUESTED_NOT_INSTRUMENTED`，仅此字段或capture配置不能作为模型图通过证据。
若无法确认实际图执行，标记“待确认”并返回日志，我们在本级补充诊断，不据此进入第二级。
六项齐全的自动汇总应为 `AUTOMATED_CHECKS_PASSED`；人工确认仍为 `PENDING`，不自动解锁下一级。

## 回传材料与确认

- [ ] TP4结果包：`/tmp/replayssm-level1-bf16-tp4-r1.tar.gz`。
- [ ] TP8结果包：`/tmp/replayssm-level1-bf16-tp8-r1.tar.gz`。
- [ ] 说明模型图实际执行证据所在文件/日志行，或明确标注尚无证据。
- [ ] 如失败，提供第一个失败case、完整traceback、退出码以及重试前后的环境差异。

可按以下模板反馈；日志发出前可审阅，脚本不会自动上传：

```text
源码commit / 是否有本地修改：
硬件精确型号 / 单卡HBM / TP：
CANN / torch / torch_npu / Triton-Ascend / FLA：
模型权重版本 / runner / 参数调整：
四个模型case的SSM state与Mamba cache dtype核对结果：
TP4与TP8结果包：
第一个失败case（若有）：
AR及MTP target图capture/replay证据（或待确认）：
结论：需修复第一级 / 确认第一级通过，可以开始第二级
```

用户确认须与源码commit和结果包关联记录；当前不预填通过结论。

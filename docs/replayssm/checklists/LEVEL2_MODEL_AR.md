# L2.3：ReplaySSM AR dispatch 与缓存生命周期上机验证

## 本轮边界

本轮把已经通过 F1–F3 的 `_C_ascend.replayssm_ar_step_` 接入 Qwen GDN target decode。
启动参数必须保持以下基线：模型、卷积 cache 和 SSM checkpoint 均为 BF16；Replay d/k ring
为 FP16，g ring 为 FP32；CANN 9.2.0；vLLM `0.30.0+empty`；本仓 vLLM Ascend 0.30。

首轮只测非 speculative AR。代码会主动拒绝 Replay AR 与 MTP/DFlash 同时启用；这是为了在
ReplaySSM SD 的 accepted-token commit/rollback 接入前避免破坏状态。`mamba_cache_mode=none`
是主验收路径，`align` 留作后续 prefix-cache 子轮；`all` 当前会被拒绝。

本轮涉及三处侵入式插件修改：

- **高侵入：cache 布局。** Qwen GDN 每层由 conv/checkpoint 两块状态扩展为
  conv/checkpoint/d/k/g 五块状态；只在 `--use-replayssm` 时生效。
- **中侵入：metadata 生命周期。** Ascend GDN builder 根据 vLLM 0.30 已有的
  `replayssm_decode_base_cpu` 生成固定地址 `write_pos/flush` device tensor，覆盖窗口 flush、
  请求恢复、slot 复用和图 padding。
- **中侵入：decode dispatch。** pure decode 和 mixed decode 的 decode slice 调用 AscendC Replay；
  causal conv、gating、L2 norm、prefill 和输出投影继续走原有实现。

插件还对 Qwen3Next/Qwen3.5 架构的 `supports_replayssm` 和静态 cache shape/dtype 做了限定
monkey patch。这是为了不修改 `vllm 0.30.0+empty` 源码，属于**侵入式兼容补丁**；后续应向
upstream vLLM 提交通用 GDN Replay cache 接口后删除。

## A0：重建和 CPU/Meta 回归

拉取目标提交后重新构建扩展；不能只让 editable Python 生效：

```bash
git switch releases/v0.30.0
git pull --ff-only origin releases/v0.30.0
git rev-parse HEAD

COMPILE_CUSTOM_KERNELS=1 SOC_VERSION=ascend910b1 \
python3 -m pip install --no-build-isolation --no-deps -v -e . \
  2>&1 | tee /tmp/replayssm-l23-build.log

python3 -m unittest tests.ut.replayssm.test_model_ar_integration -v
python3 -m unittest tests.ut.replayssm.test_level2_contract tests.ut.replayssm.test_call_budget -v
```

同时补齐 L2.2 遗留的 Meta 证据。服务器会自动发现本轮安装的 `vllm_ascend_C`：

```bash
python3 tests/ut/replayssm/test_framework_meta.py
```

只有本地 CPU CI 使用独立 Meta `.so` 时才显式传入其真实路径；不存在的路径现在会直接失败，
不会再以 skip 退出 0。

- [ ] 安装版本指向本轮 commit，`vllm_ascend_C` 与 kernel `.so` 为本轮重编译产物。
- [ ] cache/cursor 单测通过；Meta/FakeTensor 只推导通过。
- [ ] 不出现 Qwen 模型类导入、`supports_replayssm`、五块 cache 数量或 dtype/shape 错误。

## A1：TP4 Replay AR `vllm serve` 首次启动

使用服务器已有模型目录；模型的 architecture 应解析为当前 0.30 的 Qwen3Next/Qwen3.5 GDN
实现。先用独占端口启动 Replay 服务：

```bash
export REPLAYSSM_MODEL=/path/to/Qwen3.6-27B

vllm serve "$REPLAYSSM_MODEL" \
  --host 127.0.0.1 --port 18083 --served-model-name replayssm-ar-l23 \
  --tensor-parallel-size 4 --pipeline-parallel-size 1 \
  --dtype bfloat16 --mamba-cache-dtype bfloat16 \
  --mamba-ssm-cache-dtype bfloat16 --mamba-cache-mode none \
  --use-replayssm --replayssm-buffer-len 16 \
  --no-enable-prefix-caching --enable-chunked-prefill --language-model-only \
  --max-model-len 4096 --max-num-seqs 16 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 --distributed-executor-backend mp \
  --no-async-scheduling --seed 42 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  2>&1 | tee /tmp/replayssm-l23-tp4.server.log
```

另开终端依次检查 `/health`、`/v1/models`，再用 `/v1/completions` 发送 temperature=0、
`max_tokens=64` 的请求。至少一个请求生成超过 32 tokens，确保跨过两次 L=16 flush。

- [ ] 服务加载完成；日志显示五块 GDN state，未回退 native recurrent。
- [ ] 首 token、pos 15 flush、pos 16 重新从 checkpoint 开始时无 C++ 校验或设备错误。
- [ ] HTTP 返回完整，生成 token 全部有限且服务在请求后继续健康。
- [ ] 停止服务后 4 卡无残留进程、board 无 error。

## A2：与原生 AR 做确定性对照

停止 Replay 服务，用同一模型、TP、seed、图配置和 batch 限制重启原生服务，只删除：

```text
--use-replayssm --replayssm-buffer-len 16
```

对 native 和 Replay 各发送同一组 temperature=0 请求并保存完整 JSON：

1. B=1，短 prompt，生成 64 tokens；
2. B=8，同时发起，生成 64 tokens；
3. B=16，同时发起，生成 64 tokens；
4. 长短 prompt 混合，迫使 prefill 与 decode 同批出现；
5. 连续三轮各 32 个短请求，验证完成请求的 slot 被新请求复用；
6. 降低 `max_num_batched_tokens` 形成调度压力，验证请求暂停/恢复后的 decode base 重锚。

- [ ] 每项 Replay 与 native 的 completion token IDs 一致；如出现首次分歧，回传 prompt、两份
  response、分歧 token 下标和前后各 8 个 token。
- [ ] B=1/8/16 均跨越至少三次 flush；无旧请求历史污染新 slot。
- [ ] mixed batch 中 decode slice 走 Replay，prefill 仍能正确写 checkpoint。
- [ ] graph replay 后改变 request/slot/cursor，输出随输入变化，不复用捕获期结果。

## A3：TP8 重跑与性能记录

把 A1/A2 的 `--tensor-parallel-size` 改为 8，端口和输出目录使用新名字。至少重跑
B=1/8/16、mixed batch、slot churn。分别保存 native/Replay 的 HTTP 端到端时延和服务端
decode profile；本轮不以性能数字决定正确性通过。

已知观察：F3 中 TP4/B16 的单算子 Replay 比 native 慢，TP8/B16 基本持平。先确认真实模型
是否复现，再决定 tile/core 或 FLA tiling 优化；不得为追性能跳过生命周期验证。

## 通过条件与回传材料

本轮通过需要 TP4/TP8 的 A0–A3 都有证据；L2.2 遗留的普通 native serve smoke 可由 A2 的
native 对照补齐。请回传：

- 本轮 commit、`git status --short`、构建日志和已安装包版本；
- TP4/TP8 的 native 与 Replay 启动命令、完整 server log；
- 每个矩阵的 HTTP 请求/响应、token 对照摘要、p50/p90；
- 首次不一致或崩溃的最小请求、对应 rank/device 日志；
- 每轮结束后的残留进程、`npu-smi` 和 board error 检查。

本清单完成前 `next_level_allowed` 保持 `false`，不开始 SD/MTP 算子实现。

# L3.6 ReplaySSM SD：`vllm serve` 首轮端到端验证

本轮代码以 CANN 9.2.0 / Ascend910B4 / torch_npu 2.10.0.post4 / vLLM `0.30.0+empty` / 本分支源码编译的 vLLM Ascend 为目标。仅验证 Qwen3.5/Qwen3.6 GDN、MTP3、TP4/TP8、BF16 模型与 BF16 Mamba/SSM cache、`mamba-cache-mode none`、V1 runner。verify 固定 **v13/tile128/cores40**，compact 固定 **v1/tile32/cores40**，cursor 使用已有 AscendC kernel。v3/v4 compact 保留在算子消融入口，serve 不启用。

插件通过 `--additional-config '{"replayssm_sd":true}'` 开启 SD。vLLM 0.30 原始配置校验将 `--use-replayssm` + MTP 解释为 KDA RecoverSSM 并拒绝 Qwen GDN，因此本轮 **不要同时传** `--use-replayssm`。Ascend 平台在上游校验后启用 `cache_config.use_replayssm`，复用上游 request 的 decode-base 生命周期；GDN cache spec 将 speculative block 数改为 0，候选都指向同一物理 block。此配置差异只在插件内实现，属于对 vLLM/Ascend 内部 cache-spec、metadata builder、model runner 与 GDN dispatch 的侵入式适配；没有修改 vLLM 源码。

## 0. 构建与独立门槛

1. 拉取本次提交，记录 `git rev-parse HEAD`、`git status --short`、实际 package 版本、CANN 路径和可见 910B4 卡数。服务器本地修改先另存，不要合并到本轮结果。
2. 在 9.2.0 容器重新从此 checkout 编译安装 vLLM Ascend，确认三种 `_C_ascend.replayssm_sd_*_` op 和 Meta 注册、kernel `.so` RUNPATH 都来自新构建；编译失败时保留首个 CCE/链接错误，不运行旧 `.so`。
3. 复跑 verify v13/tile128、compact v1、cursor 的 TP4 B2 smoke/graph 与 TP8 B2 graph；cursor 特别检查 accepted=0/1/4、reset、slot0，compact 检查 base31 跨尾和混合 flush。若 v13 在某 TP 或图模式下未通过，先停在算子层并回传结果；v11/tile64 可作诊断对照。

## 1. 顺序启动 native 与 Replay 两个服务

两次使用同一容器、权重、TP、编译模式、GPU 利用率、seed 和请求。先运行 native，完全退出后再运行 Replay。下方以 TP4、短上下文 smoke 为例；TP8 只改 `--tensor-parallel-size 8` 和可见卡。保存每次完整日志（包括 4/8 个 worker 的 `REPLAYSSM_MEMORY_JSON`）。

```bash
export REPLAYSSM_MODEL=/path/to/Qwen3.6-27B
export VLLM_ASCEND_REPLAYSSM_MEMORY_PROBE=1

vllm serve "$REPLAYSSM_MODEL" \
  --host 127.0.0.1 --port 18083 --served-model-name replayssm-sd \
  --tensor-parallel-size 4 --pipeline-parallel-size 1 \
  --dtype bfloat16 --mamba-cache-dtype bfloat16 \
  --mamba-ssm-cache-dtype bfloat16 --mamba-cache-mode none \
  --no-enable-prefix-caching --enable-chunked-prefill --language-model-only \
  --max-model-len 4096 --max-num-seqs 32 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 --distributed-executor-backend mp \
  --no-async-scheduling --seed 42 --no-enforce-eager \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
  2>&1 | tee /tmp/replayssm-sd-native-tp4.log
```

Replay 服务在 `2>&1 | tee` 之前增加 `--additional-config '{"replayssm_sd":true}'`，日志改为 `/tmp/replayssm-sd-replay-tp4.log`。**先确认启动日志显示 ReplaySSM SD enabled、图 capture 完成、`/health` 通过，再发请求。** 若模型的实际 speculative method 经上游规范化后不是 `mtp`，回传 config/首条异常；不要绕过配置校验。

## 2. HTTP 正确性与稳定性

准备同一份短 prompt 文件。native 与 Replay 各跑一次，每次输出到新路径；Replay 对 native 的 token IDs 做逐请求比较。脚本保留延迟、usage、token IDs、waiting 峰值和前后 `/metrics`。`--alternate-prompt-file` 可给出不同长度 prompt，触发混合 prefill/decode 和请求重排。

```bash
printf '请用两句话解释矩阵乘法，并给出一个例子。\n' > /tmp/replayssm-sd-short.txt
python3 scripts/replayssm/validate_sd_serve.py \
  --model replayssm-sd --prompt-file /tmp/replayssm-sd-short.txt \
  --concurrency 1 --repetitions 2 --max-tokens 128 \
  --output /tmp/replayssm-sd-native-smoke.json
```

Replay 启动后运行相同命令，输出改为 `/tmp/replayssm-sd-replay-smoke.json` 并加入 `--compare /tmp/replayssm-sd-native-smoke.json`。先 TP4，再 TP8。短上下文通过后，用实际 tokenizer 确认接近 4k 与 31k 的 prompt 文件；`max-model-len=32768` 时应给 512/1024 输出留足 token 空间。按 B1→B32→B64、输出 128→512→1024 逐级跑；对照组和 Replay 组保持相同 prompt、并发和输出长度。长 prompt 测试需要重启服务，把两组的 `max-model-len` 都改为 32768、`max-num-seqs` 改为 64，其余参数一致。

- [ ] native 与 Replay 单请求的 HTTP 状态、usage、token IDs 一致；若首个不同 token 出现，保存两个 JSON 与对应 worker 日志，先分析是否是 BF16/FP16 ring 误差，不继续压力测试。
- [ ] Replay 同一请求重复两次结果一致；短 prompt、长 prompt、T1 无 draft step、T4 verify、reject tail 都未发生状态漂移。通过 metrics 确认 MTP 草稿数增加。
- [ ] B32/B64 混合长度请求在 waiting、request 重排、部分请求完成后仍生成有效结果，无 cursor/verify/compact status 异常、NPU error、残留进程。若测试需要取消请求，记录取消的 request ID 和后续同 slot 新请求结果。
- [ ] 图 capture 与 replay 均通过；检查日志中实际调用 `replayssm_sd_verify_` v13、`replayssm_sd_compact_` v1 与 cursor，不把独立算子 graph 测试当作 serve 图证据。若失败，回传第一条 worker traceback 和图模式。

## 3. 显存与吞吐验收

在每次服务启动完成、预热后、B32/B64 运行中记录各卡 `npu-smi info`。对 native 和 Replay 日志执行：

```bash
python3 scripts/replayssm/summarize_memory_probe.py \
  --native /tmp/replayssm-sd-native-tp4.log \
  --replay /tmp/replayssm-sd-replay-tp4.log
```

- [ ] 每 rank 的 GDN cache plan：native `num_speculative_blocks=3` 对应的页数，Replay 为 0；Replay state 应有 BF16 conv/checkpoint、FP16 d/k ring、FP32 g ring，ring 长度 32。记录每个 GDN layer page bytes、每请求所需 block 数、`num_blocks` 和实际可容纳并发。
- [ ] 同时记录 `unique_view_backing_bytes`、`torch_allocated_bytes`、`torch_reserved_bytes`、`npu-smi` HBM；混合 cache group 可能 overlay，同一预算下 allocator 会用省下的每请求容量分配更多 block，因此**不能只用启动后总显存数判断收益**。
- [ ] B32/B64 的 `peak_waiting_requests`、成功请求数、总完成时间、每请求 latency p50/p90 与 native 对照；目标是先确认 Replay 提高可运行并发且没有不可接受的端到端时延回退。保存 TP4 与 TP8 各自的完整数据，不跨 TP 比较绝对 HBM。

本轮最可能需要上机修正的环节是配置校验时机、图内 metadata 地址、FLA speculative conv 对单物理 block 的读写约定、请求暂停/恢复后的 accepted count，以及 hybrid page overlay 的真实显存账。失败时停止后续依赖矩阵，回传 commit、命令、首条错误、相关 JSON、内存日志和卡健康状态；不要把未执行项标为通过。

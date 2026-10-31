# L3.9 ReplaySSM SD：MTP1/2/3 与 Qwen3.5-9B TP1/TP2 上机清单

目标环境：Ascend910B4、CANN 9.2.0、torch_npu 2.10.0.post4、vLLM `0.30.0+empty`、本分支源码编译的 vllm-ascend。沿用 BF16 模型/conv/checkpoint、FP16 D/K ring、FP32 G ring、history=16、独立请求 ring。verify 仍用 v13/tile128/cores40，compact 用 v1/tile32/cores40。本轮不改变已知 FP16 ring 数值口径。

TP2 是在 L3.9 首版之上的独立补测：当时仅扩展了测试几何。此后修复了 C++ 入口遗留的 `variant>=2` 且 `value_heads<=12` 限制。9B TP1/TP2 使用 v13 时分别需要 32/16 个本地 value heads；执行这两组验证前必须从包含此修复的源码重新编译并加载扩展，不能沿用仅含 L3.9 首版的 `.so`。原先在该校验处失败的 9B 组合按下文 smoke/graph 用例重测；已通过的 27B 组合不必重复。

`--speculative-config` 的 `num_speculative_tokens=N` 决定逻辑 MTP size（N=1、2、3）。verify 的物理输入保持 `[B,4,...]`，缺少的候选补零，真实 `lengths` 为 1 到 N+1；ring 长度仍为 32。KV `--block-size` 与这个 N 无关。9B TP1/TP2 每 rank 本地 QK/V head 数分别为 16/32、8/16；verify 的候选 gate/beta UB 暂存容量已扩到 32 个 value heads，C++ 入口超过 32 时明确拒绝。按 v13/tile128 的 `InitBuffer` 字节数相加约 184,352 B，距 192 KiB UB 上限约 12,256 B；编译器实际分配仍以上机结果为准。TP2 不改算子，只补验证几何和服务入口。

## 1. 构建与 CPU 门槛

1. 记录 commit、`git status --short`、CANN/torch_npu/vLLM/vllm-ascend 版本。拉取后重新从当前 checkout 编译安装 vllm-ascend，核对 `replayssm_sd_verify` 的新 `.so` 时间与路径，不能沿用旧二进制。若失败，先回传第一条 CCE/C++ 编译错误和周围日志，停止后续算子测试。
2. 执行：

   ```bash
   python3 -m unittest \
     tests.ut.replayssm.test_model_ar_integration \
     experimental.replayssm_sd_ascendc.tests.test_reference \
     experimental.replayssm_sd_cursor_ascendc.tests.test_reference \
     tests.ut.replayssm.test_framework_meta
   ```

   检查 MTP1/2/3 的 4-token 打包、9B TP1/TP2 的 16/32 与 8/16 head CPU 参考、cursor flush 边界和 Meta schema。任何失败先停在 CPU 层。

## 2. 单算子与图回放

先用空闲设备逐项跑，不在同一设备并行多个 NPU 进程。每个输出使用新路径；任何非零 status、精度超差、NPU error 或图捕获失败，先停在该组合。`run.py` 的 `graph` 阶段比较多个位置/长度的输出和 d/k/g 与 CPU 参考；`check_sd_independent_ring.py` 比较独立 ring 与同槽路径的 verify+compact 结果。

```bash
# 27B 几何；TP4 与 TP8 各跑 N=1,2,3
for tp in 4 8; do
  for n in 1 2 3; do
    python3 experimental/replayssm_sd_ascendc/run.py \
      --backend framework --geometry 27b --tp "$tp" --mtp-size "$n" \
      --variant 13 --tile-v 128 --cores 40 --stage graph --device 0 \
      --output "/tmp/l39-verify-27b-tp${tp}-mtp${n}.json" || exit 1
    python3 scripts/replayssm/check_sd_independent_ring.py \
      --geometry 27b --tp "$tp" --mtp-size "$n" --device 0 \
      --output "/tmp/l39-ring-27b-tp${tp}-mtp${n}.json" || exit 1
  done
done

# 9B 单卡；先 smoke，再 N=1,2,3 图回放
python3 experimental/replayssm_sd_ascendc/run.py \
  --backend framework --geometry 9b --tp 1 --mtp-size 3 \
  --variant 13 --tile-v 128 --cores 40 --stage smoke --device 0 \
  --output /tmp/l39-verify-9b-tp1-mtp3-smoke.json
for n in 1 2 3; do
  python3 experimental/replayssm_sd_ascendc/run.py \
    --backend framework --geometry 9b --tp 1 --mtp-size "$n" \
    --variant 13 --tile-v 128 --cores 40 --stage graph --device 0 \
    --output "/tmp/l39-verify-9b-tp1-mtp${n}.json" || exit 1
  python3 scripts/replayssm/check_sd_independent_ring.py \
    --geometry 9b --tp 1 --mtp-size "$n" --device 0 \
    --output "/tmp/l39-ring-9b-tp1-mtp${n}.json" || exit 1
done

# 9B TP2 补充：单 rank 算子只需一张空闲卡；先 smoke，再图回放。
python3 experimental/replayssm_sd_ascendc/run.py \
  --backend framework --geometry 9b --tp 2 --mtp-size 3 \
  --variant 13 --tile-v 128 --cores 40 --stage smoke --device 0 \
  --output /tmp/l39-verify-9b-tp2-mtp3-smoke.json
for n in 1 2 3; do
  python3 experimental/replayssm_sd_ascendc/run.py \
    --backend framework --geometry 9b --tp 2 --mtp-size "$n" \
    --variant 13 --tile-v 128 --cores 40 --stage graph --device 0 \
    --output "/tmp/l39-verify-9b-tp2-mtp${n}.json" || exit 1
  python3 scripts/replayssm/check_sd_independent_ring.py \
    --geometry 9b --tp 2 --mtp-size "$n" --device 0 \
    --output "/tmp/l39-ring-9b-tp2-mtp${n}.json" || exit 1
done

# Cursor 与模型几何无关；三种逻辑投机长度各跑一次图回放
for n in 1 2 3; do
  python3 experimental/replayssm_sd_cursor_ascendc/run.py \
    --backend framework --mtp-size "$n" --batch 32 --stage graph --device 0 \
    --output "/tmp/l39-cursor-mtp${n}.json" || exit 1
done
```

## 3. `vllm serve` 逐级验证

沿用 [L3.8 服务参数](LEVEL3_SD_INDEPENDENT_RING.md)与 [L3.6 完整启动命令](LEVEL3_SD_SERVE.md)。每轮只改变模型、TP、`num_speculative_tokens`，其余参数保持一致；启用 `--additional-config '{"replayssm_sd":true}'`，不要传 `--use-replayssm`。每个服务实例的 N 在启动时固定，不支持运行中按请求切换。先对已通过的 27B TP4/MTP3 做回归，再逐个跑 TP4 的 MTP1/2、TP8 的 MTP1/2/3，最后跑 9B TP1/TP2 的 MTP1/2/3。9B 用实际 Qwen3.5-9B BF16 权重，先 `--language-model-only` 做文本请求；从 `--max-model-len 4096 --max-num-seqs 8` 开始，再按显存余量升到 16/32。9B 的 ring 粗算每请求槽约 TP1 9.1 MiB/卡、TP2 4.55 MiB/卡（24 个 GDN 层、长度 32），配置上限和实际占用须以 memory probe 为准。

每轮使用对应的投机配置，例如 N=2：

```bash
--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":2,"enforce_eager":true}'
```

9B TP1 的首个服务冒烟可直接使用（将模型路径替换为服务器上的实际路径，N 先取 1）：

```bash
export REPLAYSSM_9B_MODEL=/path/to/Qwen3.5-9B
export VLLM_ASCEND_REPLAYSSM_MEMORY_PROBE=1
vllm serve "$REPLAYSSM_9B_MODEL" \
  --host 127.0.0.1 --port 18083 --served-model-name replayssm-9b \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --dtype bfloat16 --mamba-cache-dtype bfloat16 \
  --mamba-ssm-cache-dtype bfloat16 --mamba-cache-mode none \
  --no-enable-prefix-caching --enable-chunked-prefill --language-model-only \
  --max-model-len 4096 --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 --no-async-scheduling --seed 42 \
  --no-enforce-eager --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1,"enforce_eager":true}' \
  --additional-config '{"replayssm_sd":true}' \
  2>&1 | tee /tmp/l39-serve-9b-tp1-mtp1.log
```

9B TP2 在两张空闲卡上以同一命令启动，改 `--tensor-parallel-size 2`，日志改 `/tmp/l39-serve-9b-tp2-mtp1.log`；通过后把 `num_speculative_tokens` 依次改为 2、3，每次重新启动服务并使用独立日志/JSON。已有 TP1 和 27B 测试可继续按原提交执行，TP2 是单独追加的验证组合。

检查启动日志中的 `MTP1/2/3`、`independent_sd_ring`、`ring_len=32`、图 capture 和 `/health`=200。用 `scripts/replayssm/validate_sd_serve.py` 依次跑 B1 短 prompt、混合长度 B8、B32（9B 先不超过已启动的 `max-num-seqs`），然后 27B 的 32k/512 B32 场景。至少覆盖无草稿 token、部分接受、全接受、请求完成后 ring id 重用、waiting→running；每轮保存 HTTP JSON、worker 日志、memory probe、metrics、NPU 健康状态。

对相同模型/TP/MTP size 的 native 与 Replay 使用同容器、同权重、同 seed 和请求，记录首分叉位置与 top-2 margin。L3.7 的 index 21 近平局分叉是已知数值遗留问题；新组合不得出现明显更早的大 margin 分叉、状态码错误、历史污染、OOM 或 NPU error。不要只凭输出语句通顺判为数值通过。

## 4. 性能与回传

分别记录 verify、compact、cursor 的 msprof task duration 和完整投机轮次延迟；27B 重点比较 N=1/2/3 的同场景，9B TP1/TP2 各自建基线，不跨模型或 TP 推算单算子时延。记下有效 running 峰值、waiting 峰值、总完成时间、TTFT/TPOT（若工具可得）、`num_gpu_blocks`、ring 容量和每 rank HBM。MTP3/27B 与 L3.8 已通过的 **peak running=18** 对照，若回退先检查 UB/图捕获和缓存容量。

回传 commit、服务命令、每项 JSON、首个失败 traceback、CANN 编译首错、`npu-smi` 健康状态。第 2 节全部通过后才进入服务测试；9B TP1/TP2 的端到端结论必须以实际权重和完整图回放为准。

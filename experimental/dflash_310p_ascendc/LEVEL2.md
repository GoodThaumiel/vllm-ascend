# 310P DFlash MRv1 · L2A / L2B 服务验证

硬件为 Ascend310P3，两张空闲卡，CANN 9.2.1，vLLM / vLLM-Ascend 0.30.0。复用通过 LEVEL1 的 `experimental/dflash_310p_ascendc/build/libdflash_310p_torch.so`。本轮 Python-only，已有该库时无需重新编译 AscendC 或 vLLM-Ascend。

使用现有 35B-A3B W8A8 target 及对应版本的 DFlash 权重。确认 3.5 或 3.6 配对、记录两个 checkpoint 的配置和 revision。首轮只测文本、TP2/DP1/PP1、FP16、greedy、K=7、原生 RGDR。关闭 async scheduling、prefix caching、grouped RGDR 和 ReplaySSM。

## 1. 启动前

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
export ASCEND_RT_VISIBLE_DEVICES=4,5  # 换成实际空闲的两张卡
unset VLLM_ASCEND_310P_GROUPED_RGDR_LIBRARY
unset VLLM_ASCEND_REPLAYSSM_310_TRACE_CONFIG
export TARGET=/path/to/35B-A3B-w8a8
export DRAFT=/path/to/matching-DFlash
export RESULTS=/data/l00972401/dflash-level2
mkdir -p "$RESULTS"
git rev-parse HEAD
git status --short
python3 -m unittest \
  experimental.dflash_310p_ascendc.test_reference \
  tests.ut.replayssm.test_dflash_310p_mask \
  tests.ut.replayssm.test_dflash_310p_service_contract
```

验证结果目录内放一份代表性短 prompt `short.txt`，和至少 20 条不同长度的 `prompts.jsonl`。后者每行使用 `{"prompt":"..."}`，混合长短请求，最长约 2k token，用于 C10 补入。保留相同文件用于 eager/graph 对照。

## 2. L2A · eager C1

```bash
SPEC=$(python3 -c 'import json,os; print(json.dumps(dict(method="dflash", model=os.environ["DRAFT"], num_speculative_tokens=7, draft_tensor_parallel_size=2, draft_sample_method="greedy")))')
COMMON=(
  "$TARGET" --host 127.0.0.1 --port 18083 --served-model-name dflash310
  --tensor-parallel-size 2 --dtype float16 --quantization ascend
  --mamba-cache-dtype float16 --mamba-ssm-cache-dtype float16
  --max-model-len 4608 --max-num-batched-tokens 4608 --max-num-seqs 10
  --gpu-memory-utilization 0.85 --no-enable-prefix-caching --no-async-scheduling
  --speculative-config "$SPEC"
)
vllm serve "${COMMON[@]}" --enforce-eager 2>&1 | tee "$RESULTS/eager.log"
```

另一个终端，确认 `/health` 返回 200，两个 rank 均打印 `310P DFlash MRv1 draft KV groups`。应看到全注意力和滑窗层分配到不同物理组。跑短 prompt 和 2k prompt，验证接受 token 计数随请求增加。

```bash
python3 scripts/replayssm/validate_sd_serve.py --model dflash310 \
  --prompt-file "$RESULTS/short.txt" --max-tokens 128 --logprobs 5 \
  --concurrency 1 --repetitions 2 --output "$RESULTS/eager-c1.json"
python3 scripts/replayssm/validate_sd_serve.py --model dflash310 \
  --prompts-jsonl "$RESULTS/prompts.jsonl" --max-tokens 400 \
  --concurrency 10 --repetitions 2 --output "$RESULTS/eager-c10.json"
```

## 3. L2B · 双图 C1 / C10 补入

停止 eager 服务后，在相同设备启动。首版使用 8-token 的精确 descriptor，覆盖 1–10 个活跃请求。

```bash
vllm serve "${COMMON[@]}" \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[8,16,24,32,40,48,56,64,72,80]}' \
  2>&1 | tee "$RESULTS/graph.log"
```

```bash
python3 scripts/replayssm/validate_sd_serve.py --model dflash310 \
  --prompt-file "$RESULTS/short.txt" --max-tokens 128 --logprobs 5 \
  --concurrency 1 --repetitions 2 --compare "$RESULTS/eager-c1.json" \
  --output "$RESULTS/graph-c1.json"
python3 scripts/replayssm/validate_sd_serve.py --model dflash310 \
  --prompts-jsonl "$RESULTS/prompts.jsonl" --max-tokens 400 \
  --concurrency 10 --repetitions 2 --output "$RESULTS/graph-c10.json"
grep -E '310P DFlash (MRv1 draft KV groups|graph replay|graph ready)' "$RESULTS/graph.log"
```

C1 要求每个 TP rank 都出现 `component=target` 和 `component=draft` 的 8-token **replay**。C10 要求 20/20 完成、出现多请求 descriptor 的双图 replay，并且补入后无 QBM/L0C、507015、越界或非法 metadata。出现分叉时保留同一 prompt 序列的首个不匹配位置和 top-5，作为后续固定历史定位的输入。原始服务压测 JSON 与日志用于本轮验收，性能基线另测。

回传两份服务日志、四份 JSON、首个错误（含 rank、descriptor 和 CANN 日志），以及 `git status --short`。若服务启动或 C1 已失败，可直接回传该项日志。

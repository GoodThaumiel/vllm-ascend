# 310P ReplaySSM SD V12 服务实验

使用 Ascend310P3 双卡和 V12 的 CANN 9.2.1、驱动 26.1.0.b098、torch_npu
2.10.0.post7 环境。模型为 Qwen3.5-35B-A3B W8A8，TP2、PP1、MTP3，模型计算
和 Mamba cache 均为 FP16。服务上限 64，压测输入 2048 token、输出 400 token。

Replay 通过 `--additional-config '{"replayssm_sd":true}'` 启用，使用 V1 runner
和同步调度，关闭 prefix caching 与 sequence parallelism。保持上游
`cache_config.use_replayssm` 为 false。卷积保留原生 MTP 历史，SSM 使用一份
FP16 checkpoint 加独立请求 ring。Verify 为 V12 tile64/status variant 1，
compact 为 tile32，均使用 7 核。Cursor 按服务语义接受最多 4 个 token
（anchor + 3 drafts），因此 flush 阈值采用窗口宽度 4。

## 1. 构建与上线前检查

在 310P 节点上使用本 checkout 的 editable vLLM Ascend 环境。主插件的
`_C_ascend` 仍需包含原生 V310 算子。另行重建 Replay 扩展，路径必须与下方
一致，worker 会从当前源码 checkout 加载这份扩展。

```bash
git rev-parse HEAD
git status --short
npu-smi info

cmake -S experimental/replayssm_sd_310p_ascendc \
  -B experimental/replayssm_sd_310p_ascendc/build \
  -DSOC_VERSION=Ascend310P3 \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.1 \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build experimental/replayssm_sd_310p_ascendc/build -j 8

python3 -m unittest tests.ut.replayssm.test_310p_sd_demo \
  experimental.replayssm_sd_310p_ascendc.tests.test_reference
for stage in smoke sequence graph; do
  python3 experimental/replayssm_sd_310p_ascendc/run.py \
    --stage "$stage" --mtp 3 --tile-v 64 --cores 7 --status-variant 1 \
    --output "/tmp/replayssm-310p-serve-$stage.json" || break
done
```

`sequence` 必须覆盖全接受 4、部分接受和 ring flush/wrap。三项均通过后启动服务。
若原生算子加载缺少符号，沿用 [V12 的 vendor OPP 路径](LEVEL310P_V12.md)。

## 2. 同配置启动 native 和 Replay

先串行跑 native/eager 与 replay/eager，确认服务与生命周期，再跑
native/graph 与 replay/graph。每轮完成后停止服务并等待两卡释放。
`REPLAYSSM_MODEL` 指向同一份已验证的 W8A8 权重，保留权重原生量化配置。
ModelSlim 权重需要显式量化选择时，两边均使用 `--quantization ascend`。

```bash
export REPLAYSSM_MODEL=/path/to/Qwen3.5-35B-A3B-W8A8
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ASCEND_REPLAYSSM_MEMORY_PROBE=1
export RESULTS=/tmp/replayssm-310p-serve
mkdir -p "$RESULTS"

run_serve() {
  local variant="$1" execution="$2"
  local -a opts=()
  if [[ "$variant" == replay ]]; then
    opts+=(--additional-config '{"replayssm_sd":true}')
  fi
  if [[ "$execution" == eager ]]; then
    opts+=(--enforce-eager)
  else
    opts+=(--cudagraph-metrics --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,16,32,64,128,192,256]}')
  fi
  vllm serve "$REPLAYSSM_MODEL" \
    --host 127.0.0.1 --port 18083 --served-model-name replayssm-310p \
    --tensor-parallel-size 2 --pipeline-parallel-size 1 --distributed-executor-backend mp \
    --dtype float16 --mamba-cache-dtype float16 --mamba-ssm-cache-dtype float16 \
    --mamba-cache-mode none --no-enable-prefix-caching --enable-chunked-prefill \
    --no-async-scheduling --language-model-only --seed 42 \
    --max-model-len 4096 --max-num-seqs 64 --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.85 \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
    "${opts[@]}" 2>&1 | tee "$RESULTS/$variant-$execution.log"
}

run_serve native eager
# 后续各轮分别启动：run_serve replay eager、run_serve native graph、run_serve replay graph
```

Replay 首轮 graph 仅复用与已捕获形状完全一致的满宽 decode，其他形状走 eager。
Graph 比较时记录实际 replay 覆盖率。Native 默认可重放有 padding 的形状，因而
eager A/B 可用于隔离算子和缓存收益，graph A/B 用于衡量当前服务策略下的总收益。

## 3. 正确性和 2k/400 压测

每轮在另一终端设置同一 `REPLAYSSM_MODEL`、`RESULTS`，并按运行中的服务设置
`VARIANT` 和 `EXECUTION`。先确认 `/health`，检查 Replay 两个 rank 均有
`310P ReplaySSM SD V12 dispatch` 以及独立 ring 分配日志。用固定的短 prompt
记录两次温度为零的输出。Replay 用同模式下 native 的 JSON 作 token 对照。

```bash
VARIANT=native
EXECUTION=eager
printf '请用三句话解释矩阵乘法。\n' > "$RESULTS/prompt.txt"
compare=()
if [[ "$VARIANT" == replay ]]; then
  compare+=(--compare "$RESULTS/native-$EXECUTION-smoke.json")
fi
python3 scripts/replayssm/validate_sd_serve.py \
  --model replayssm-310p --prompt-file "$RESULTS/prompt.txt" \
  --max-tokens 128 --logprobs 5 --concurrency 1 --repetitions 2 \
  --output "$RESULTS/$VARIANT-$EXECUTION-smoke.json" "${compare[@]}"
```

记录首个 token 分歧及其 logprob，再用同一短 prompt 在连续完成/新建请求的
情况下重跑，检查是否与 ring slot 复用有关。出现 NaN、请求失败、历史串扰或
明显偏离时，先停止后续压测并保留 worker 日志。Native 和 Replay 的 FP16
状态重建舍入可能导致贪心分歧，需要结合首个分歧的数据判断。

```bash
for bs in 1 16 32 64; do
  prefix="$RESULTS/$VARIANT-$EXECUTION-b$bs"
  curl -fsS http://127.0.0.1:18083/metrics > "$prefix.before.metrics"
  (
    while true; do
      date +%s
      curl -fsS http://127.0.0.1:18083/metrics | \
        rg '^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_preemptions_total)'
      sleep 1
    done
  ) > "$prefix.samples" &
  sampler=$!
  vllm bench serve --backend openai --endpoint /v1/completions \
    --host 127.0.0.1 --port 18083 --model "$REPLAYSSM_MODEL" --served-model-name replayssm-310p \
    --dataset-name random --random-input-len 2048 --random-output-len 400 --random-range-ratio 0 \
    --seed 42 --temperature 0 --ignore-eos --request-rate inf --max-concurrency "$bs" \
    --num-prompts "$((2 * bs))" --num-warmups 4 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --save-detailed --result-dir "$RESULTS" --result-filename "${prefix##*/}.json"
  bench_status=$?
  kill "$sampler"
  wait "$sampler" 2>/dev/null || true
  curl -fsS http://127.0.0.1:18083/metrics > "$prefix.after.metrics"
  npu-smi info > "$prefix.npu-smi"
  [[ "$bench_status" == 0 ]] || break
done
```

回传四组服务日志、HTTP smoke JSON、bench JSON、metrics 与 npu-smi。对照
实际完成数、运行/等待峰值、preemption、输出 tok/s、TTFT/TPOT p50/p90、
MTP 接受数与草稿数增量，以及 KV block 容量。记录每个 rank 在
`REPLAYSSM_MEMORY_JSON` 中的 checkpoint、ring 与 cache backing。
每卡 64 槽的 ring 张量 payload 约 369.4 MiB，满载 MTP3 下仅 SSM 张量的
投影净节省约 2.45 GiB/卡。实际可运行并发以 scheduler 指标为准。

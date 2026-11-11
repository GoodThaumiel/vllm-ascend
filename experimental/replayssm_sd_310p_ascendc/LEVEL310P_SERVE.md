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
  local mtp="${MTP:-3}" max_seqs="${MAX_SEQS:-64}" memory_util="${MEMORY_UTIL:-0.85}"
  shift 2
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
    --max-model-len 4096 --max-num-seqs "$max_seqs" --max-num-batched-tokens 4096 \
    --gpu-memory-utilization "$memory_util" \
    --speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$mtp,\"enforce_eager\":true}" \
    "${opts[@]}" "$@" 2>&1 | tee "$RESULTS/$variant-$execution.log"
  return "${PIPESTATUS[0]}"
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
run_bench() {
  local bs="$1" prefix="$2"
  local sampler bench_status=0
  curl -fsS http://127.0.0.1:18083/metrics > "$prefix.before.metrics" || return
  npu-smi info > "$prefix.before.npu-smi"
  (
    while true; do
      date +%s
      curl -fsS --max-time 2 http://127.0.0.1:18083/metrics | \
        rg '^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_preemptions_total|prompt_tokens_total|generation_tokens_total)(\{|[[:space:]])'
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
    --save-result --save-detailed --result-dir "$RESULTS" --result-filename "${prefix##*/}.json" || bench_status=$?
  kill "$sampler"
  wait "$sampler" 2>/dev/null || true
  curl -fsS http://127.0.0.1:18083/metrics > "$prefix.after.metrics"
  npu-smi info > "$prefix.npu-smi"
  return "$bench_status"
}

for bs in 1 16 32 64; do
  run_bench "$bs" "$RESULTS/$VARIANT-$EXECUTION-b$bs" || break
done
```

回传四组服务日志、HTTP smoke JSON、bench JSON、metrics 与 npu-smi。对照
实际完成数、运行/等待峰值、preemption、输出 tok/s、TTFT/TPOT p50/p90、
MTP 接受数与草稿数增量，以及 KV block 容量。记录每个 rank 在
`REPLAYSSM_MEMORY_JSON` 中的 checkpoint、ring 与 cache backing。
每卡 64 槽的 ring 张量 payload 约 369.4 MiB。满载 MTP3 下按逻辑 SSM
张量内容估算的净节省约 2.45 GiB/卡，实际 backing 和可运行并发按第 5 节实测。

## 4. 2026-11-10 上机结果复测

原报告记录两条 128-token Replay 请求首次分歧均在 idx 45，eager 与 graph
重现一致。V12 的 3.2%–6.5% 是耗时下降，独立算子的数值检查通过。
BS64 的 acceptance 和显存记录作为观测值保留。要判断是否有 state 污染，
需先比对相同 MTP 窗口。显存记录说明分配和规划变化，容量收益还需以同预算下
实际可运行的请求数为准。

在报告使用的双卡和同一份 Qwen3.6-35B-A3B-W8A8 权重上，用第 2 节的
eager 启动参数。此版本纳入了上机时手工加的 FP16 dtype 门控。Python 源码可
复用报告中已重新构建的 V12 扩展。若需重建扩展，用新的 build 目录，沿用第
1 节的普通构建命令。

1. 先用无 trace 的 Replay 服务复跑原 prompt，确认启动及 idx 45 是否重现。
2. 分别启动下方的 native 录制服务和四组 Replay 对照服务，每次只发送同一个
   请求，使用报告原来的 prompt 文件。每组完成后停止当前服务。
3. 比对两个 rank 的相同输入、相同 draft 窗口和相同已接受 token。Replay
   继续计算自己的 logits 和 sampler 决策，之后按 Native 记录推进下一轮。
   使用独立输出目录，保留充足空间供各层 state 快照写盘。

```bash
# 在定义了 run_serve 的同一个 bash 中执行。TRACE_ROOT 必须是全新目录。
export TRACE_ROOT="$RESULTS/idx45-trace"
mkdir -p "$TRACE_ROOT"
export VLLM_ASCEND_REPLAYSSM_310_TRACE_CONFIG="$TRACE_ROOT/config.json"

printf '{"output":"%s/native"}\n' "$TRACE_ROOT" > "$VLLM_ASCEND_REPLAYSSM_310_TRACE_CONFIG"
run_serve native eager
```

在另一终端用以下命令提交一次。每轮设置对应的 `RUN` 值，native 为
`native`，其余为 `t32-v0`、`t64-v0`、`t32-v1`、`t64-v1`。保留原
`prompt.txt`，避免重新分词后换了复现输入。

```bash
python3 scripts/replayssm/validate_sd_serve.py \
  --model replayssm-310p --prompt-file "$RESULTS/prompt.txt" \
  --concurrency 1 --repetitions 1 --max-tokens 128 \
  --output "$TRACE_ROOT/http-${RUN}.json"
```

Native 完成后，逐组在服务终端设置 `TILE`、`VARIANT` 并启动 Replay。
每次服务日志可从第 2 节的日志路径归档到对应的输出目录。

```bash
TILE=32
VARIANT=0
RUN="t$TILE-v$VARIANT"
printf '{"output":"%s/%s","reference":"%s/native","tile":%d,"status_variant":%d}\n' \
  "$TRACE_ROOT" "$RUN" "$TRACE_ROOT" "$TILE" "$VARIANT" > "$VLLM_ASCEND_REPLAYSSM_310_TRACE_CONFIG"
run_serve replay eager
```

```bash
for run in t32-v0 t64-v0 t32-v1 t64-v1; do
  python3 scripts/replayssm/compare_310p_serve_trace.py \
    "$TRACE_ROOT/native" "$TRACE_ROOT/$run" --output "$TRACE_ROOT/$run.json" || break
done
```

回传四组对照 JSON、服务日志和对应的 HTTP JSON。重点看首个非零状态码、
首个超容差的 GDN 输入/输出、实际 ring position/base/flush/reset、
重建的候选 SSM state、完整 logits 误差及同一候选对的 margin。
若 Native 上一轮接受数超过当前窗口长度，保留对应的 step 与 state slot。
原生 V310 的选槽逻辑对这种组合有提前退出分支，需要由实际 trace 确认
有没有命中。

上面的强制对照用于定位发散。关闭 trace 后用同一 prompt 自由生成确认
修复，再恢复 2k/400 压测。

## 5. 并发容量与同并发显存

沿用上机报告的同一份 W8A8 权重、同一对卡和第 2 节启动参数。先完成第 3 节的
服务正确性检查。两边均关闭 trace，保持 TP2、MTP3、2k/400、`max-num-seqs=64`
和 Replay 默认 64 个请求 ring 槽。首轮比较 eager，有余量再用同样口径比较 graph。
每次先停止上一服务并等待显存释放，再启动下一服务。

### 同预算，测有效并发

Native 和 Replay 均使用 `gpu-memory-utilization=0.85`，使用各自自动规划的
block 数。每个服务完成两轮 B64，每轮 128 个测量请求。服务终端执行
`unset VLLM_ASCEND_REPLAYSSM_310_TRACE_CONFIG` 后，依次使用第 2 节的
`run_serve native eager` 和 `run_serve replay eager`。
两边终端使用同一个新 `RESULTS` 目录，避免覆盖之前的服务日志。客户端复用
第 3 节的 `run_bench` 函数，按服务分别设置 `VARIANT` 后执行。

```bash
VARIANT=native  # Replay 服务启动后改为 replay
EXECUTION=eager
for round in 1 2; do
  run_bench 64 "$RESULTS/capacity-$VARIANT-$EXECUTION-r$round" || break
done
```

- 检查每轮 bench JSON 的完成数为 128、失败数为 0，并核对两边实际输入和输出
  token 长度。记录输出 tok/s、TTFT p50/p90 和 TPOT p50/p90。
- 用 `.samples` 标出仍有 waiting 且持续产出 token 的 decode 时段。选取
  prefill 明显放缓、尚未进入收尾的连续样本，记录时间范围和其中的 running
  中位数及范围，同时报告整轮峰值。用前后 metrics 的差值计算 preemption。
  若只有瞬时峰值或始终没有排队，报告该限制。若两边在 decode 阶段均维持 64 且无等待，
  当前测试只确认都能承载 64。
- 有效并发收益按两轮可复现的 decode running 对比。结合 waiting、
  KV usage、preemption 和 TTFT 判断是否缓解了 cache 准入压力。
  把 prefill 或收尾造成的短暂波动留在原始记录里。

### 同有效并发，测实际显存

如果 Replay 在同预算下持续承载更多请求，用 native 自动预算的结果作目标。
保持 B64 的请求负载和服务上限，逐轮用 `--num-gpu-blocks-override` 降低
Replay 的 block 数。起点取低于 Replay 自动规划值的档位，每次重启后核对探针
记录的实际 block 数。找到两轮均能维持 native decode 并发且无新增 preemption
或请求失败的较小档位后，再比较显存。若无法匹配，回传最接近的档位和对应并发。

```bash
# 服务终端执行，输入本轮实测后选定的 block 数。
read -r -p 'Replay block 数 ' BLOCKS
run_serve replay eager --num-gpu-blocks-override "$BLOCKS"
```

每次启动前把两边终端的 `RESULTS` 指向新目录，保留该档位自己的服务日志和
压测结果。压测仍使用上面的两轮 B64 命令。

回传每轮的 bench JSON、`.samples`、前后完整 metrics、启动日志和
`npu-smi`。从每个 rank 的 `kv_cache_initialized` 阶段同时提取
`cache_plan.num_blocks`、`cache_views.unique_backing_bytes`、
`independent_sd_ring.allocated_bytes_per_rank`、`torch_allocated_bytes` 和
`torch_reserved_bytes`。用去重 backing 加独立 ring 对比物理缓存分配，再用
同一阶段的 allocator 和设备读数对照整卡占用。`planned_device_backing_bytes`
留作诊断 block planner。

## 6. MTP7、96 并发容量对照

使用 V12 上机报告的同一份 Qwen3.6-35B-A3B W8A8 权重、同一对 310P3 卡、
CANN 9.2.1 和 FP16 state。Native 与 Replay 均使用 eager、TP2、MTP7、
`max-num-seqs=96`、2k/400 和相同的显存利用率 0.85。Replay 的逻辑 history
上限为 16，每个请求的物理 ring 长度固定为 32，槽池为 96 个请求槽加 slot 0。
按已测模型几何预计 ring 约 551.3 MiB/卡。Cursor 沿用为两轮候选留空间的规则，
在 MTP7 下首轮有接受结果后，后续有非空历史的轮次会触发 compact flush。

先用第 1 节命令重建独立扩展并跑 CPU 单测。用下列命令验证 8-token 路径，
包括 B96 状态块归属、混合 flush、物理 ring 跨尾及接受 8 token 后的状态推进。
若有失败，回传首条报错与对应 JSON，再排查服务。

```bash
export RESULTS=/tmp/replayssm-310p-mtp7-capacity
mkdir -p "$RESULTS"
(
  set -e
  for test in smoke:2 sequence:1 smoke:96; do
    stage="${test%:*}"
    batch="${test#*:}"
    python3 experimental/replayssm_sd_310p_ascendc/run.py \
      --stage "$stage" --mtp 7 --batch "$batch" --tile-v 64 --cores 7 --status-variant 1 \
      --output "$RESULTS/operator-$stage-b$batch.json"
  done
)
```

在定义了第 2 节 `run_serve` 的服务终端执行。Native 跑完后停止服务，等两卡
释放后启动 Replay。每次先检查 `/health`，按第 3 节发送短请求。
Replay 两个 rank 应报告 `MTP7`、`width=8`、`capacity=96` 和 `ring_len=32`。
检查 Native GDN 的 `num_speculative_blocks=7`，Replay 为 0。发现请求失败、
状态错误或明显异常输出时，保留日志并暂停容量测试。

```bash
unset VLLM_ASCEND_REPLAYSSM_310_TRACE_CONFIG
export RESULTS=/tmp/replayssm-310p-mtp7-capacity
export MTP=7 MAX_SEQS=96 MEMORY_UTIL=0.85
run_serve native eager
# Native 结束后执行 run_serve replay eager
```

客户端复用第 3 节的 `run_bench` 定义，在同一 `RESULTS` 目录下执行。每轮
192 个测量请求，客户端最多同时发 96 个。按当前服务设置 `VARIANT`。

```bash
VARIANT=native  # Replay 服务启动后改为 replay
for round in 1 2; do
  run_bench 96 "$RESULTS/mtp7-capacity-$VARIANT-r$round" || break
done
```

按第 5 节口径回传稳定 decode 的 running 范围与中位数、waiting、KV usage、
preemption，以及每 rank 的实际缓存 backing、ring 和 allocator 数据。若两边
均达到 96 且 decode 无等待，记录双方实测上限均至少为 96。需要继续找容量拐点时，
保持 96 槽和 MTP7，另开结果目录，在两边用相同的较低 `MEMORY_UTIL` 重测。
若 0.85 下 Replay 已显示更高有效并发，可按第 5 节在匹配 Native 并发的档位
补测实际显存差值。

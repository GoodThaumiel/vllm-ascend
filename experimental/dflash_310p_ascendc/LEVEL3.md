# 310P DFlash MRv1 · L3 ReplaySSM SD 服务验收

在 Ascend310P3 两张空闲卡上使用 CANN 9.2.1、torch_npu
2.10.0.post7、vLLM/vLLM-Ascend 0.30.0 和匹配的 35B-A3B W8A8
target/drafter。沿用 TP2/DP1/PP1、MRv1、FP16 state、greedy、
无 prefix cache、同步调度和 `FULL_DECODE_ONLY`。先验收 K=7。

Replay 路径为 **cursor → verify → compact**。元数据阶段按 GDN KV
group 运行 cursor，按上一轮接受数（含 anchor）推进请求 ring。
Target forward 中每个 GDN 层由 grouped Cube verify 写当前候选，再由
batched FP32 compact 按需把旧历史折回独立的 FP16 checkpoint。
D/K ring 为 FP16，G ring 为 FP32，逻辑 history 16，物理 ring 32。
Ring 容量默认等于 `max-num-seqs`，含一个额外的保留槽。DFlash
仍使用原有的 draft forward 图和逐组 KV，MTP3/7 沿用之前的 V12 分派。

## 1. 构建和单算子

在仓库根目录执行。保留通过 L1 的 DFlash 输入准备 `.so`，重建
独立 ReplaySSM 扩展。两份库均从本仓库 checkout 的 `build` 读取。

```bash
set -eo pipefail
export TARGET=/path/to/Qwen3.6-35B-A3B-w8a8
export DRAFT=/path/to/Qwen3.6-35B-A3B-DFlash
export RESULTS=/data/l00972401/dflash-level3
mkdir -p "$RESULTS"
git rev-parse HEAD | tee "$RESULTS/commit.txt"
git status --short | tee "$RESULTS/git-status.txt"

op=experimental/replayssm_sd_310p_ascendc
cmake -S "$op" -B "$op/build" -DCMAKE_BUILD_TYPE=Release \
  -DSOC_VERSION=Ascend310P3 -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.1 \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build "$op/build" -j8 2>&1 | tee "$RESULTS/build.log"
lib="$op/build/libreplayssm_sd_310p_torch.so"
readelf -d "$lib" | grep -E 'RUNPATH|RPATH'
ldd "$lib"
sha256sum "$lib" experimental/dflash_310p_ascendc/build/libdflash_310p_torch.so

python3 -m unittest \
  experimental.replayssm_sd_310p_ascendc.tests.test_reference \
  experimental.dflash_310p_ascendc.test_reference \
  tests.ut.replayssm.test_310p_sd_demo \
  tests.ut.replayssm.test_dflash_310p_service_contract

# 使用一张空闲卡，--device 对应可见设备编号。
for stage in smoke sequence graph; do
  python3 "$op/run.py" --mtp 7 --batch 2 --device 0 --stage "$stage" \
    --library "$lib" --cores 7 --tile-v 64 --status-variant 2 --compact-variant 2 \
    --output "$RESULTS/mtp7-kernel-$stage.json"
done
```

RUNPATH 应含 `$ORIGIN/lib`。单算子验收读各 JSON 的状态、误差、
跨尾及 ring 不变性。发生设备错误或精度失败时，服务测试前先回传
首个失败用例、原始栈和对应 CANN 日志。若要核对纯算子时延，可在
同一卡按 [Cube/compact 消融](../replayssm_sd_310p_ascendc/CUBE_GROUPED_REDESIGN.md)
测 B32/B64 的 mixed、full_flush 及两版 compact 对照。

## 2. K=7 服务启动

启动前使用当前仓库的 editable vLLM-Ascend，并记录实际加载的
vLLM/vLLM-Ascend 路径、版本与两份 checkpoint revision。移除外置
v24 KV fix 和 0.23 RGDR monkey patch，保持所有路线的 prefill
开关、权重、卡对、预算与 capture sizes 一致。使用之前的
`short.txt` 和至少 20 条固定长短混合的 `prompts.jsonl`。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1  # 换成实际空闲卡
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ASCEND_310P_DFLASH_SHARED_KV=1
export VLLM_ASCEND_310P_DFLASH_V24_KV_BUDGET=1
export VLLM_ASCEND_REPLAYSSM_MEMORY_PROBE=1
export VLLM_ASCEND_310P_DFLASH_STAGE_TRACE=0
unset VLLM_DFLASH_KV_FIX VLLM_ASCEND_310P_PREFILL_JIT
unset VLLM_ASCEND_REPLAYSSM_310_TRACE_CONFIG VLLM_ASCEND_310P_GROUPED_RGDR_LIBRARY
export K=7 MAX_SEQS=64
# 若设置 GROUPED_SO，使用 LEVEL2 §9 构建于本分支的 RGDR 扩展。
export GROUPED_SO="$PWD/experimental/rgdr_v310_ablation/build-c921/librgdr_v310_cube_fp16_torch.so"

run_serve() {
  local route="$1" mode="$2"
  shift 2
  local spec comp grouped=
  local -a route_args=() mode_args=()
  case "$route" in
    native) ;;
    grouped) grouped="${GROUPED_SO:?Set GROUPED_SO to the LEVEL2 RGDR library}" ;;
    replay) route_args=(--additional-config '{"replayssm_sd":true}') ;;
    *) return 2 ;;
  esac
  spec=$(python3 -c 'import json,os; print(json.dumps(dict(method="dflash", model=os.environ["DRAFT"], num_speculative_tokens=int(os.environ["K"]), draft_tensor_parallel_size=2, draft_sample_method="greedy")))')
  if [[ "$mode" == eager ]]; then
    mode_args=(--enforce-eager)
  else
    comp=$(python3 -c 'import json,os; w=int(os.environ["K"])+1; limit=int(os.environ["MAX_SEQS"]); print(json.dumps(dict(cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[w*n for n in (*range(1,11),16,24,28,32,34,36,40,48,56,64) if n<=limit])))')
    mode_args=(--compilation-config "$comp" --cudagraph-metrics)
  fi
  VLLM_ASCEND_310P_GROUPED_RGDR_LIBRARY="$grouped" vllm serve "$TARGET" \
    --host 127.0.0.1 --port 18083 --served-model-name dflash310 \
    --tensor-parallel-size 2 --dtype float16 --quantization ascend \
    --mamba-cache-dtype float16 --mamba-ssm-cache-dtype float16 --mamba-cache-mode none \
    --max-model-len 4096 --max-num-batched-tokens 8192 --max-num-seqs "$MAX_SEQS" \
    --gpu-memory-utilization 0.85 --no-enable-prefix-caching --no-async-scheduling \
    --language-model-only --seed 42 --speculative-config "$spec" \
    "${route_args[@]}" "${mode_args[@]}" "$@" 2>&1 | tee "$RESULTS/$route-k$K-$mode.log"
  return "${PIPESTATUS[0]}"
}

# 各轮使用同一对卡，逐一关闭旧实例后再启动下一轮。
VLLM_ASCEND_310P_DFLASH_STAGE_TRACE=32 run_serve replay eager
# VLLM_ASCEND_310P_DFLASH_STAGE_TRACE=32 run_serve replay graph
# run_serve native graph
# run_serve grouped graph
```

Replay 两个 rank 应显示 `method=dflash logical_width=8 physical_width=8`、
`verify=2/tile64 compact=2/tile32 cores=7`。每个 Mamba group 在
分配日志中的 `speculative_blocks` 应为 0，独立 ring 的字节数应出现在
内存探针中。对照组分别确认原生或 grouped kernel。

## 3. 正确性和生命周期

每轮 `/health` 返回 200 后，在另一终端设置相应的 `ROUTE`、`MODE`、
`RESULTS`、`K`。先 C1，后 C10 持续补入。
Trace 只保留前 N 个调度步。若 C1 已用完 trace 配额，归档日志后
单独重启 Replay 跑 C10，可按需提高 N。

```bash
ROUTE=replay MODE=eager K=7
python3 scripts/replayssm/validate_sd_serve.py --model dflash310 \
  --prompt-file "$RESULTS/short.txt" --concurrency 1 --repetitions 2 \
  --max-tokens 128 --logprobs 5 --output "$RESULTS/$ROUTE-k$K-$MODE-c1.json"
python3 scripts/replayssm/validate_sd_serve.py --model dflash310 \
  --prompts-jsonl "$RESULTS/prompts.jsonl" --concurrency 10 --repetitions 2 \
  --max-tokens 400 --output "$RESULTS/$ROUTE-k$K-$MODE-c10.json"
grep -E '310P DFlash (graph replay|stage .*phase=replayssm_sd)|ReplaySSM SD dispatch|REPLAYSSM_MEMORY_JSON' \
  "$RESULTS/$ROUTE-k$K-$MODE.log"
```

- 两 rank 均须有 8-token target 和 draft replay。C10 检查多请求
  descriptor 及短请求结束后的持续补入。混合、短窗或未配置的
  descriptor 应按 L2 规则走图外。
- `phase=replayssm_sd` 逐层记录 cursor/verify/compact 状态，正常
  行均为 0。首轮或抢占重算后首轮 decode 应有 `reset=1`、`pos=0`、
  `base=0`。同一请求各 group 的 ring ID、接受数、position/base
  应一致，checkpoint ID 按各 group 独立。检查至少一次 ring 跨
  尾，以及短请求释放后被新请求复用的 ring ID。
- K=7 的窗口含 8 token，沿用 `pos + 2 * width > 16` 的 flush
  规则。正常接收后后续窗口常见 flush。采集 `accepted` 和
  `lengths`，覆盖 1、介于 1 和 8 之间、全接收三种情况。
- 在小一档显式 KV 预算下混合 2k/400 和更长输出，观察抢占计数。
  记录实际抢占再恢复的请求和 trace，确认重算后从空 ring 开始。
  若未出现抢占，单列此项，其他生命周期结果分别回传。
- 记录首个 token 分歧和 top logprobs，作为排查材料。验收优先看
  非法状态、重复请求串扰、请求失败、非有限值和设备错误。

## 4. 同预算的 2k/400 容量与性能

关闭 stage trace，按三条路径分别跑 C32/C64。保持图模式和捕获
列表一致。先按 §2 复现 v24 自动定容，再做物理预算下的比较。显式
比较时保持 **KV backing + 独立 ring** 预算一致，以原生每卡显式
预算为基准，Replay 从该预算减去实测 ring 字节数。其余 workspace、
模型加载和捕获内存分别记录。若以自动预算比较，使用两边同一个
`gpu-memory-utilization`，同时记录两边实际分得的物理 KV 字节。

```bash
export VLLM_ASCEND_310P_DFLASH_STAGE_TRACE=0
# 重启每条路线后执行，设置 ROUTE=native、grouped 或 replay。
for c in 32 64; do
  curl -fsS http://127.0.0.1:18083/metrics > "$RESULTS/$ROUTE-k$K-c$c.before.prom"
  vllm bench serve --backend vllm --host 127.0.0.1 --port 18083 \
    --model dflash310 --tokenizer "$TARGET" --endpoint /v1/completions \
    --dataset-name random --random-input-len 2048 --random-output-len 400 \
    --ignore-eos --temperature 0 --seed 42 --num-prompts "$((c * 3))" --max-concurrency "$c" \
    --percentile-metrics ttft,tpot,itl --metric-percentiles 50,90,99 \
    --save-result --result-dir "$RESULTS" --result-filename "$ROUTE-k$K-c$c.json"
  curl -fsS http://127.0.0.1:18083/metrics > "$RESULTS/$ROUTE-k$K-c$c.after.prom"
done
```

记录输出吞吐、TTFT、TPOT/ITL、接受长度、持续和峰值 running、
waiting、preemption 以及图命中。用两 rank 的 `shared KV budget`、
`shared KV allocation` 和 `REPLAYSSM_MEMORY_JSON` 核对 pool block
数、每请求 block 数、draft KV、target checkpoint、独立 ring 和
图池。再记录加载后和服务稳定时的每卡 HBM。典型 30 层/每 rank
HK=8、HV=16 的 ring64 约 369 MiB/rank，随固定容量变化。

在稳定 decode 区间抓同规模 profiling。分别汇报 native
`RecurrentGatedDeltaRuleV310`、grouped RGDR，以及 Replay
`replayssm_sd_310p_verify_cube`、`replayssm_sd_310p_compact_batch`、
`replayssm_sd_310p_cursor` 的时延和次数。比较完整 target forward，
并按实际 group 数计 cursor、按 GDN 层数计 verify/compact。

## 5. 其余宽度

K=7 正确性通过后，逐档跑 K=1–6 的 C1 双图和短窗 smoke。K=1/2/3
的物理宽度为 4，K=4/5 为 6，K=6/7 为 8。宽度 4/8 使用 Cube，
宽度 6 使用 V12 vector。所有档位使用 compact 2，cursor 一律用
真实逻辑宽度 `K+1`。K=1/3/5 加跑多轮接受/拒绝和 C10 补入。
按 [310P ReplaySSM SD Serve](../replayssm_sd_310p_ascendc/LEVEL310P_SERVE.md)
回归 MTP3/7 的 dispatch 和 C1。

回传 commit、build 日志和库哈希、原始服务日志、测试 JSON、metrics
和 profiler 摘要。出错时附首个 CANN 错误、rank、graph descriptor、
请求阶段和相关状态。

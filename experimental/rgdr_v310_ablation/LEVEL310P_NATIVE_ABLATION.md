# Ascend310P3 native GDR: first on-board ablation

## Environment and build

- Branch: `replayssm-310`; use the exact commit shown by `git rev-parse HEAD`.
- Ascend310P3, CANN 9.2.1, torch_npu 2.10.0.post7, vLLM/vllm-ascend 0.30.0.
- Use an idle device. Record `npu-smi info`, package versions, `git status --short`,
  and the first compiler error if a build fails. Do not load an older `.so`.
- This directory is independent from `csrc/` and does not change `vllm serve`.

```bash
cd experimental/rgdr_v310_ablation
rm -rf build
cmake -S . -B build \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.1 \
  -DSOC_VERSION=Ascend310P3 \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build --target rgdr_v310_vec_torch -j 8
readelf -d build/librgdr_v310_vec_torch.so | grep -E 'RUNPATH|RPATH'
PYTHONPATH="$(git rev-parse --show-toplevel)" \
  python3 -m unittest experimental.rgdr_v310_ablation.tests.test_task_mapping
```

The vector binding must load and compile first. If it fails, send the first
CCE/C++ error and stop. Build Cube separately after the vector checks pass:

```bash
cmake --build build --target rgdr_v310_cube_torch -j 8
readelf -d build/librgdr_v310_cube_torch.so | grep -E 'RUNPATH|RPATH'
```

Both torch libraries should have `$ORIGIN/lib` in RUNPATH and find the CANN
generated libraries under `build/lib`.

## Gate A: full vector operator correctness

For each MTP size, run native first, then `vec0`, then the three independent
ablations. The native result is compared with `vec0` using FP16 tolerance;
`vec1`–`vec3` must match `vec0` bitwise for output **and every state slot**.
Each run writes JSON. Replace `3` with `7` for the second set.

```bash
mkdir -p /tmp/rgdr-v310-ablation
python3 bench.py --mtp 3 --batch 32 --variant native \
  --write-golden /tmp/rgdr-v310-ablation/native-mtp3.pt \
  --output /tmp/rgdr-v310-ablation/native-mtp3.json
python3 bench.py --mtp 3 --batch 32 --variant 0 --cores 7 \
  --native-golden /tmp/rgdr-v310-ablation/native-mtp3.pt \
  --write-golden /tmp/rgdr-v310-ablation/vec0-mtp3.pt \
  --output /tmp/rgdr-v310-ablation/vec0-mtp3.json
for variant in 1 2 3; do
  python3 bench.py --mtp 3 --batch 32 --variant "$variant" --cores 7 \
    --golden /tmp/rgdr-v310-ablation/vec0-mtp3.pt \
    --output "/tmp/rgdr-v310-ablation/vec${variant}-mtp3.json"
done
```

Repeat for `--mtp 7`, using `mtp7` in each output/golden filename. A mismatch,
NPU fault, graph-capture failure or timeout stops that MTP set before perf
interpretation. Return the JSON and complete first traceback. The script
always writes JSON on failure.

## Gate B: Cube and vector matvec microprobe

First run smoke with 32 tiles. It compares all Cube columns with a CPU FP32
reference and compares the live first column against the vector path.

```bash
python3 cube_probe.py --mtp 3 --stage smoke --tiles 32 \
  --output /tmp/rgdr-v310-ablation/cube-smoke-mtp3.json
python3 cube_probe.py --mtp 7 --stage smoke --tiles 32 \
  --output /tmp/rgdr-v310-ablation/cube-smoke-mtp7.json
```

If both pass, measure 100 graph replays per path after 20 warmups. Each
`--stage perf` run benchmarks vector and Cube on the same tiles; the JSON
reports each p50/p90 and their ratio. MTP3 and MTP7 use a tile count
proportional to `batch × 16 heads × (MTP+1) × 8` matrix operations.

```bash
python3 cube_probe.py --mtp 3 --batch 32 --stage perf \
  --output /tmp/rgdr-v310-ablation/cube-perf-mtp3.json
python3 cube_probe.py --mtp 7 --batch 32 --stage perf \
  --output /tmp/rgdr-v310-ablation/cube-perf-mtp7.json
```

The Cube probe omits recurrence, gate, beta, state writeback, and the cost
of preparing its 16-column padded RHS. Do not compare its p50 directly to
the full native operator or claim an end-to-end gain from it.

## Profiling and return data

Use the same native/vec0/vec1/vec2/vec3 cases for `msprof` after Gate A,
with MTP3 and MTP7, B32, 7 vector cores. Record task duration, AIV Scalar,
Vector, MTE2/MTE3, launch count and the tiling/core count reported by native.
For the Cube probe, record AIC/AIV and transfer times separately. A quick
repeat with reversed case order helps detect thermal drift. Do not sum engine
durations to explain task duration because they overlap.

Return the ten full-operator JSON files, four Cube JSON files, first failing
compile/runtime logs, `npu-smi info`, and the relevant `msprof` summary.
The full vector variant's results are actionable after correctness passes
and it is faster than native on both MTP sizes. The next section benchmarks
the full Cube operator separately.

## FP16 single-pass Cube RGDR

This experiment covers uniform MTP3 and MTP7 windows on 310P3. Each
`(request, value head, V tile)` retains its FP16 state in UB across tokens.
One `LoadData2D` / `Mmad` path computes two live columns with FP16 L0C.
The extra K block encodes `[Hg | v] × [k,q; −1,0]`. This places `Hgk−v`
and `Hgq` directly in L0C. Vector updates the state and adds
`delta × (k·q)` to the output. Every candidate state is written to GM.

- `cube-fp16` drains the 16×16 fractal once and computes `−beta` scaling in UB.
- `cube-deq` drains the same L0C fractal twice. The ordinary drain supplies
  `Hgq`. The FP16-to-FP16 DEQ drain applies `−beta` during transfer and supplies
  delta. The second drain and transpose are included in its latency.

The target builds independently from the vector and matvec probes.
Follow the environment and CMake configure steps above, then build:

```bash
cmake --build build --target rgdr_v310_cube_fp16_torch -j 8
readelf -d build/librgdr_v310_cube_fp16_torch.so | grep -E 'RUNPATH|RPATH'
```

### This round: profile Native vs Cube DEQ

Compare only the original `npu_recurrent_gated_delta_rule_310` and
`cube-deq`, using B32 and MTP3/7. Run the small Cube route sentinel before
starting profiling. It checks the sign of `−v`, the K/Q columns, DEQ scaling,
and state writeback.

```bash
export RGDR_RESULTS=/tmp/rgdr-v310-cube-deq-profile
mkdir -p "$RGDR_RESULTS"
python3 bench.py --route-smoke-only --mtp 3 --batch 1 --variant cube-deq \
  --device 0 --cores 1 --output "$RGDR_RESULTS/cube-deq-smoke.json"
```

After the sentinel passes, run four separate captures on the same idle
device. `--profile-only` submits the complete operator eagerly, with 20
warmups and 100 measured launches per capture. It restores the same input
state before every launch.

```bash
(
  set -e
  for mtp in 3 7; do
    for variant in native cube-deq; do
      name="$variant-mtp$mtp-b32"
      msprof --output="$RGDR_RESULTS/$name" --ai-core=on --aic-metrics=PipeUtilization \
        python3 bench.py --profile-only --variant "$variant" --mtp "$mtp" --batch 32 \
          --device 0 --cores 7 --fp16-gate-input --warmup 20 --iterations 100 \
          --output "$RGDR_RESULTS/$name.json"
      python3 -c 'import json, sys; sys.exit(json.load(open(sys.argv[1]))["status"] != "PASSED")' \
        "$RGDR_RESULTS/$name.json"
    done
  done
)
```

The inputs share a seed, including Q/K/V, beta, the initial state, state
indices and accepted counts. Native receives the same FP16-rounded log-gate
values in its FP32 input ABI. The Cube task includes Q/K and state packing,
both L0C drains, recurrence and candidate state writes.

In each `op_summary_*.csv`, select the Native V310 or
`rgdr_v310_cube_fp16` task and order matching rows by start time. Expect
120 matching launches and discard the first 20. Use the remaining 100
task durations for p50/p90. Exclude state-reset copies, tensor preparation,
and the post-run finite checks by selecting the RGDR task rows. Record
reported block counts and available Cube, Vector, Scalar and MTE times.
If task counts differ, return the raw trace and the observed kernel names
for inspection. Return the four JSONs and profiler summaries.

## Grouped FP16 Cube RGDR vs the preceding Cube DEQ

`cube-grouped` combines the scalar and transfer changes in one kernel.
A task owns the two value heads for a request's key head. It validates the
candidate slots and computes their state addresses before the token loop.
It stages the whole window, batches the gate and dot operations, and loads
the same RHS into L0B once per token for both heads. Each head uses a full
V=128 tile, one FP16 L0C drain and UB vector coefficients. The two output
heads and their candidate states are each written with one contiguous copy.
All candidate states and the original `−v` augmentation are retained.

Configure a fresh Release build for CANN 9.2.1. This keeps generated M200
objects from the earlier build out of the link. The target includes both the
old DEQ and grouped kernels and the half-negation compiler fix.

```bash
cmake -S . -B build-grouped -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.1 \
  -DSOC_VERSION=Ascend310P3 -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build-grouped --target rgdr_v310_cube_fp16_torch -j 8
export RGDR_GROUPED_LIB="$PWD/build-grouped/librgdr_v310_cube_fp16_torch.so"
export RGDR_GROUPED_RESULTS=/tmp/rgdr-v310-cube-grouped
mkdir -p "$RGDR_GROUPED_RESULTS"
```

First validate both window widths. The route sentinel checks both heads
with different beta values and selects the last input slot as its initial
state. The B32 comparison checks output and every candidate state against
the native FP16 reference at `atol=rtol=2e-3`.

```bash
(
  set -e
  for mtp in 3 7; do
    python3 bench.py --route-smoke-only --variant cube-grouped --mtp "$mtp" \
      --cores 1 --library "$RGDR_GROUPED_LIB" \
      --output "$RGDR_GROUPED_RESULTS/route-mtp$mtp.json"
    python3 bench.py --variant native --mtp "$mtp" --batch 32 --fp16-gate-input \
      --warmup 0 --iterations 1 --write-golden "$RGDR_GROUPED_RESULTS/native-mtp$mtp.pt" \
      --output "$RGDR_GROUPED_RESULTS/native-check-mtp$mtp.json"
    python3 bench.py --variant cube-grouped --mtp "$mtp" --batch 32 --cores 7 \
      --library "$RGDR_GROUPED_LIB" --warmup 0 --iterations 1 \
      --native-golden "$RGDR_GROUPED_RESULTS/native-mtp$mtp.pt" \
      --output "$RGDR_GROUPED_RESULTS/grouped-check-mtp$mtp.json"
  done
)
```

After those checks pass, profile Native, the preceding Cube DEQ, and the
grouped candidate on the same idle device. Use the same FP16-rounded gate
values, seed and 7-core Cube launch in every capture. Select RGDR tasks
and drop the 20 warmups using the same method as the preceding round.

```bash
(
  set -e
  for mtp in 3 7; do
    for variant in native cube-deq cube-grouped; do
      name="$variant-mtp$mtp-b32"
      msprof --output="$RGDR_GROUPED_RESULTS/$name" --ai-core=on --aic-metrics=PipeUtilization \
        python3 bench.py --profile-only --variant "$variant" --mtp "$mtp" --batch 32 \
          --device 0 --cores 7 --fp16-gate-input --warmup 20 --iterations 100 \
          --library "$RGDR_GROUPED_LIB" --output "$RGDR_GROUPED_RESULTS/$name.json"
      python3 -c 'import json, sys; sys.exit(json.load(open(sys.argv[1]))["status"] != "PASSED")' \
        "$RGDR_GROUPED_RESULTS/$name.json"
    done
  done
)
```

Run the gate once for MTP3 and once for MTP7, passing the matching
`op_summary_*.csv` files from the DEQ and grouped captures.

```bash
python3 compare_cube_profiles.py "$BASELINE_CSV" "$GROUPED_CSV"
```

The candidate passes when both MTP sizes show lower p50 and no higher p90
than the reprofiled DEQ baseline. Repeat a near-tie with capture order
reversed. Return both gate results, native timings, and the raw summaries.
Inspect Scalar, MTE2, MTE3 and Cube time alongside task duration.

## Packed grouped RGDR and vLLM Ascend service pilot

The grouped kernel accepts actual packed row lengths of 1–8 tokens and a
separate candidate slot table of width 2–8 for MTP1–7. The previous
acceptance selects from the full slot row, even if the current query
window is shorter. Only current candidates write state. Zero-length
padding rows leave the cache untouched. The Torch ABI assumes the
scheduler assigns distinct writable slots to active requests. It writes
directly to contiguous or page-strided FP16 SSM cache.

From this directory, configure and build a fresh Release extension.

```bash
cmake -S . -B build-packed -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.1 \
  -DSOC_VERSION=Ascend310P3 -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build-packed --target rgdr_v310_cube_fp16_torch -j 8
export RGDR_PACKED_LIB="$PWD/build-packed/librgdr_v310_cube_fp16_torch.so"
export RGDR_PACKED_RESULTS=/tmp/rgdr-v310-packed
mkdir -p "$RGDR_PACKED_RESULTS"
```

### Single-operator matrix

Run route smoke for MTP1–7. It checks the full window, a one-token window
with previous acceptance beyond the current length, and padding with
distinct beta values for the paired heads. It also checks that writes to
a page-strided cache preserve the guard regions. Compare every candidate state
and output to native for B1, B16, B32 and B64. Include mixed lengths at B32.
Native golden and profiler artifacts from `bacade0` are reusable for
uniform B32 MTP3/7 when generated with the same FP16-rounded gate values.

```bash
# Optional. Point at the bacade0 directory containing native-mtp3.pt/7.pt.
export RGDR_NATIVE_HISTORY="${RGDR_NATIVE_HISTORY:-}"
(
  set -e
  for mtp in 1 2 3 4 5 6 7; do
    python3 bench.py --route-smoke-only --variant cube-grouped --mtp "$mtp" \
      --library "$RGDR_PACKED_LIB" --output "$RGDR_PACKED_RESULTS/route-mtp$mtp.json"
    for shape in 1 16 32 64 ragged; do
      batch="$shape"
      options=()
      if [[ "$shape" == ragged ]]; then
        batch=32
        options+=(--ragged)
      fi
      name="mtp$mtp-$shape"
      golden="$RGDR_PACKED_RESULTS/native-$name.pt"
      if [[ "$shape" == 32 && "$mtp" =~ ^[37]$ && -n "$RGDR_NATIVE_HISTORY" &&
            -f "$RGDR_NATIVE_HISTORY/native-mtp$mtp.pt" ]]; then
        golden="$RGDR_NATIVE_HISTORY/native-mtp$mtp.pt"
      else
        python3 bench.py --variant native --mtp "$mtp" --batch "$batch" "${options[@]}" \
          --fp16-gate-input --warmup 0 --iterations 1 --write-golden "$golden" \
          --output "$RGDR_PACKED_RESULTS/native-$name.json"
      fi
      python3 bench.py --variant cube-grouped --mtp "$mtp" --batch "$batch" "${options[@]}" \
        --library "$RGDR_PACKED_LIB" --warmup 0 --iterations 1 --native-golden "$golden" \
        --output "$RGDR_PACKED_RESULTS/grouped-$name.json"
    done
  done
)
```

Use direct `msprof` at B32 for all MTP widths and the B32 ragged cases.
For a batch scaling check, repeat uniform MTP3/7 at B1/16/64 by changing
`batch` and narrowing the loop. Use the historical native CSVs for uniform
B32 MTP3/7 if the device, stack and capture settings match. For the
reported `bacade0` files, set `RGDR_NATIVE_HISTORY=/data/l00972401/rgdr-v310-cube-grouped`.

```bash
(
  set -e
  batch=32
  for mtp in 1 2 3 4 5 6 7; do
    for shape in uniform ragged; do
      options=()
      if [[ "$shape" == ragged ]]; then
        options+=(--ragged)
      fi
      for variant in native cube-grouped; do
        if [[ "$variant" == native && "$shape" == uniform && "$batch" == 32 &&
              "$mtp" =~ ^[37]$ && -n "$RGDR_NATIVE_HISTORY" ]] &&
           compgen -G "$RGDR_NATIVE_HISTORY/native-mtp$mtp-b32/PROF_*/mindstudio_profiler_output/op_summary_*.csv" >/dev/null; then
          continue
        fi
        name="$variant-mtp$mtp-b$batch-$shape"
        msprof --output="$RGDR_PACKED_RESULTS/$name" --ai-core=on --aic-metrics=PipeUtilization \
          python3 bench.py --profile-only --variant "$variant" --mtp "$mtp" --batch "$batch" \
            "${options[@]}" --library "$RGDR_PACKED_LIB" --fp16-gate-input \
            --output "$RGDR_PACKED_RESULTS/$name.json"
        python3 -c 'import json, sys; sys.exit(json.load(open(sys.argv[1]))["status"] != "PASSED")' \
          "$RGDR_PACKED_RESULTS/$name.json"
      done
    done
  done
)
```

Use 100 task durations after dropping 20 warmups. The new Cube task is
`rgdr_v310_cube_grouped_packed_fp16`. Pass the observed Native kernel
name via `--baseline-kernel` to `compare_cube_profiles.py`. For the
uniform B32 MTP3/7 regression check, also compare against the previous
grouped CSVs (1323.1/2534.4 us p50 reported at `bacade0`).

```bash
python3 compare_cube_profiles.py "$NATIVE_CSV" "$PACKED_CSV" \
  --baseline-kernel "$NATIVE_KERNEL" --candidate-kernel rgdr_v310_cube_grouped_packed_fp16
```

### Service performance, ReplaySSM off

Start from the same W8A8 checkpoint, TP2, FP16 state, eager/graph mode and
service limits as the archived Native B32 tests. The example below uses
eager and a service cap of 64. If the archived Native service used a
different cap or capture configuration, reproduce that setting on the new
service. The client load stays B32 for both MTP sizes. Use the archived
Native **service** JSON for the matching MTP and run mode. Single-op timings
are a separate comparison.

```bash
# Run from the repository root in the service terminal.
set -o pipefail
export REPLAYSSM_MODEL=/path/to/Qwen3.6-35B-A3B-W8A8
export ASCEND_RT_VISIBLE_DEVICES=0,1 VLLM_USE_V2_MODEL_RUNNER=0
unset VLLM_ASCEND_REPLAYSSM_310_TRACE_CONFIG
export VLLM_ASCEND_310P_GROUPED_RGDR_LIBRARY="$RGDR_PACKED_LIB"

# Restart for 7 after completing the MTP3 run.
MTP=3
MODE=eager
# Match the archived service's graph settings when using MODE=graph.
if [[ "$MODE" == graph ]]; then
  mode_flags=(--cudagraph-metrics --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,16,32,64,128,192,256]}')
else
  mode_flags=(--enforce-eager)
fi
vllm serve "$REPLAYSSM_MODEL" --host 127.0.0.1 --port 18083 \
  --served-model-name replayssm-310p --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --distributed-executor-backend mp --dtype float16 --mamba-cache-dtype float16 \
  --mamba-ssm-cache-dtype float16 --mamba-cache-mode none --no-enable-prefix-caching \
  --enable-chunked-prefill --no-async-scheduling --language-model-only --seed 42 \
  --max-model-len 4096 --max-num-seqs 64 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 "${mode_flags[@]}" \
  --speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$MTP,\"enforce_eager\":true}" \
  2>&1 | tee "$RGDR_PACKED_RESULTS/grouped-serve-mtp$MTP.log"
```

Confirm both workers log `310P grouped FP16 RGDR enabled` and run one short
request first. Use a short service trace to confirm the packed Cube symbol
runs on speculative steps. The original conv, prefill, non-spec decode,
MTP sampler and checkpoint allocation remain in use. Keep the profiler
outside the benchmark runs.

```bash
# In the client terminal, set MTP to the running service and export both paths.
MTP=3
curl -fsS http://127.0.0.1:18083/health
curl -fsS http://127.0.0.1:18083/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"replayssm-310p","prompt":"Hello","max_tokens":32,"temperature":0}' \
  > "$RGDR_PACKED_RESULTS/grouped-smoke-mtp$MTP.json"
curl -fsS http://127.0.0.1:18083/metrics > "$RGDR_PACKED_RESULTS/grouped-mtp$MTP.before.metrics"
vllm bench serve --backend openai --endpoint /v1/completions --host 127.0.0.1 --port 18083 \
  --model "$REPLAYSSM_MODEL" --served-model-name replayssm-310p --dataset-name random \
  --random-input-len 2048 --random-output-len 400 --random-range-ratio 0 --seed 42 \
  --temperature 0 --ignore-eos --request-rate inf --max-concurrency 32 --num-prompts 64 --num-warmups 4 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --save-result --save-detailed --result-dir "$RGDR_PACKED_RESULTS" \
  --result-filename "grouped-serve-mtp$MTP-b32.json"
curl -fsS http://127.0.0.1:18083/metrics > "$RGDR_PACKED_RESULTS/grouped-mtp$MTP.after.metrics"
```

Return throughput, TTFT/TPOT/ITL p50/p90, successful request count, preemptions
and MTP accepted/draft deltas. Compare each MTP size against its own archived
Native run. If a matching Native service result is unavailable, retain the
new measurements and record the missing baseline. The FP16 recurrence may
change token choices and acceptance, so report those metrics with throughput.

For a separate dispatch trace, add these arguments to the serve command.

```bash
--profiler-config.profiler=torch \
--profiler-config.torch_profiler_dir="$RGDR_PACKED_RESULTS/serve-trace-mtp$MTP"
```

After the short smoke request succeeds, capture one more request between
`/start_profile` and `/stop_profile`. Look for the packed kernel in the
worker NPU traces or exported operator summaries.

```bash
curl -fsS -X POST http://127.0.0.1:18083/start_profile
curl -fsS http://127.0.0.1:18083/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"replayssm-310p","prompt":"Hello","max_tokens":32,"temperature":0}'
curl -fsS -X POST http://127.0.0.1:18083/stop_profile
```

# ReplaySSM SD + DFlash / DFlash2 on 910C

This checklist covers MRv1 with a Qwen GDN target and a separate DFlash draft.
Use `method=dflash` for both draft architectures. vLLM Ascend selects the
DFlash2 proposer from `DFlash2DraftModel` in the draft config.
`num_speculative_tokens=K` selects 1–7 draft tokens. The target verify
window has up to `K+1` tokens, including the anchor. The checkpoint is BF16,
ring D/K are FP16, and ring G is FP32. The mixed library is compiled with
physical width 8 and uses packed token offsets for the logical window.

| Target | Draft | Rank geometry | Blue2 logical dies |
|---|---|---|---|
| Qwen3.5-9B | Qwen3.5-9B-DFlash | TP1, 16 key / 32 value heads | 13 |
| Qwen3.8-27B | Qwen3.8-27B-DFlash2 | TP2, 8 key / 24 value heads | 11, 15 |

## Build and serve

Run in the CANN 9.2.1 environment after importing the vLLM Ascend checkout
and its compiled extension. On Blue2 use `SOC_VERSION=ascend910_9362`.

```bash
cmake -S experimental/replayssm_cube_910b/mixed_verify -B /tmp/replayssm-dflash-mix \
  -DSOC_VERSION=ascend910_9362 \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.1 \
  -DREPLAYSSM_VERIFY_MAX_TOKENS=8
cmake --build /tmp/replayssm-dflash-mix -j 8
MIXED=/tmp/replayssm-dflash-mix/libreplayssm_sd_mix_torch.so
```

Set `TARGET` and `DRAFT` to the complete local model paths. Set `TP` and
`ASCEND_RT_VISIBLE_DEVICES` together with `MAX_SEQS=64` for 9B and
`MAX_SEQS=32` for 27B. Use the same serve flags and same die(s) for both paths.
Set `EXTRA='{}'` for Native. For Replay use the additional config shown below.
Use a separate `VLLM_CACHE_ROOT` for each source version, mode and K.

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
K=7
WINDOW=$((K+1))
SIZES="$WINDOW,$((WINDOW*4)),$((WINDOW*8)),$((WINDOW*16)),$((WINDOW*MAX_SEQS))"
SPEC=$(printf '{"method":"dflash","model":"%s","num_speculative_tokens":%d}' "$DRAFT" "$K")
GRAPH=$(printf '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[%s]}' "$SIZES")
EXTRA=$(printf '{"replayssm_sd":true,"replayssm_sd_mixed_library":"%s"}' "$MIXED")

vllm serve "$TARGET" --served-model-name dflash-test --host 127.0.0.1 --port 18145 \
  --language-model-only --tensor-parallel-size "$TP" --dtype bfloat16 \
  --max-model-len 3072 --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens 3072 \
  --gpu-memory-utilization 0.85 --no-enable-prefix-caching \
  --enable-chunked-prefill --no-async-scheduling \
  --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype bfloat16 --mamba-cache-mode none \
  --speculative-config "$SPEC" --compilation-config "$GRAPH" --cudagraph-metrics \
  --additional-config "$EXTRA"
```

Verify capture and FULL replay for both target and draft using the engine's
graph log. Confirm the Replay dispatch log and check proposed and accepted
draft-token counts after traffic. Record source commit, library hashes, exact
target and draft IDs, TP, logical dies, concurrency, and the dataset hash.

## Benchmark matrix

For each K 1–7 and mode, run greedy requests against an idle server. Use the
same ordered prompt file in both modes. Use C1, C8 and C32 for both targets
and C64 for 9B. Measure steady output rate with fixed output length. Measure
acceptance separately on GSM8K math, general text, prompt-only JSON and
schema-constrained JSON. Speculative acceptance is accepted draft tokens
divided by proposed draft tokens. An absent denominator is recorded as null.
Per-position rates are emitted when every counted draft proposed all K tokens.

```bash
python3 scripts/replayssm/benchmark_dflash_serve.py --url http://127.0.0.1:18145 \
  --model dflash-test --dataset perf.jsonl --draft-tokens "$K" \
  --concurrency 32 --requests 32 --max-tokens 160 --ignore-eos --output perf-c32.json
python3 scripts/replayssm/benchmark_dflash_serve.py --url http://127.0.0.1:18145 \
  --model dflash-test --dataset math.jsonl --draft-tokens "$K" \
  --concurrency 8 --requests 32 --max-tokens 320 --output math-c8.json
```

Each JSONL row has a `prompt`. Set `expect_json` to count parseable output.
Supply a `json_schema` object to request server-side structured generation.
Save the reports, Prometheus snapshots, and the service log together.

## Blue2 results

Blue2 uses Ascend910_9362, CANN 9.2.1, torch_npu 2.10.0.post7 and vLLM
`0.30.0+empty`. Earlier 27B TP1 DFlash2 Native K7 PIECEWISE emitted
96 tokens and accepted 63 of 238 drafts. That smoke used one request
and the artifact is `dflash2-die11.gpn0Dl/graph.CdOg6Z`.

The first TP2 FULL Native attempt at source `e2a0931f` failed in FIA workspace
tiling. The captured DFlash2 metadata had no mask and the attention layer
declared sliding-window configuration. This chose sparse mode 4 even though
the query was noncausal. On this branch, capture and eager use dense mode 0
for noncausal metadata. The failure log is `dflash2-native-full-tp2.P0cOXn`.

### 27B TP2, DFlash2 K7

Native at `ee7b1ce` and Replay at `6b02f37` used dies 11 and 15,
both with BF16, 3072 model and per-step token limits, a 32-request limit,
and FULL capture sizes 8/32/64/128/256. Replay used the default 24-entry
logical threshold and capacity 32, with physical ring width 32 and mixed
verify window width 8. The mixed Torch library SHA-256 was
`a173fd018d47ce8a10f7d202eee0aff89ab3c234c4bb3514070155d51dc80031`.
The Replay log contains independent FULL replay messages for target and
draft. Native logs contain the draft FULL wrapper, capture completion and
FULL runtime graph hits. All fixed-length requests completed.

| Client concurrency / requests | Native tok/s | Replay tok/s | Native TPOT p50 (ms) | Replay TPOT p50 (ms) |
|---|---:|---:|---:|---:|
| 1 / 2 | 51.41 | 43.63 | 18.01 | 21.35 |
| 8 / 16 | 129.21 | 237.23 | 36.13 | 28.11 |
| 32 / 32 | 546.73 | 607.86 | 49.67 | 43.99 |

C1 used 128 output tokens per request. C8 and C32 used 160. Each cell is
one run. C8 TTFT p50 also changed from 2.48 s to 0.41 s, so the large
throughput delta needs a same-commit repeat before attribution. Native and
Replay reported 116,307 and 249,977 tokens of KV capacity. No per-row flush
trace was collected in these service runs.

| Acceptance workload (C8) | Native accepted / proposed | Replay accepted / proposed | Passed Native / Replay |
|---|---:|---:|---:|
| GSM8K, 32 requests | 8190/11116 (73.68%) | 8183/11053 (74.03%) | 32/32, 32/32 |
| General text, 32 requests | 3864/16065 (24.05%) | 3861/15995 (24.14%) | 32/32, 32/32 |
| Prompt-only JSON, 16 requests | 762/1743 (43.72%) | 765/1778 (43.03%) | 16/16, 16/16 |

Both paths returned parseable JSON for 16/16 prompt-only JSON requests.
The Native schema-constrained test hit
`aclnnRecurrentGatedDeltaRuleGetWorkspaceSize: 561103` after requests entered
the engine. All 16 requests failed, so its partial counters are excluded.
The Replay schema-constrained test is pending. Artifact directories are
`dflash-27b-native-k7.BKCgn8` and `dflash-27b-replay-k7.8KokLw` on Blue2.

### 27B TP2, DFlash2 K4

Both modes at `6b02f37` passed the same six performance and acceptance
cases and independently logged FULL target and draft replay. The graph
shapes were 5/20/40/80/160 and the Replay logical threshold was 21. The
physical ring and mixed library were the same as K7. No per-row flush
trace was collected. One run per mode.

| Concurrency | Native tok/s | Replay tok/s | Native TPOT p50 (ms) | Replay TPOT p50 (ms) |
|---|---:|---:|---:|---:|
| 1 | 47.53 | 43.22 | 19.52 | 21.55 |
| 8 | 238.39 | 236.96 | 30.03 | 29.69 |
| 32 | 617.73 | 631.76 | 43.81 | 43.48 |

At C8, Native / Replay acceptance was 85.23% / 84.74% on GSM8K,
38.37% / 38.71% on general text and 54.48% / 55.16% on prompt-only
JSON. Both modes returned parseable JSON for all 16 samples. Artifacts
are `dflash-27b-native-k4.nq1DX5` and `dflash-27b-replay-k4.Z0Z9ZN`.

### 27B TP2, DFlash2 K6

Both modes at `6b02f37` passed the same six cases and independently
logged FULL target and draft replay. Graph shapes were 7/28/56/112/224
and the Replay logical threshold was 23. No per-row flush trace was
collected. One run per mode.

| Concurrency | Native tok/s | Replay tok/s | Native TPOT p50 (ms) | Replay TPOT p50 (ms) |
|---|---:|---:|---:|---:|
| 1 | 49.40 | 44.91 | 18.77 | 20.70 |
| 8 | 236.02 | 233.67 | 29.18 | 29.84 |
| 32 | 564.40 | 575.12 | 48.85 | 44.66 |

At C8, Native / Replay acceptance was 77.66% / 77.76% on GSM8K,
27.15% / 27.09% on text and 44.18% / 43.41% on prompt-only JSON.
Both modes returned 16/16 parseable JSON responses. Artifacts are
`dflash-27b-native-k6.OWGdG1` and `dflash-27b-replay-k6.jlJGBl`.

### 9B TP1 mixed attention preflight

The 9B draft has five sliding layers with a 4096-token window and one full
attention layer. Its verified weight SHA-256 is
`0a42274b32554f48de1faa0d42824e9c2ceda649c30ae0a731cddf410dd698c7`.
The first Replay MRv1 K7 attempt at `ee7b1ce` stopped at the upstream
mixed sliding/full capability check before loading weights. The MRv2
Native K7 health and two C1 requests later passed with 156/714 accepted
draft tokens. It captured graphs, but the log lacks independent target
and draft replay markers. These runs are
`dflash-9b-replay-k7.bMb2We` and `dflash-9b-native-k7.NNt9Ns` on Blue2.
The subsequent MRv1 work keeps the model's original layer attention
modes and routes each draft KV group to its own page table and slot maps.
At `eb799cb`, weights loaded and the draft FULL wrapper initialized.
Memory profiling failed because it clears dummy slot mappings before
the per-group buffers are assigned. `c0f10e4` fixes that initialization
order. The failed profile report is `dflash-9b-replay-k7.41b8vD`.

At `c0f10e4`, both MRv1 modes passed K7 capture and logged independent
target/draft FULL replay at C1. The two C1 requests passed per mode.
Draft acceptance was 23/1617 (1.42%) Native and 38/1512 (2.51%) Replay,
versus 156/714 (21.85%) for the MRv2 Native smoke on the same two prompts.
Both MRv1 paths failed C8 with device error 507011. The CANN debug logs
identify `ScatterPaKvCache_e1075f1c4d344a306bb205d49ce88d2b_1001` in both.
The C1 smoke is insufficient evidence of correct draft cache addressing.
Artifacts are `dflash-9b-native-k7.ojFKVi` and
`dflash-9b-replay-k7.UhgQ6m`. A separate diagnostic checks the per-group
slot mappings and active cache layout without changing the checkpoint.

The Replay slot diagnostic `dflash-9b-replay-k7.5zWf7V` reproduced the
C8 scatter failure. Logged live slots were within bounds, with an
interleaved full-attention cache and dense sliding-layer cache views.
The separate `horizon-die13.Qozz2D/probe-dflash-scatter` probe used
64 BF16 token rows and 1,024 pages on die 13. It matched written K and V
exactly in both eager and graph replay for contiguous and interleaved
views. These checks exercised the allocator stride but not server graph
input lifetimes. In `c0f10e4`, dummy capture used the original context
mapping tensor and the first request rebound it to a per-layer list.
`84d8ae3` binds the list before graph capture and masks the unused
context slots at replay. The first retry, `dflash-9b-replay-k7.cJlvH4`,
stopped before ACLGraph capture in a reused npugraph_ex AOT cache with
an argument arity mismatch. The clean-cache retry uses a cache root
specific to source version, mode and K.

### 9B TP1, DFlash K7 after binding context slots before capture

Replay at `84d8ae3` used die 13, max-seqs 64, FULL graph sizes
8/32/64/128/512, the same mixed library and a 24-entry logical threshold.
The run independently logged target and draft FULL replay. The C8
scatter failure did not recur in the suite. C1 acceptance on the
same two requests recovered to 152/749 (20.29%). No per-row flush
trace was collected. The MRv1 Native rerun uses the same code and
serve settings on the same die.

| Concurrency | Passed | Replay tok/s | Replay TPOT p50 (ms) |
|---|---:|---:|---:|
| 1 | 2/2 | 32.95 | 29.35 |
| 8 | 16/16 | 220.52 | 33.66 |
| 32 | 32/32 | 499.97 | 57.75 |
| 64 | 64/64 | 859.04 | 66.04 |

At C8, Replay accepted 8172/11795 (69.28%) for 32 GSM8K requests,
3585/17920 (20.01%) for 32 general-text requests and 743/1974 (37.64%)
for 16 prompt-only JSON requests. All requests completed and all
16 JSON outputs parsed. Schema-constrained output remains a separate
test. The verified Replay artifact is `dflash-9b-replay-k7.El0FDU`.

### 9B TP1 paired results for K4 and K7

The 9B pairs used `84d8ae3`, die 13, max-seqs 64 and the same mixed
library and launch settings as the K7 Replay run above. K4 used FULL
sizes 5/20/40/80/320 and a 21-entry logical threshold. Both paths
independently logged target and draft FULL replay at each K, with all
fixed-length C1/C8/C32/C64 requests completed. One run per cell.

| K | Concurrency | Native tok/s | Replay tok/s | Native TPOT p50 (ms) | Replay TPOT p50 (ms) |
|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 41.87 | 31.87 | 22.92 | 30.37 |
| 4 | 8 | 260.79 | 218.28 | 28.17 | 33.41 |
| 4 | 32 | 611.92 | 553.63 | 47.22 | 52.03 |
| 4 | 64 | 1053.08 | 970.12 | 53.98 | 58.02 |
| 7 | 1 | 44.57 | 32.95 | 21.46 | 29.35 |
| 7 | 8 | 264.11 | 220.52 | 27.96 | 33.66 |
| 7 | 32 | 526.34 | 499.97 | 54.95 | 57.75 |
| 7 | 64 | 870.17 | 859.04 | 64.76 | 66.04 |

| K | C8 dataset | Native accepted / proposed | Replay accepted / proposed |
|---:|---|---:|---:|
| 4 | GSM8K, 32 requests | 7509/9260 (81.09%) | 7530/9272 (81.21%) |
| 4 | General text, 32 requests | 3489/10632 (32.82%) | 3500/10584 (33.07%) |
| 4 | Prompt-only JSON, 16 requests | 691/1340 (51.57%) | 688/1332 (51.65%) |
| 7 | GSM8K, 32 requests | 8157/11998 (67.99%) | 8172/11795 (69.28%) |
| 7 | General text, 32 requests | 3566/18081 (19.72%) | 3585/17920 (20.01%) |
| 7 | Prompt-only JSON, 16 requests | 743/1974 (37.64%) | 743/1974 (37.64%) |

All acceptance requests completed. All 16 JSON responses parsed in each
cell. At C64 the K7 throughput gap was 1.3%. Absolute throughput peaked
at K4 among these two measured windows, in both modes. No flush trace was
collected. Artifacts are `dflash-9b-native-k4.FTbves`,
`dflash-9b-replay-k4.0S5PHa`, `dflash-9b-native-k7.oSaKte`,
and `dflash-9b-replay-k7.El0FDU`.

### 27B TP2, DFlash2 K1

Both modes at `6b02f37` used dies 11 and 15, max-seqs 32 and FULL sizes
2/8/16/32/64. Replay used the same mixed library as K7 and an 18-entry
logical threshold. Both passed all six cases and independently logged
target and draft FULL replay. All fixed-length requests completed.

| Concurrency | Native tok/s | Replay tok/s | Native TPOT p50 (ms) | Replay TPOT p50 (ms) |
|---:|---:|---:|---:|---:|
| 1 | 33.24 | 19.35 | 28.71 | 41.81 |
| 8 | 203.47 | 154.94 | 36.37 | 41.45 |
| 32 | 615.97 | 579.30 | 46.66 | 50.09 |

At C8 the accepted / proposed token counts in Native and Replay were
4725/4904 and 4696/4872 on GSM8K, 2572/3554 and 2555/3570 on general
text, and 455/553 and 456/552 on prompt-only JSON. All outputs completed
and 16/16 JSON responses parsed in both modes. Compared with K4 and K6,
Replay at K1 lost throughput at C32 for this prompt suite. No per-row
flush trace was collected. Artifacts are `dflash-27b-native-k1.rM6O1W`
and `dflash-27b-replay-k1.3pe2dj`.

### Native ragged state-slot diagnostic

The native GDR ABI expects a flat state index array of length T, the
physical token count. It uses the previous accepted count to select
the input state within each request's index row. A structured request
can shorten its current query below that previous accepted count.

The isolated die 10 probe
`horizon-die13.Qozz2D/probe-dflash-native-ragged/probe.log` reproduced
561103 on CANN 9.2.1 with packed lengths [1, 4] and two eight-slot
rows. Padding each query to its physical width passed. The valid
outputs and states matched the one-token recurrence reference with
previous accepted counts [3, 2], including the row with only one
current token. `57c388c` applies this only when Native GDR sees a
packed query shorter than its state index table. It retains decode rows
in the speculative state layout even if filtering removes all drafts.
The dense path still calls the same native op directly.

The original expanded mask sizing regressed a CPU graph-padding test.
`0a6771a` keeps the existing mask sizing for other speculative modes.
The two GDN metadata and mixed-batch modules then passed 80/80 CPU
tests in the Blue2 environment, including the shortened DFlash query
and anchor-only decode cases.

9B TP1 Native at `57c388c` on die 10 passed strict JSON Schema over
`vllm serve`. Settings matched the K7 matrix, with the separate device
and mode-specific compile cache. A JSON Schema validator independently
checked every response. C1 passed 4/4, accepting 184/420 proposed
tokens (43.81%). C8 passed 16/16, accepting 731/1642 (44.52%).
The ordinary C1 warmup logged actual FULL replay for both target and
draft. This runtime check establishes the shape fix for the 9B Native
K7 case. It does not measure graph coverage specifically during
structured generation. Artifact `dflash-9b-native-k7.TDuNWs`.

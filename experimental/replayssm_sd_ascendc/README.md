# L3.1: independent AscendC GDN speculative verify pilot

This directory retains an independent build and `torch.ops.replayssm_sd_lab.verify_` entry for diagnosis. After the L3.1 device validation, its kernel and C++ launcher are shared with the vLLM Ascend extension, which registers `_C_ascend.replayssm_sd_verify_`. The target is Ascend910B4, CANN 9.2.0, BF16 checkpoint/model tensors, FP16 d/k ring, FP32 g ring, TP4/TP8 local head geometry. Neither entry is connected to MTP serving until flush and cursor lifecycle are implemented.

## Scope and math

The first gate is **non-flush** target verification with `T=1..4` and at most 16 committed records. Inputs `q/k/v` are `[B,4,H or HV,128]`, with preprocessed `g/beta` `[B,4,HV]`; `lengths[B]` masks inactive tokens. The checkpoint is `[slots,HV,128,128]`, and the physical d/k/g ring has 16/32/64 positions. `positions[B]` counts valid committed records and `bases[B]` gives the physical circular origin. Only candidate d/k/g entries and outputs are written; the checkpoint is read-only. The caller must guarantee unique positive slots in one launch. Slot 0 is padding.

Each `(request, value head, V tile)` task loads one BF16 checkpoint tile. For every candidate it computes `S_before @ q` and `S_before @ k` from the checkpoint plus previous committed and draft correction records, then obtains the next correction and output. Earlier draft corrections remain in UB. This evaluates the GDN triangular dependency in token order, without constructing or writing an intermediate `V×K` state per candidate. It is a correctness/performance probe, **not** yet a tuned parallel triangular solve. Unlike the slow AR1 kernel it does not replay full-matrix outer products for every draft token. The following gate will implement flush, and the next gate will implement accepted-token cursor commit/reset before model dispatch.

The exact numerical contract is compared with an independent FP64 CPU dense recurrence. Previous ring records are read at their stored FP16/FP32 precision; within the current verification window, the just-computed draft correction remains FP32 until all outputs have been computed. The Python test also compares a separate output-only formula against the dense recurrence.

The kernel has no torch allocations or host readback. It accepts shared page-backed state views with compact inner dimensions and runtime slot strides, and runs on the current NPU stream. The PyTorch binding validates static shape/dtype/stride metadata. Device-side status codes are 0 success/padding, 1 invalid slot, 2 invalid position/length or flush-required window, 3 invalid ring base, 4 invalid padding metadata. Invalid rows write zero output and leave state untouched.

## Tiling and first performance question

Variants 0–2 distribute AIV-only tasks over `B×HV×(128/tile_v)`. Variants 3–12 assign one task to `(request, value head)` and process every V tile inside that task, sharing the K-side dot table. `tile_v` may be 16/32/64; variant 12 alone requires tile128. K is fixed at 128. The 910B4 hardware inventory supplied by the user reports **192 KiB Vector UB**. The original `tile_v=32` kernel requests 60,832 bytes of explicit UB buffers; that figure was the *allocation*, not the hardware limit. The 64-row tile and the two-entry checkpoint queue fit by source-level arithmetic, but the CANN compiler's resource report is authoritative. The host defaults to **40 vector cores**, capped at the variant's task count. At TP8/B1 the shared-head variants expose only six tasks, so compare B1 and B32/64 separately. `T=4`, history 12/16 is the key throughput probe; `T=1` is the draftless fallback.

The selectable ablations are: 0 validated baseline; 1 K=128 add-fold reduction; 2 plus bulk gate/beta loads; 3 plus per-head dot reuse across V tiles; 4 plus a two-entry BF16 checkpoint prefetch queue; 5 v3 with batched dot-row reduction; 6 v5 with the checkpoint prefetch moved before dot-table computation; 7 v5 with UB-resident history dots and batched weighted history reduction; 8 v7 with speculative correction in UB; 9 v7 with bulk ring K/D loads; 10 v7 with `MulAddDst` speculative accumulation; 11 v9 with multi-row speculative Q/K/V loads; 12 v9 with one tile128 V pass and a reduced UB input queue. Variants 0–10 passed the user's CANN 9.2.0 correctness/graph checks on TP4; v9 is the best measured B32/B64 variant. Variants 11–12 await device verification. All use the same binding with an optional final `variant` argument and leave MTP dispatch unchanged. See `docs/replayssm/checklists/LEVEL3_SD_V12_TILE128.md` for the tile128 validation stage.

Variant 7 keeps the history dot table in UB and replaces per-record history `Accumulate` calls with a 16-row weighted reduction. It retains scalar gate/candidate handling and the short draft dependency loop. The user's TP4 CANN 9.2.0 tests passed precision and graph replay; the compiler's full resource report remains unavailable. At `tile_v=64`, the explicit UB request rises from 104,096 bytes in v5 to 112,992 bytes in v7 (excluding compiler temporaries). Variants 8–10 were independent experiments on v7; v9's bulk history K/D transfers won on B32/B64, while v8 regressed and v10 was neutral. Variant 11 preserves v9 arithmetic and adds a `4×tile_v×FP32` UB buffer for candidate V rows (1,024 bytes at tile64). Variant 12 preserves v9's candidate load path and arithmetic, splits the checkpoint input into four 8 KiB copies, and uses two disjoint regions of the product buffer for history correction. Its source-level explicit UB request is 169,056 B at tile128 versus 210,016 B without these savings; neither figure includes compiler temporaries.

## Local and on-machine checks

Local CPU reference tests:

```bash
.venv/bin/python -m unittest discover -s experimental/replayssm_sd_ascendc/tests -v
```

In the CANN 9.2.0 container, build only this isolated directory (substitute the server's active Python when needed):

```bash
cmake -S experimental/replayssm_sd_ascendc \
  -B experimental/replayssm_sd_ascendc/build \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.0 \
  -DSOC_VERSION=Ascend910B4 \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build experimental/replayssm_sd_ascendc/build --parallel 4
```

The binding is `build/libreplayssm_sd_torch.so`, and CANN places its kernel in `build/lib/`. RPATH includes `$ORIGIN/lib`, as required by the validated AR build layout.

After build, first run one smoke and one graph case on an idle card:

```bash
python3 experimental/replayssm_sd_ascendc/run.py --stage smoke \
  --tp 4 --batch 2 --cores 40 --device 0 \
  --output /tmp/replayssm-sd-l31-smoke-tp4.json
python3 experimental/replayssm_sd_ascendc/run.py --stage graph \
  --tp 4 --batch 2 --cores 40 --device 0 \
  --output /tmp/replayssm-sd-l31-graph-tp4.json
```

Once both pass, compare 8/20/40 cores at TP4 and TP8, and batches 1/32/64. Run each invocation separately or assign independent invocations to idle cards; a device trap should not be followed by more launches on that card until its status is checked. For example:

```bash
for tp in 4 8; do
  for batch in 1 32 64; do
    for cores in 8 20 40; do
      python3 experimental/replayssm_sd_ascendc/run.py --stage perf \
        --tp "$tp" --batch "$batch" --cores "$cores" --device 0 \
        --output "/tmp/replayssm-sd-l31-tp${tp}-b${batch}-c${cores}.json"
    done
  done
done
```

The `smoke` report covers isolated correctness, `graph` covers capture/replay correctness, and `perf` adds graph replay p50/p90. Time only graph replay; cache reset is outside the timed event interval. The first run includes JIT compilation and is not a latency measurement. This stage does **not** establish the eventual SD speedup because flush, cursor operations, model scheduling, and snapshot removal are not yet included.

Return the first failing JSON/log if any, plus the build log, source commit, installed CANN/torch_npu versions, and device health. If smoke or graph fails, stop this level and fix it before running the large performance matrix. No `vllm serve` switch should point to this pilot yet.

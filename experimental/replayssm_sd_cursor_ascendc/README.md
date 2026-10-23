# ReplaySSM SD accepted-prefix cursor pilot

This is the isolated AscendC implementation of the **block-keyed metadata** update that follows an SD verify and optional flush. It does not mutate checkpoint or d/k/g ring data. The order is `verify -> flush using old cursor -> accepted-prefix cursor update`; the kernel combines the original separate commit and first-decode reset launches by letting reset override stale acceptance on each reset row. It is not wired to `vllm serve` yet.

Inputs are `write_pos[num_slots]` and `cache_base[num_slots]` (int32), `is_flush[num_slots]` (int8), `accepted[B]` and `slots[B]` (int32), `do_reset[B]` (int8), and preallocated `status[B]` (int32). Slot 0 is padding. Positive physical slots must be unique within a launch; the host cannot check tensor values without synchronizing. Status codes: 0 success/padding, 1 invalid slot, 2 invalid acceptance, 3 invalid old cursor, 4 logical window overflow. Invalid rows leave cursor state unchanged. A zero accepted count is a no-op. For valid reset rows, reset wins even if accepted or previous cursor is stale; slot 0 remains untouched. The normal MTP path reports at least one accepted token because it includes the bonus token.

The update matches original ReplaySSM's GDN cursor rule. With logical threshold `L`, maximum speculative length `T`, and physical power-of-two ring `R`: if the **old** `is_flush` is true and `accepted > 0`, `new_base = (old_base + old_write_pos) & (R-1)` and `new_write_pos = accepted`; otherwise `new_base = old_base` and `new_write_pos = old_write_pos + accepted`. Then `new_is_flush = (new_write_pos + 2*T > L)`. A reset writes `(0,0,2*T>L)`. The initial target geometry is `L=20, T=4, R=32` for an accepted-history budget of 16.

The first implementation uses one AIV core and 32-byte UB staging for every scalar metadata read/write. This avoids cross-core writes to adjacent words of a GM cache line. It deliberately favors correctness over throughput. The standalone graph-replay `perf` stage must establish whether its B32/B64 latency is material before parallelizing; hardware compile and performance are unverified locally.

## Build and on-machine validation

```bash
cmake -S experimental/replayssm_sd_cursor_ascendc \
  -B experimental/replayssm_sd_cursor_ascendc/build \
  -DSOC_VERSION=Ascend910B4 \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.0 \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build experimental/replayssm_sd_cursor_ascendc/build -j 8
readelf -d experimental/replayssm_sd_cursor_ascendc/build/libreplayssm_sd_cursor_torch.so | grep RUNPATH
python3 experimental/replayssm_sd_cursor_ascendc/run.py \
  --stage smoke --batch 1 --output /tmp/replayssm-sd-cursor-smoke-b1.json
python3 experimental/replayssm_sd_cursor_ascendc/run.py \
  --stage graph --batch 32 --output /tmp/replayssm-sd-cursor-graph-b32.json
python3 experimental/replayssm_sd_cursor_ascendc/run.py \
  --stage perf --batch 32 --iterations 50 --output /tmp/replayssm-sd-cursor-perf-b32.json
python3 experimental/replayssm_sd_cursor_ascendc/run.py \
  --stage perf --batch 64 --iterations 50 --output /tmp/replayssm-sd-cursor-perf-b64.json
```

Check all JSON `status=PASSED`, each case's exact int32/int8 cursor values and graph replay behavior, and `RUNPATH` containing `$ORIGIN/lib`. Compare B1/B32/B64 p50/p90 with the SD verify and flush costs. The timing excludes CPU-side reset and `torch.npu.Event` setup. Report source commit, CANN/torch_npu versions, build log, first failing JSON/log, and NPU health. Stop at the first compile, precision, or graph failure; do not enable MTP serving from these isolated results.

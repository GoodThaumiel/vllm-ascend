# ReplaySSM SD committed-history compaction pilot

This independent AscendC target tests the flush checkpoint update on Ascend910B4 and CANN 9.2.0. It is separate from speculative GDN dispatch. The operator takes a BF16 checkpoint, FP16 d/k ring, FP32 g ring, block-keyed metadata and a per-row flush flag. It modifies only checkpoint and status.

The required order on one NPU stream is **verify → compact → accepted-token cursor commit**. Verify writes outputs and draft d/k/g records at logical positions [position, position+T). Compact folds only the previously committed [0, position) records into BF16 checkpoint when flush=1. Cursor commit then moves the logical base by the old position and sets write_pos=accepted. Unaccepted draft records remain allocated but become invisible. Ring length 32 accommodates the original L=20 logical window (B=16, T=4). This pilot does not choose accepted tokens or update cursors.

Each AIV task owns (row, value head, V tile) and reads a BF16 tile_v×128 checkpoint tile. It sequentially applies each committed S ← exp(g_i) S + d_i k_iᵀ update in FP32 and casts once to BF16. At tile_v=32, explicit UB allocations total roughly 49 KiB. There is no inter-core communication. Forty requested cores are capped at the task count. Flush=0 and position=0 skip checkpoint traffic. Active slots must be unique.

The implementation follows the existing AR AscendC scalar staging, vector outer-add and page-stride conventions. It is a correctness/performance pilot; separate compaction adds a launch and needs profiling before serving integration.

## On-machine checklist

Build in the CANN 9.2.0 container using the same Python and torch_npu:

```bash
cmake -S experimental/replayssm_sd_compact_ascendc -B experimental/replayssm_sd_compact_ascendc/build \
  -DSOC_VERSION=Ascend910B4 -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.0 \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build experimental/replayssm_sd_compact_ascendc/build -j 8
readelf -d experimental/replayssm_sd_compact_ascendc/build/libreplayssm_sd_compact_torch.so | grep RUNPATH
```

- [ ] RUNPATH includes $ORIGIN/lib and the kernel dependency resolves at load.
- [ ] CPU reference tests pass: python3 -m unittest experimental.replayssm_sd_compact_ascendc.tests.test_reference.
- [ ] TP4 B2 smoke, then graph, then TP8 graph all pass:

```bash
python3 experimental/replayssm_sd_compact_ascendc/run.py --stage smoke --tp 4 --batch 2 \
  --cores 40 --output /tmp/replayssm-sd-compact-tp4-smoke.json
python3 experimental/replayssm_sd_compact_ascendc/run.py --stage graph --tp 4 --batch 2 \
  --cores 40 --output /tmp/replayssm-sd-compact-tp4-graph.json
python3 experimental/replayssm_sd_compact_ascendc/run.py --stage graph --tp 8 --batch 2 \
  --cores 40 --output /tmp/replayssm-sd-compact-tp8-graph.json
```

- [ ] Graph cases cover position 0/13/16, base 31 wrap, no-flush and mixed padding rows; compare checkpoint and all ring buffers.
- [ ] Position-16/T4 reads only 16 old records. Candidate positions 16–19 remain untouched by compaction.
- [ ] After graph passes, run --stage perf at B1 then B32/64, TP4/8 and cores 8/20/40. Report graph p50/p90 and compare with verify-only at the same geometry; combined pipeline timing is a separate test.
- [ ] On any build, precision, graph or device-health error, stop and return the first failing JSON, build/device logs and source commit. Do not enable speculative serving yet.

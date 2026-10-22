# Replay AR profiling and cache-memory probe (2026-10-22)

## Source baseline and layout boundary

- Official vLLM `v0.30.0rc1`: `a00a3544b93edbd66c8eda7285e4468f1202dc4b`.
- Official vLLM Ascend `v0.30.0rc1`: `5f828ad638ee4f9b12db87f9d5481f29076d37f9`.
- The rc1 Dockerfile's default FLA NPU source tag is `v26.9.1-beta2` (`a4a795857868729fbdbb0118c1e06c8655a7b31b`); verify the baseline image's installed wheel before treating this as its exact source.
- Development remains vLLM `0.30.0+empty`, vLLM Ascend `releases/v0.30.0`, CANN 9.2.0. The rc1 trees are read-only comparisons.

[PR #14340](https://github.com/vllm-project/vllm-ascend/pull/14340) introduced page-strided hybrid Attention/Mamba cache views and optimized GDN gather/scatter. The old advanced-indexing ablation in that PR changed TPOT from 36.0 to 87.8 ms under its own workload. [PR #17924](https://github.com/vllm-project/vllm-ascend/pull/17924) restored contiguous K/V allocation only for eligible ordinary Paged Attention layers; hybrid backing remained non-contiguous. Both changes are represented in the rc1 source. [PR #17925](https://github.com/vllm-project/vllm-ascend/pull/17925) separately adds `contiguous()` for non-PA FIA key/value inputs. The rc1-default FLA NPU recurrent Python/stable-ABI adapters pass the strided state descriptor through without an explicit state `.contiguous()` call. Therefore FIA/layout conversion is a stronger source-level hypothesis for the two long entries than a Python-level FLA state copy, but the aggregate profiler table does **not** establish their exact call sites; CANN could also insert an internal conversion. Capture a stack trace and input shapes before attribution. The newer [PR #18621](https://github.com/vllm-project/vllm-ascend/pull/18621) proposes restoring contiguous hybrid layouts; it is open as of this note and is not part of our 0.30 baseline.

## Current AR kernel versus native recurrent

The TP4/BS1 profile reports 48 native `RecurrentGatedDeltaRule` calls totaling 0.7073 ms (14.74 µs/call), versus 48 `replayssm_ar_4` calls totaling 2.9766 ms (62.01 µs/call): 4.21× per call and +2.2693 ms over the 48-layer path. The detailed single-kernel rows show 14.36 versus 62.38 µs. Replay vector time is 32.367 versus 4.351 µs, scalar 19.119 versus 4.465 µs, and MTE2 10.344 versus 1.494 µs. These device counters overlap and must not be added together. The earlier 125–175 µs eager NPU Event results include host launch gaps and cannot predict graph-resident kernel time.

`csrc/kernels/replayssm_ar.cpp` currently loads the full BF16 checkpoint tile on every decode, replays `pos` history rows serially with state-wide multiply and outer-add, then performs the current GDN update. Three scalar metadata loads and each history step introduce dependencies/barriers. Native recurrent performs only the current update. The extra Replay work, especially vector and scalar serialization, is a more direct explanation of the 62 µs than a simple memory-bandwidth argument. Next performance probe should group graph kernel durations by ring position 0/8/15 and flush flag. Optimize the non-flush output path to avoid reconstructing the full matrix only after algebra and precision are validated; keep flush as an explicit checkpoint materialization path.

At the current 30.399 ms device-op aggregate, Replay GDN itself is 9.8%; even eliminating that kernel completely would save at most 2.977 ms in this particular aggregate. Comparing the 30.399 ms and 68.875 ms totals across different CANN/vLLM releases is invalid, especially while the old trace contains roughly 30 ms of layout conversions. Measure native and Replay in the **same** image and graph configuration.

## Memory accounting: AR versus SD

AR retains a full checkpoint and adds D/K/G history. It saves repeated **state writes** between flushes, not the logical persistent-state size. With TP4, L=16, HV=12, H=4 and BF16/FP16/FP32 as configured, one GDN slot adds 48 KiB D + 16 KiB K + 0.75 KiB G = **64.75 KiB logical ring**. Across 48 layers and 1510 slots this is 4.476 GiB of logical payload per rank. Physical growth can be lower or zero if the ring fits existing page padding; cache groups can also overlay the same backing. SD's proposed snapshot/rollback memory saving is separate and is not enabled by current AR dispatch.

The probe logs three different quantities per rank: `logical_state_component_bytes` from shapes, `planned_device_backing_bytes` as the sum of `KVCacheConfig.kv_cache_tensors` entries, and `unique_backing_bytes` from deduplicated cache view storages. The configured sum may double-count overlaid hybrid allocations; it is **not** a physical HBM total. The probe also logs `torch_allocated_bytes` and `torch_reserved_bytes` after weights, after KV initialization, and before/after graph capture. It does not infer process-wide HBM from PyTorch allocator counters; use `npu-smi` alongside it.

The first same-build TP4 report returned identical native and Replay allocator values at every stage: 12.691 GiB after weights, 24.488 GiB after KV initialization, and 24.492 GiB allocated after graph capture. Both had 1510 blocks and 11.797 GiB of deduplicated cache backing. The configured allocation sum was 47.188 GiB in both runs, four times the deduplicated view storage, so it cannot be read as physical usage. Replay added D/K/G logical components (about 4.476 GiB at 1510 slots and 48 layers) inside the existing padded pages, with no measurable physical cache or graph-pool change in this configuration. Therefore the earlier cross-image ~1 GiB difference is not an AR storage saving.

## On-machine validation checklist

Use **the same** 0.30.0+empty / vLLM Ascend source build, CANN 9.2.0, model, TP, GPU utilization, graph mode and prompt for both runs. First stop other model processes. Set `REPLAYSSM_MODEL` to the server model path. In separate, sequential runs:

```bash
export VLLM_ASCEND_REPLAYSSM_MEMORY_PROBE=1
export REPLAYSSM_MODEL=/path/to/Qwen3.6-27B

vllm serve "$REPLAYSSM_MODEL" \
  --host 127.0.0.1 --port 18083 --served-model-name memory-native \
  --tensor-parallel-size 4 --pipeline-parallel-size 1 \
  --dtype bfloat16 --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype bfloat16 \
  --mamba-cache-mode none --no-enable-prefix-caching --enable-chunked-prefill \
  --language-model-only --max-model-len 4096 --max-num-seqs 16 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.85 \
  --distributed-executor-backend mp --no-async-scheduling --seed 42 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  2>&1 | tee /tmp/replayssm-memory-native.log
```

After `/health` passes, send one `POST /v1/completions` request with `temperature=0` and `max_tokens=64`; record `npu-smi info`, then stop the server and verify all ranks exited. Repeat the same command with `--served-model-name memory-replay --use-replayssm --replayssm-buffer-len 16` and `/tmp/replayssm-memory-replay.log`. Run:

```bash
python scripts/replayssm/summarize_memory_probe.py \
  --native /tmp/replayssm-memory-native.log \
  --replay /tmp/replayssm-memory-replay.log
```

- [ ] Four workers emit `model_loaded`, `kv_cache_initialized`, `graph_capture_before`, `graph_capture_after`; no `ReplaySSM memory probe failed` line.
- [ ] Record source commit, `vllm`/`vllm-ascend` packages, CANN version, TP and per-rank `num_blocks` for both runs.
- [ ] Compare deduplicated `unique_view_backing_bytes`, allocator deltas and GDN page sizes first; treat `planned_device_backing_bytes` as an overlapping plan sum. If `num_blocks` differs, compare per-block geometry before total GiB.
- [ ] Compare KV initialization and graph capture deltas in `torch_allocated_bytes`/`torch_reserved_bytes`; record idle and post-request `npu-smi` per card.
- [ ] Confirm one sample GDN view has expected page stride and five components under Replay; report any unexpected full-cache `contiguous` profiler rows separately with shape and call stack.
- [ ] Once TP4 reports are complete, repeat with `--tensor-parallel-size 8` and eight idle cards; keep every other setting identical between native and Replay TP8 runs.

The probe is opt-in and never runs inside decode. It does not change kernel dispatch, allocator layout, or cache capacity.

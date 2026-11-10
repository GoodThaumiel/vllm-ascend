# ReplaySSM SD 310P operator pilot

This directory builds the Ascend310P3 ReplaySSM operators. The separate
[serve pilot](LEVEL310P_SERVE.md) documents how to opt in to the live path.
The three in-place Torch operators are registered under `replayssm_sd_310p`:
`verify_`, `compact_`, and `cursor_`. Verify dispatches a separate AscendC
instantiation for four (MTP3) or six (MTP5) physical candidate tokens.

Inputs and checkpoints are FP16, `g` and the gate ring are FP32, and the
`d`/`k` rings are FP16. The logical history is 16 and the physical ring is 32.
The output of verify is FP16. The implementation keeps the validated 910B
ReplaySSM SD recurrence, but uses the 310P native recurrent kernel as the
reference for supported AscendC APIs and for operator-only timing.

Build and on-device validation commands are in `LEVEL310P_OPERATORS.md`.
The 310P baseline is CANN 9.2.1 with torch_npu 2.10.0.post7. A complete,
zero-exit CMake build is required before a device result is considered valid.

V12 adds an isolated 32-byte status-write ablation and a request-owned ring
allocation demo. See `LEVEL310P_V12.md` for the new device checklist. The
demo reports projected FP16 SSM payload savings; it does not change the live
vLLM Mamba cache allocation or enable `vllm serve` dispatch.

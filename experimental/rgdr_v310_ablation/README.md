# 310P native recurrent GDR ablations

This is an isolated CANN 9.2.1 / Ascend310P3 project. It does not alter the
production V310 operator or vLLM dispatch. The input contract matches the
preprocessed call to `npu_recurrent_gated_delta_rule_310`: FP16 Q/K/V, beta
and SSM state, FP32 g, packed int32 sequence lengths/state indices/accepted
counts. Q/K normalization and MTP sampling are outside the timed region.

The target geometry is TP2 Qwen3.5-35B-A3B, batch 32, NK=8, NV=16,
DK=DV=128. MTP3 and MTP7 correspond to 4 and 8 tokens per request.
The vector kernel keeps the native V310 recurrence, FP16 state writeback,
and host tiling's 128-wide V step. Its four entrypoints isolate two changes:

| Entry | Task mapping | State-index reads |
| --- | --- | --- |
| `vec0` | Native dynamic load scan | Per token and value head |
| `vec1` | Direct contiguous `(request, value head)` partition | Per token and value head |
| `vec2` | Native dynamic load scan | Once per token per assigned request |
| `vec3` | Direct partition | Once per token per assigned request |

`vec0` is the correctness oracle for `vec1`–`vec3`. The separately built
native ACLNN operator is the performance baseline. `bench.py` captures one
operator call per graph and excludes state reset from the timed region.
Results against the native operator use an FP16 tolerance; the three vector
ablations must match `vec0` bitwise.

The Cube experiment is a **microprobe**, not a full recurrent GDR kernel.
It computes 16x128 FP16 state tiles times 128-element FP16 vectors with
low-level GM→L1, `LoadData`, `Mmad`, L0C→UB, and UB→GM transfers. Its
control uses the native-style vector `Mul`/fold/reduce on identical tiles.
The Cube path pads each vector to a 128x16 RHS with only column zero used.
This packing and its extra storage are prepared outside graph timing, so a
Cube microprobe win is only a reason to investigate a fused full operator.

On-board commands and stop conditions are in [LEVEL310P_NATIVE_ABLATION.md](LEVEL310P_NATIVE_ABLATION.md).

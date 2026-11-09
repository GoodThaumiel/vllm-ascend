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
An improvement is actionable only if correctness passes and the **full**
vector variant is faster than native on both MTP sizes. Cube integration
requires a later full-operator implementation and its own correctness/perf
gate.

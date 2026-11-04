# SPDX-License-Identifier: Apache-2.0
"""Isolated, non-production probes. One process = one workload and symbol."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experimental.replayssm_sd_ascendc.lab import make_case, recurrent_golden  # noqa: E402

MODES = (
    "baseline",
    "a1_control",
    "a1_replace",
    "a2_control",
    "a2_remove",
    "local",
    "u1_chunk4k",
    "u1_chunk8k",
    "u2_history_tree",
)
NAMES = (
    "q",
    "k",
    "v",
    "g",
    "beta",
    "checkpoint",
    "d_cache",
    "k_cache",
    "g_cache",
    "slots",
    "ring_slots",
    "positions",
    "bases",
    "lengths",
    "out",
    "status",
)


def fixture(batch, pos, length, base, padding=False):
    cpu = make_case(
        tp=4, batch=batch, position=16, length=length, base=base, ring_len=32, tile_v=128, seed=42, padding=padding
    )
    cpu["positions"].fill_(pos)
    if padding:
        cpu["positions"][-1] = 0
    for name in ("d_cache", "k_cache", "g_cache", "out"):
        cpu[name].nan_to_num_(nan=0.0)
    old = cpu["checkpoint"]
    cpu["checkpoint"] = torch.zeros((batch + 5, 12, 128, 128), dtype=old.dtype)
    cpu["checkpoint"][5:].copy_(old[1:])
    cpu["ring_slots"] = cpu["slots"].clone()
    cpu["slots"] = torch.where(cpu["slots"] == 0, 0, cpu["slots"] + 4)
    return cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--batch", type=int, choices=(1, 2, 8, 16, 32, 64), default=32)
    parser.add_argument("--pos", type=int, choices=range(17), default=7)
    parser.add_argument("--length", type=int, choices=(1, 4), default=4)
    parser.add_argument("--base", type=int, choices=range(32), default=0)
    parser.add_argument("--cores", type=int, choices=(1, 4, 8, 40), default=40)
    parser.add_argument("--inner-repeats", type=int, choices=(1, 16, 64), default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--padding", action="store_true")
    parser.add_argument("--profiling-only", action="store_true", required=True)
    parser.add_argument("--library", type=Path, default=Path(__file__).parent / "build/libreplayssm_sd_torch.so")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or (args.inner_repeats != 1 and args.mode not in MODES[5:]):
        parser.error("iterations must be positive; repeated full-grid runs are not part of this experiment")
    if args.mode in MODES[5:] and args.padding:
        parser.error("local/API probes require an active first row")
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    library = args.library.resolve()
    torch.ops.load_library(str(library))
    cpu = fixture(args.batch, args.pos, args.length, args.base, args.padding)
    device = torch.device(f"npu:{args.device}")
    case = {k: v.to(device).contiguous() for k, v in cpu.items()}
    # Guard both ends of sink; no timed reset/copy. No cursor movement: candidate writes never overlap history.
    backing = torch.full((args.batch * 12 * 4096 + 16,), 12345.0, device=device)
    sink = backing[8:-8].view(args.batch * 12, 4096)
    tensors = [case[n] for n in NAMES]

    def call():
        torch.ops.replayssm_sd_probing.run_(tensors, sink, MODES.index(args.mode), args.cores, args.inner_repeats, True)

    golden = None
    if args.batch == 2:
        tied = dict(cpu)
        tied["slots"] = cpu["ring_slots"]
        tied["checkpoint"] = torch.cat((cpu["checkpoint"][:1], cpu["checkpoint"][5:]))
        golden = recurrent_golden(tied)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "profiling_only": True,
        "command": sys.argv,
        "parameters": vars(args),
        "layout": "detached_contiguous",
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "torch-npu")},
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "input_sha256": {n: hashlib.sha256(bytes(t.untyped_storage())).hexdigest() for n, t in cpu.items()},
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, default=str, indent=2) + "\n")
    # One safety invocation; only baseline/control are expected to preserve the dense result.
    call()
    torch.npu.synchronize()
    if args.mode in MODES[:5]:
        assert not case["status"].cpu().any(), "nonzero status"
        assert torch.isfinite(case["out"]).all().item(), "non-finite output"
    if golden is not None and args.mode in ("baseline", "a1_control", "a2_control"):
        for name in ("out", "d_cache", "k_cache", "g_cache"):
            torch.testing.assert_close(case[name].cpu().float(), golden[name].float(), atol=0.01, rtol=0.03)
        torch.testing.assert_close(case["checkpoint"].cpu(), cpu["checkpoint"], atol=0, rtol=0)
    if golden is not None and args.mode == "local":
        torch.testing.assert_close(
            case["out"][0, :, 0].cpu().float(), golden["out"][0, :, 0].float(), atol=0.01, rtol=0.03
        )
    assert torch.equal(backing[:8].cpu(), torch.full((8,), 12345.0))
    assert torch.equal(backing[-8:].cpu(), torch.full((8,), 12345.0))
    # All modes operate on immutable input history; capture adds one kernel invocation.
    for _ in range(2):
        call()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        call()
    print(f"MEASURE_BEGIN mode={args.mode} calls={args.iterations}", flush=True)
    times = []
    for _ in range(args.iterations):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000)
    print("MEASURE_END", flush=True)
    assert torch.isfinite(sink).all().item()
    assert torch.equal(backing[:8].cpu(), torch.full((8,), 12345.0))
    assert torch.equal(backing[-8:].cpu(), torch.full((8,), 12345.0))
    torch.testing.assert_close(case["checkpoint"].cpu(), cpu["checkpoint"], atol=0, rtol=0)
    for row in range(args.batch):
        ring_slot = int(cpu["ring_slots"][row])
        count = int(cpu["positions"][row])
        indices = [(int(cpu["bases"][row]) + i) % 32 for i in range(count)]
        for name in ("d_cache", "k_cache", "g_cache"):
            torch.testing.assert_close(
                case[name].cpu()[ring_slot, :, indices], cpu[name][ring_slot, :, indices], atol=0, rtol=0
            )
    result = dict(vars(args))
    result.update(
        status="PASSED",
        profiling_only=True,
        numerical_equivalence_required=False,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        source_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
        input_sha256={n: hashlib.sha256(bytes(t.untyped_storage())).hexdigest() for n, t in cpu.items()},
        shapes={n: {"shape": list(t.shape), "stride": list(t.stride())} for n, t in case.items()},
        graph_us_p50=statistics.median(times),
        graph_us_p90=sorted(times)[int(0.9 * (len(times) - 1))],
        graph_us_samples=times,
        expected_symbol_calls=args.iterations + 4,
        tasks=1 if args.mode in MODES[5:] else args.batch * 12,
        note=(
            "1 safety + 2 warmup + 1 capture + N measured calls; "
            "verify profiler actual count. No packing/reset in timed graph."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, default=str, indent=2) + "\n")


if __name__ == "__main__":
    main()

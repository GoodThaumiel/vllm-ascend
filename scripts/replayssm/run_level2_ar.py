# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""User-operated Level 2 correctness and NPUGraph probe for ReplaySSM AR."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def child_command(args, device: int, report: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--tp",
        str(args.tp),
        "--output",
        str(report.parent),
        "--child-device",
        str(device),
        "--child-report",
        str(report),
        "--capacity",
        str(args.capacity),
    ]


def run_device(device: int, tp: int, capacity: int) -> dict:
    import torch
    import torch_npu  # noqa: F401

    from vllm_ascend.ops.triton.fla import replayssm_ar
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    fused_recurrent_gated_delta_rule_replayssm = replayssm_ar.fused_recurrent_gated_delta_rule_replayssm

    torch.npu.set_device(device)
    init_device_properties_triton()
    torch.manual_seed(20261019 + device + tp)
    npu = torch.device(f"npu:{device}")
    batch = 2
    slots = batch + 1
    key_heads = 16 // tp
    value_heads = 48 // tp
    key_dim = value_dim = 128
    scale = key_dim**-0.5

    checkpoint = torch.zeros(slots, value_heads, value_dim, key_dim, dtype=torch.bfloat16, device=npu)
    d_cache = torch.zeros(slots, value_heads, capacity, value_dim, dtype=torch.float16, device=npu)
    k_cache = torch.zeros(slots, key_heads, capacity, key_dim, dtype=torch.float16, device=npu)
    g_cache = torch.zeros(slots, value_heads, capacity, dtype=torch.float32, device=npu)
    indices = torch.tensor([1, 2], dtype=torch.int32, device=npu)
    write_pos = torch.zeros(batch, dtype=torch.int32, device=npu)
    is_flush = torch.zeros(batch, dtype=torch.int8, device=npu)
    out = torch.empty(batch, 1, value_heads, value_dim, dtype=torch.bfloat16, device=npu)
    a_log_cpu = torch.linspace(-2.2, -1.8, value_heads, dtype=torch.float32)
    dt_bias_cpu = torch.linspace(-0.1, 0.1, value_heads, dtype=torch.float32)
    a_log = a_log_cpu.to(npu)
    dt_bias = dt_bias_cpu.to(npu)

    checkpoint_ref = torch.zeros(slots, value_heads, value_dim, key_dim, dtype=torch.float32)
    direct_state = checkpoint_ref.clone()
    history: list[list[list[tuple[torch.Tensor, torch.Tensor, float]]]] = [
        [[] for _ in range(value_heads)] for _ in range(slots)
    ]
    positions = [0] * batch
    max_output_error = 0.0
    max_checkpoint_error = 0.0
    max_direct_error = 0.0
    first_launch_ms = None

    def make_inputs(step: int, rows: int = batch):
        generator = torch.Generator().manual_seed(7100 + device * 101 + tp * 17 + step)
        q = torch.randn(rows, key_heads, key_dim, generator=generator) * 0.15
        k = torch.randn(rows, key_heads, key_dim, generator=generator) * 0.15
        v = torch.randn(rows, value_heads, value_dim, generator=generator) * 0.15
        mixed = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=1).to(torch.bfloat16)
        a = (torch.randn(rows, value_heads, generator=generator) * 0.3).to(torch.bfloat16)
        b = (torch.randn(rows, value_heads, generator=generator) * 0.3).to(torch.bfloat16)
        return mixed, a, b

    def reference_step(mixed, a_cpu, b_cpu, slots_for_rows, flush_flags, histories, checkpoints, direct):
        q_width = key_heads * key_dim
        q = mixed[:, :q_width].float().reshape(-1, key_heads, key_dim)
        k = mixed[:, q_width : 2 * q_width].float().reshape(-1, key_heads, key_dim)
        v = mixed[:, 2 * q_width :].float().reshape(-1, value_heads, value_dim)
        q = q / torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt(torch.sum(k * k, dim=-1, keepdim=True) + 1e-6)
        gate = -torch.exp(a_log_cpu)[None, :] * torch.nn.functional.softplus(a_cpu.float() + dt_bias_cpu)
        beta = torch.sigmoid(b_cpu.float()).to(b_cpu.dtype).float()
        expected = torch.zeros(len(slots_for_rows), value_heads, value_dim, dtype=torch.float32)
        current_d = {}
        current_k = {}
        for row, slot in enumerate(slots_for_rows):
            if slot <= 0:
                continue
            for value_head in range(value_heads):
                key_head = value_head // (value_heads // key_heads)
                state = checkpoints[slot, value_head].clone()
                for old_d, old_k, old_g in histories[slot][value_head]:
                    state = torch.exp(torch.tensor(old_g)) * state + old_d[:, None] * old_k[None, :]
                alpha = torch.exp(gate[row, value_head])
                decayed = alpha * state
                delta = beta[row, value_head] * (v[row, value_head] - decayed @ k[row, key_head])
                new_state = decayed + delta[:, None] * k[row, key_head][None, :]
                expected[row, value_head] = new_state @ (q[row, key_head] * scale)
                direct_decayed = alpha * direct[slot, value_head]
                direct_delta = beta[row, value_head] * (v[row, value_head] - direct_decayed @ k[row, key_head])
                direct[slot, value_head] = direct_decayed + direct_delta[:, None] * k[row, key_head][None, :]
                if flush_flags[row]:
                    checkpoints[slot, value_head] = new_state.to(torch.bfloat16).float()
                    histories[slot][value_head].clear()
                else:
                    stored_d = delta.to(torch.float16).float()
                    stored_k = k[row, key_head].to(torch.float16).float()
                    histories[slot][value_head].append((stored_d, stored_k, float(gate[row, value_head])))
                    current_d[(slot, value_head)] = stored_d
                    current_k[(slot, key_head)] = stored_k
        return expected.to(torch.bfloat16), current_d, current_k, gate

    sequence_steps = capacity * 2 + 3
    for step in range(sequence_steps):
        mixed_cpu, a_cpu, b_cpu = make_inputs(step)
        mixed_npu, a_npu, b_npu = mixed_cpu.to(npu), a_cpu.to(npu), b_cpu.to(npu)
        flush_flags = [position == capacity - 1 for position in positions]
        if step == 2:
            flush_flags[0] = True
        write_pos.copy_(torch.tensor(positions, dtype=torch.int32, device=npu))
        is_flush.copy_(torch.tensor(flush_flags, dtype=torch.int8, device=npu))
        expected, current_d, current_k, gate = reference_step(
            mixed_cpu,
            a_cpu,
            b_cpu,
            [1, 2],
            flush_flags,
            history,
            checkpoint_ref,
            direct_state,
        )
        launch_started = time.perf_counter()
        fused_recurrent_gated_delta_rule_replayssm(
            mixed_npu,
            a_npu,
            b_npu,
            a_log,
            dt_bias,
            scale,
            checkpoint,
            d_cache,
            k_cache,
            g_cache,
            out,
            indices,
            write_pos,
            is_flush,
            use_qk_l2norm_in_kernel=True,
        )
        torch.npu.synchronize()
        if first_launch_ms is None:
            first_launch_ms = (time.perf_counter() - launch_started) * 1000
        actual = out[:, 0].cpu()
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
        max_output_error = max(max_output_error, float((actual.float() - expected.float()).abs().max()))
        for row, slot in enumerate((1, 2)):
            if flush_flags[row]:
                actual_checkpoint = checkpoint[slot].cpu().float()
                expected_checkpoint = checkpoint_ref[slot]
                torch.testing.assert_close(actual_checkpoint, expected_checkpoint, atol=3e-2, rtol=3e-2)
                max_checkpoint_error = max(
                    max_checkpoint_error, float((actual_checkpoint - expected_checkpoint).abs().max())
                )
                max_direct_error = max(max_direct_error, float((actual_checkpoint - direct_state[slot]).abs().max()))
                positions[row] = 0
            else:
                position = positions[row]
                for value_head in range(value_heads):
                    torch.testing.assert_close(
                        d_cache[slot, value_head, position].cpu().float(),
                        current_d[(slot, value_head)],
                        atol=3e-2,
                        rtol=3e-2,
                    )
                    torch.testing.assert_close(
                        g_cache[slot, value_head, position].cpu(),
                        gate[row, value_head],
                        atol=5e-4,
                        rtol=5e-4,
                    )
                for key_head in range(key_heads):
                    torch.testing.assert_close(
                        k_cache[slot, key_head, position].cpu().float(),
                        current_k[(slot, key_head)],
                        atol=3e-2,
                        rtol=3e-2,
                    )
                positions[row] += 1

    # Reset reuses a slot with a zeroed checkpoint and write_pos=0. Stale ring
    # bytes remain deliberately populated and must be ignored.
    checkpoint[1].zero_()
    reset_indices = torch.tensor([1, 0], dtype=torch.int32, device=npu)
    write_pos.zero_()
    is_flush.zero_()
    mixed_cpu, a_cpu, b_cpu = make_inputs(850)
    reset_history = [[[] for _ in range(value_heads)] for _ in range(slots)]
    reset_checkpoint_ref = torch.zeros_like(checkpoint_ref)
    reset_direct = torch.zeros_like(direct_state)
    reset_expected, _, _, _ = reference_step(
        mixed_cpu,
        a_cpu,
        b_cpu,
        [1, 0],
        [False, False],
        reset_history,
        reset_checkpoint_ref,
        reset_direct,
    )
    fused_recurrent_gated_delta_rule_replayssm(
        mixed_cpu.to(npu),
        a_cpu.to(npu),
        b_cpu.to(npu),
        a_log,
        dt_bias,
        scale,
        checkpoint,
        d_cache,
        k_cache,
        g_cache,
        out,
        reset_indices,
        write_pos,
        is_flush,
        use_qk_l2norm_in_kernel=True,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(out[:, 0].cpu(), reset_expected, atol=3e-2, rtol=3e-2)

    # Slot zero is padding: its output must be deterministic zero and no state is addressed.
    padded_indices = torch.tensor([1, 0], dtype=torch.int32, device=npu)
    mixed_cpu, a_cpu, b_cpu = make_inputs(900)
    write_pos.zero_()
    is_flush.fill_(1)
    fused_recurrent_gated_delta_rule_replayssm(
        mixed_cpu.to(npu),
        a_cpu.to(npu),
        b_cpu.to(npu),
        a_log,
        dt_bias,
        scale,
        checkpoint,
        d_cache,
        k_cache,
        g_cache,
        out,
        padded_indices,
        write_pos,
        is_flush,
        use_qk_l2norm_in_kernel=True,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(out[1].cpu(), torch.zeros_like(out[1].cpu()), atol=0, rtol=0)

    # Capture one static launch, then change inputs and the runtime flush flag.
    graph_checkpoint = torch.zeros_like(checkpoint)
    graph_d = torch.zeros_like(d_cache)
    graph_k = torch.zeros_like(k_cache)
    graph_g = torch.zeros_like(g_cache)
    graph_indices = torch.tensor([1, 0], dtype=torch.int32, device=npu)
    graph_pos = torch.zeros(batch, dtype=torch.int32, device=npu)
    graph_flush = torch.zeros(batch, dtype=torch.int8, device=npu)
    graph_out = torch.empty_like(out)
    graph_mixed = torch.zeros(
        batch, 2 * key_heads * key_dim + value_heads * value_dim, dtype=torch.bfloat16, device=npu
    )
    graph_a = torch.zeros(batch, value_heads, dtype=torch.bfloat16, device=npu)
    graph_b = torch.zeros(batch, value_heads, dtype=torch.bfloat16, device=npu)

    def graph_launch():
        fused_recurrent_gated_delta_rule_replayssm(
            graph_mixed,
            graph_a,
            graph_b,
            a_log,
            dt_bias,
            scale,
            graph_checkpoint,
            graph_d,
            graph_k,
            graph_g,
            graph_out,
            graph_indices,
            graph_pos,
            graph_flush,
            use_qk_l2norm_in_kernel=True,
        )

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_launch()
    graph_errors = []
    for replay, flush_value in enumerate((0, 1, 0)):
        graph_checkpoint.zero_()
        graph_d.zero_()
        graph_k.zero_()
        graph_g.zero_()
        mixed_cpu, a_cpu, b_cpu = make_inputs(1000 + replay)
        graph_mixed.copy_(mixed_cpu.to(npu))
        graph_a.copy_(a_cpu.to(npu))
        graph_b.copy_(b_cpu.to(npu))
        graph_flush.fill_(flush_value)
        graph_history = [[[] for _ in range(value_heads)] for _ in range(slots)]
        graph_checkpoint_ref = torch.zeros_like(checkpoint_ref)
        graph_direct = torch.zeros_like(direct_state)
        expected, _, _, _ = reference_step(
            mixed_cpu,
            a_cpu,
            b_cpu,
            [1, 0],
            [bool(flush_value), False],
            graph_history,
            graph_checkpoint_ref,
            graph_direct,
        )
        graph.replay()
        torch.npu.synchronize()
        actual = graph_out[:, 0].cpu()
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
        graph_errors.append(float((actual.float() - expected.float()).abs().max()))

    graph_flush.fill_(1)
    for _ in range(3):
        graph_launch()
    torch.npu.synchronize()
    benchmark_repeats = 20
    benchmark_started = time.perf_counter()
    for _ in range(benchmark_repeats):
        graph_launch()
    torch.npu.synchronize()
    eager_mean_ms = (time.perf_counter() - benchmark_started) * 1000 / benchmark_repeats
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    benchmark_started = time.perf_counter()
    for _ in range(benchmark_repeats):
        graph.replay()
    torch.npu.synchronize()
    graph_replay_mean_ms = (time.perf_counter() - benchmark_started) * 1000 / benchmark_repeats

    return {
        "status": "PASSED",
        "device": device,
        "tp_geometry": tp,
        "key_heads": key_heads,
        "value_heads": value_heads,
        "key_dim": key_dim,
        "value_dim": value_dim,
        "capacity": capacity,
        "sequence_steps": sequence_steps,
        "max_output_abs_error": max_output_error,
        "max_checkpoint_abs_error": max_checkpoint_error,
        "max_error_vs_direct_recurrence_at_flush": max_direct_error,
        "graph_replays": 3,
        "graph_flush_values": [0, 1, 0],
        "graph_max_abs_errors": graph_errors,
        "reset_with_stale_ring": "PASSED",
        "first_launch_ms_including_compile": first_launch_ms,
        "eager_mean_ms": eager_mean_ms,
        "graph_replay_mean_ms": graph_replay_mean_ms,
        "benchmark_repeats": benchmark_repeats,
        "dtypes": {"checkpoint": "bfloat16", "d_cache": "float16", "k_cache": "float16", "g_cache": "float32"},
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "vllm-ascend", "torch", "torch-npu", "triton-ascend")
        },
        "module_paths": {"torch_npu": torch_npu.__file__, "kernel": replayssm_ar.__file__},
        "ascend_home_path": os.environ.get("ASCEND_HOME_PATH"),
    }


def git_evidence() -> dict:
    root = Path(__file__).resolve().parents[2]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "status", "--short"], cwd=root, capture_output=True, text=True, check=False)
    return {"commit": head.stdout.strip(), "status_short": status.stdout.splitlines()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, choices=(4, 8), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--device-timeout", type=int, default=900)
    parser.add_argument("--child-device", type=int, choices=range(8))
    parser.add_argument("--child-report", type=Path)
    args = parser.parse_args()
    if args.capacity < 2:
        parser.error("--capacity must be at least 2")

    if args.child_device is not None:
        if args.child_report is None:
            parser.error("--child-report is required with --child-device")
        record = {"status": "FAILED", "device": args.child_device, "tp_geometry": args.tp}
        try:
            record.update(run_device(args.child_device, args.tp, args.capacity))
        except Exception as exc:
            record.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        write_json(args.child_report, record)
        return 0 if record["status"] == "PASSED" else 1

    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    records = []
    for device in range(args.tp):
        report = args.output / f"device{device}.json"
        log = args.output / f"device{device}.log"
        with log.open("w") as stream:
            try:
                completed = subprocess.run(
                    child_command(args, device, report),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=args.device_timeout,
                )
            except subprocess.TimeoutExpired:
                completed = None
        if report.is_file():
            record = json.loads(report.read_text())
        elif completed is None:
            record = {"status": "FAILED", "device": device, "error": "device probe timed out"}
        else:
            record = {
                "status": "FAILED",
                "device": device,
                "error": f"missing report; returncode={completed.returncode}",
            }
        records.append(record)
        if completed is None or completed.returncode != 0 or record.get("status") != "PASSED":
            break
    summary = {
        "level": 2,
        "scope": "ReplaySSM AR Triton operator only; no model or scheduler integration",
        "status": "AUTOMATED_CHECKS_PASSED"
        if len(records) == args.tp and all(r["status"] == "PASSED" for r in records)
        else "FAILED_OR_INCOMPLETE",
        "tp_geometry": args.tp,
        "source": git_evidence(),
        "devices": records,
        "next_level_allowed": False,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "AUTOMATED_CHECKS_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

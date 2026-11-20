# SPDX-License-Identifier: Apache-2.0
"""Compare 310P compute WY and the complete chunk GDN against Torch WY."""

import argparse
import json
import math
import os
import sys
from functools import partial
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401

from vllm_ascend import envs
from vllm_ascend._310p.ops.compute_wy import load_compute_wy_library
from vllm_ascend._310p.ops.fla import chunk_gated_delta_rule as chunk
from vllm_ascend.utils import enable_custom_op

CHUNK_REL_L2_LIMIT = 0.01
WY_NAMES = ("q", "k", "w", "u", "g")


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual = actual.cpu().float()
    expected = expected.cpu().float()
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        return {"max_abs": None, "relative_l2": None, "worst_index": None}
    diff = actual - expected
    rel_l2 = (diff.norm() / expected.norm().clamp_min(1e-12)).item()
    peak = diff.abs().flatten().argmax()
    index = [int(coordinate) for coordinate in torch.unravel_index(peak, diff.shape)]
    return {"max_abs": diff.abs().max().item(), "relative_l2": rel_l2, "worst_index": index}


def compare(actual: torch.Tensor, expected: torch.Tensor, name: str, limit: float) -> dict:
    metrics = error_metrics(actual, expected)
    rel_l2 = metrics["relative_l2"]
    if rel_l2 is None:
        raise AssertionError(f"{name} contains nonfinite values")
    if rel_l2 > limit:
        raise AssertionError(f"{name} relative L2 {rel_l2:.6g} exceeds {limit}")
    return metrics


def phase(case: str, step: str) -> None:
    print(f"[compute WY smoke] {case}: {step}", file=sys.stderr, flush=True)


def run_npu_phase(case, step, call, *args, **kwargs):
    phase(case, f"submit {step}")
    result = call(*args, **kwargs)
    phase(case, f"synchronize {step}")
    torch.npu.synchronize()
    return result


def inputs(batch, tokens, heads, dim, device, seed):
    generator = torch.Generator().manual_seed(seed)
    q, k = (
        torch.nn.functional.normalize(torch.randn(batch, tokens, heads, dim, generator=generator), dim=-1)
        .half()
        .to(device)
        for _ in range(2)
    )
    v = (torch.randn(batch, tokens, 2 * heads, dim, generator=generator) * 0.1).half().to(device)
    g = (-torch.rand(batch, tokens, 2 * heads, generator=generator) * 0.2).to(device)
    beta = (0.1 + 0.8 * torch.rand(batch, tokens, 2 * heads, generator=generator)).half().to(device)
    return q, k, v, g, beta


def check_wy(device: torch.device) -> None:
    for batch, tokens, heads, dim in ((1, 128, 8, 64), (2, 128, 8, 128), (1, 2048, 8, 128)):
        case = f"WY B={batch} T={tokens} D={dim}"
        phase(case, "prepare inputs")
        host = inputs(batch, tokens, heads, dim, "cpu", seed=310)
        tensors = tuple(t.to(device) for t in host)
        torch.npu.synchronize()

        actual = run_npu_phase(case, "AscendC", chunk.compute_wy_for_prefill, *tensors, 64)
        assert actual is not None
        phase(case, "copy outputs to CPU")
        actual = tuple(t.cpu() for t in actual)
        phase(case, "compute reference and compare on CPU")
        expected = chunk._compute_kernel_inputs_from_torch_wy(*host, 64)
        for i in (0, 1):
            torch.testing.assert_close(actual[i], expected[i], rtol=0, atol=0)
        torch.testing.assert_close(actual[4], expected[4], rtol=1e-5, atol=1e-5)
        metrics = {name: compare(actual[i], expected[i], name, 0.01) for i, name in ((2, "w"), (3, "u"))}
        print(json.dumps({"batch": batch, "tokens": tokens, "dim": dim, **metrics}), flush=True)


def compute_wy_on_cpu(q, k, v, g, beta, chunk_size):
    inputs_cpu = tuple(t.cpu() for t in (q, k, v, g, beta))
    result = chunk._compute_kernel_inputs_from_torch_wy(*inputs_cpu, chunk_size)
    return tuple(t.to(q.device) for t in result)


def h_nz_to_nd(h: torch.Tensor) -> torch.Tensor:
    # Inverse of the v24 binding's zN packing, with V blocks outside K rows.
    k_dim, v_dim = h.shape[-2:]
    return h.reshape(-1, v_dim // 16, k_dim // 16, 16, 16).permute(0, 2, 3, 1, 4).reshape(h.shape)


def cpu_chunk_reference(tensors, state, cu_seqlens):
    def normalize(x):
        dim = x.shape[-1]
        weight = torch.full((dim,), 1.0 / math.sqrt(dim), dtype=x.dtype).float()
        return torch.nn.functional.rms_norm(x.float(), (dim,), weight, 1e-6 / dim).to(x.dtype)

    q, k, v, g, beta = tensors
    return chunk.chunk_gated_delta_rule_pytorch(
        normalize(q),
        normalize(k),
        v,
        g,
        beta,
        initial_state=state,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
    )


def run_chunk(case, prepare_wy, tensors, kwargs):
    snapshots = {}

    def prepare(*args):
        phase(case, "synchronize input padding")
        torch.npu.synchronize()
        result = run_npu_phase(case, "compute WY", prepare_wy, *args)
        if result is None:
            raise RuntimeError("compute WY did not dispatch for the smoke shape")
        snapshots.update({name: value.cpu() for name, value in zip(WY_NAMES, result, strict=True)})
        return result

    select_ho_ops = chunk._select_chunk_ho_ops

    def trace_ho_ops(*args):
        fwd_h, fwd_o = select_ho_ops(*args)

        def trace_h(*args, **kwargs):
            result = run_npu_phase(case, fwd_h.__name__, fwd_h, *args, **kwargs)
            h, v_new = (tensor.cpu() for tensor in result[:2])
            snapshots["h"] = h_nz_to_nd(h) if fwd_h.__name__ == "chunk_gated_delta_rule_fwd_h" else h
            snapshots["v_new"] = v_new
            return result

        return trace_h, partial(run_npu_phase, case, fwd_o.__name__, fwd_o)

    # Copy device data only after each stage has synchronized.
    with (
        patch.object(chunk, "compute_wy_for_prefill", prepare),
        patch.object(chunk, "_select_chunk_ho_ops", trace_ho_ops),
    ):
        out, checkpoint = run_npu_phase(case, "chunk GDN", chunk.chunk_gated_delta_rule_310, *tensors, **kwargs)
    assert checkpoint is not None
    snapshots.update(output=out.cpu(), checkpoint=checkpoint.cpu())
    return snapshots


def report_chunk_comparison(name, lengths, actual, expected):
    metrics = {key: error_metrics(actual[key], value) for key, value in expected.items()}
    # Stage metrics show where the calculations diverge. Keep the smoke gate
    # on the visible output and the final state, with the existing limit.
    passed = all(
        metrics[key]["relative_l2"] is not None and metrics[key]["relative_l2"] <= CHUNK_REL_L2_LIMIT
        for key in ("output", "checkpoint")
    )
    sequence_metrics = [
        {
            "output": error_metrics(got, ref),
            "checkpoint": error_metrics(actual["checkpoint"][seq], expected["checkpoint"][seq]),
        }
        for seq, (got, ref) in enumerate(
            zip(actual["output"].split(lengths, dim=1), expected["output"].split(lengths, dim=1), strict=True)
        )
    ]
    print(
        json.dumps(
            {
                "lengths": lengths,
                "comparison": name,
                "limit": CHUNK_REL_L2_LIMIT,
                "metrics": metrics,
                "per_sequence": sequence_metrics,
                "status": "PASSED" if passed else "FAILED",
            },
            allow_nan=False,
        ),
        flush=True,
    )
    return passed


def check_chunk(device: torch.device) -> None:
    # Use packed, unaligned rows and a nonzero incoming checkpoint.
    lengths = [111, 2082]
    case = f"chunk GDN lengths={lengths}"
    phase(case, "prepare inputs")
    host = inputs(1, sum(lengths), 8, 128, "cpu", seed=311)
    tensors = tuple(tensor.to(device) for tensor in host)
    cu_cpu = torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32)
    cu = cu_cpu.to(device)
    generator = torch.Generator().manual_seed(312)
    state_cpu = (0.02 * torch.randn((2, 16, 128, 128), generator=generator)).half()
    state = state_cpu.to(device)
    kwargs = dict(initial_state=state, output_final_state=True, cu_seqlens=cu, use_qk_l2norm_in_kernel=True)
    torch.npu.synchronize()

    actual = run_chunk(f"{case} candidate AscendC WY", chunk.compute_wy_for_prefill, tensors, kwargs)
    cpu_wy = run_chunk(f"{case} candidate CPU Torch WY", compute_wy_on_cpu, tensors, kwargs)
    phase(case, "compute independent CPU chunk reference")
    golden_out, golden_state = cpu_chunk_reference(host, state_cpu, cu_cpu)
    assert golden_state is not None
    golden = {"output": golden_out, "checkpoint": golden_state}
    checks = [
        report_chunk_comparison("candidate_ascendc_vs_same_ho_cpu_wy", lengths, actual, cpu_wy),
        report_chunk_comparison("candidate_ascendc_vs_cpu_chunk", lengths, actual, golden),
        report_chunk_comparison("candidate_cpu_wy_vs_cpu_chunk", lengths, cpu_wy, golden),
    ]
    if envs.VLLM_ASCEND_310P_PREFILL_LEGACY_GDN_HO or envs.VLLM_ASCEND_310P_PREFILL_RMSNORM_PAD16:
        with patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_310P_PREFILL_LEGACY_GDN_HO": "0",
                "VLLM_ASCEND_310P_PREFILL_RMSNORM_PAD16": "0",
            },
        ):
            baseline = run_chunk(f"{case} v30 baseline AscendC WY", chunk.compute_wy_for_prefill, tensors, kwargs)
            baseline_cpu_wy = run_chunk(f"{case} v30 baseline CPU Torch WY", compute_wy_on_cpu, tensors, kwargs)
        checks.extend(
            (
                report_chunk_comparison("candidate_vs_v30_baseline_ascendc_wy", lengths, actual, baseline),
                report_chunk_comparison("v30_baseline_ascendc_vs_same_ho_cpu_wy", lengths, baseline, baseline_cpu_wy),
                report_chunk_comparison("v30_baseline_ascendc_vs_cpu_chunk", lengths, baseline, golden),
                report_chunk_comparison("v30_baseline_cpu_wy_vs_cpu_chunk", lengths, baseline_cpu_wy, golden),
            )
        )
    if not all(checks):
        raise AssertionError("chunk GDN comparison failed; see per-stage and per-sequence JSON above")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--stage", choices=("all", "wy", "chunk"), default="all")
    args = parser.parse_args()
    device = torch.device(f"npu:{args.device}")
    phase("startup", f"initialize {device}")
    torch.npu.set_device(device)
    if args.stage != "wy" and not enable_custom_op():
        raise RuntimeError("The stock 310P fwd_h/fwd_o extension is required")
    phase("startup", "load compute WY library")
    library = load_compute_wy_library()
    if library is None:
        raise RuntimeError("Set VLLM_ASCEND_310P_COMPUTE_WY_LIBRARY before running the smoke test")
    if args.stage != "chunk":
        check_wy(device)
    if args.stage != "wy":
        check_chunk(device)


if __name__ == "__main__":
    main()

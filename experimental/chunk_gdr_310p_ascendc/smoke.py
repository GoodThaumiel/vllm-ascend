# SPDX-License-Identifier: Apache-2.0
"""Compare 310P compute WY and the complete chunk GDN against Torch WY."""

import argparse
import json
import os
import sys
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401

from vllm_ascend._310p.ops.compute_wy import load_compute_wy_library
from vllm_ascend._310p.ops.fla import chunk_gated_delta_rule as chunk
from vllm_ascend.utils import enable_custom_op


def compare(actual: torch.Tensor, expected: torch.Tensor, name: str, limit: float) -> dict:
    actual = actual.cpu().float()
    expected = expected.cpu().float()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError(f"{name} contains nonfinite values")
    diff = actual - expected
    rel_l2 = (diff.norm() / expected.norm().clamp_min(1e-12)).item()
    if rel_l2 > limit:
        raise AssertionError(f"{name} relative L2 {rel_l2:.6g} exceeds {limit}")
    return {"max_abs": diff.abs().max().item(), "relative_l2": rel_l2}


def phase(case: str, step: str) -> None:
    print(f"[compute WY smoke] {case}: {step}", file=sys.stderr, flush=True)


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

        phase(case, "submit AscendC")
        actual = chunk.compute_wy_for_prefill(*tensors, 64)
        assert actual is not None
        phase(case, "synchronize AscendC")
        torch.npu.synchronize()
        phase(case, "copy outputs to CPU")
        actual = tuple(t.cpu() for t in actual)
        phase(case, "compute reference and compare on CPU")
        expected = chunk._compute_kernel_inputs_from_torch_wy(*host, 64)
        for i in (0, 1):
            torch.testing.assert_close(actual[i], expected[i], rtol=0, atol=0)
        torch.testing.assert_close(actual[4], expected[4], rtol=1e-5, atol=1e-5)
        metrics = {name: compare(actual[i], expected[i], name, 0.01) for i, name in ((2, "w"), (3, "u"))}
        print(json.dumps({"batch": batch, "tokens": tokens, "dim": dim, **metrics}), flush=True)


def check_chunk(device: torch.device) -> None:
    # Use packed, unaligned rows and a nonzero incoming checkpoint.
    lengths = [111, 2081]
    case = f"chunk GDN lengths={lengths}"
    phase(case, "prepare inputs")
    tensors = inputs(1, sum(lengths), 8, 128, device, seed=311)
    cu = torch.tensor([0, lengths[0], sum(lengths)], device=device, dtype=torch.int32)
    state = torch.full((2, 16, 128, 128), 0.01, device=device, dtype=torch.float16)
    kwargs = dict(initial_state=state, output_final_state=True, cu_seqlens=cu)
    torch.npu.synchronize()

    phase(case, "submit with AscendC WY")
    out, checkpoint = chunk.chunk_gated_delta_rule_310(*tensors, **kwargs)
    phase(case, "synchronize with AscendC WY")
    torch.npu.synchronize()
    phase(case, "submit with Torch WY")
    with patch.dict(os.environ, {"VLLM_ASCEND_310P_COMPUTE_WY_LIBRARY": ""}):
        out_ref, checkpoint_ref = chunk.chunk_gated_delta_rule_310(*tensors, **kwargs)
    phase(case, "synchronize with Torch WY")
    torch.npu.synchronize()
    assert checkpoint is not None and checkpoint_ref is not None
    phase(case, "copy outputs and compare on CPU")
    print(
        json.dumps(
            {
                "lengths": lengths,
                "output": compare(out, out_ref, "output", 0.01),
                "checkpoint": compare(checkpoint, checkpoint_ref, "checkpoint", 0.01),
                "status": "PASSED",
            }
        ),
        flush=True,
    )


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

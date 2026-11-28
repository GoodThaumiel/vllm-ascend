# SPDX-License-Identifier: Apache-2.0
"""Measure same-die HBM-copy, BF16 Cube, and FP32 AIV rates for roofline plots."""

import argparse
import hashlib
import json
import sys
from functools import partial
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experimental.replayssm_cube_910b.run import measure  # noqa: E402
from scripts.replayssm.profile_sd_operator_compare import package_versions  # noqa: E402


def calibrate(device, iterations):
    copy_cases = []
    for mib in (256, 512):
        size = mib * 1024**2
        src = torch.full((size,), 17, dtype=torch.uint8, device=device)
        dst = torch.empty_like(src)
        timing = measure(partial(dst.copy_, src), lambda: None, iterations, f"roofline-copy/{mib}MiB")
        torch.testing.assert_close(dst[:: size - 1].cpu(), torch.full((2,), 17, dtype=torch.uint8))
        copy_cases.append({**timing, "bytes": size * 2, "gbs": size * 2 / (timing["p50_us"] * 1e3)})
        del src, dst

    cube_cases = []
    for size in (4096, 8192):
        a = torch.full((size, size), 0.125, dtype=torch.bfloat16, device=device)
        b = torch.full_like(a, 0.25)
        out = torch.empty_like(a)
        timing = measure(partial(torch.mm, a, b, out=out), lambda: None, iterations, f"roofline-cube/{size}")
        torch.testing.assert_close(out[0, :: size - 1].float().cpu(), torch.full((2,), size / 32))
        flops = 2 * size**3
        cube_cases.append({**timing, "dimension": size, "flops": flops, "tflops": flops / (timing["p50_us"] * 1e6)})
        del a, b, out

    vector_cases = []
    out = torch.empty((40, 4, 4096), dtype=torch.float32, device=device)
    expected = torch.arange(4, dtype=torch.float32).view(1, 4, 1).expand(out.shape)
    op = torch.ops.replayssm_mix_910b.roofline_vector_
    for repeats in (128, 256, 512):
        timing = measure(partial(op, out, repeats), lambda: None, iterations, f"roofline-aiv/{repeats}")
        torch.testing.assert_close(out.cpu(), expected + repeats / 512, atol=0, rtol=0)
        flops = 2 * out.numel() * repeats
        vector_cases.append({**timing, "repeats": repeats, "flops": flops, "tflops": flops / (timing["p50_us"] * 1e6)})

    return {
        "bandwidth_gbs": max(case["gbs"] for case in copy_cases),
        "cube_tflops": max(case["tflops"] for case in cube_cases),
        "vector_tflops": max(case["tflops"] for case in vector_cases),
        "copy": copy_cases,
        "cube": cube_cases,
        "vector": vector_cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.output.exists():
        parser.error("iterations must be positive and output must be a fresh path")
    import torch_npu  # noqa: F401

    library = args.library.resolve(strict=True)
    torch.npu.set_device(args.device)
    torch.ops.load_library(str(library))
    if not hasattr(torch.ops.replayssm_mix_910b, "roofline_vector_"):
        parser.error("build the library with -DREPLAYSSM_ROOFLINE_CALIBRATION=ON")

    with torch.inference_mode():
        result = calibrate(torch.device(f"npu:{args.device}"), args.iterations)
    result.update(
        status="PASSED",
        soc=torch.npu.get_device_name(args.device),
        iterations=args.iterations,
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        packages=package_versions(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

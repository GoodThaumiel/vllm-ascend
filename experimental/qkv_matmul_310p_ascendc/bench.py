# SPDX-License-Identifier: Apache-2.0
"""Compare the 310P QKV projection with the native torch_npu linear path."""

import argparse
import json
import os
import statistics
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu

K = 2048
N = 6144
C0 = 16
NZ = 29
ND = 2
MAX_ROWS = 4096
PROFILE_ROWS = (3981, 4001, 4080, 4095, 4096)
TAIL_ROWS = (1, 15, 16, 17, 255, 256, 257, 3981, 4095, 4096)
DEFAULT_LIBRARY = Path(__file__).parent / "build-c921" / "libqkv_matmul_310p_torch.so"


def pack_nz_cpu(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    padded_rows = (rows + C0 - 1) // C0 * C0
    padded = F.pad(x, (0, 0, 0, padded_rows - rows))
    return padded.reshape(padded_rows // C0, C0, cols // C0, C0).permute(2, 0, 1, 3).contiguous()


def layout_cases(device: torch.device, cores: int, fp32_accum: bool) -> list[dict]:
    """Exercise exact NZ indexing, all 16 K passes and partial row writeback."""
    k = torch.arange(K)
    n = torch.arange(N)
    weight = (((n[:, None] * 7 + k * 13) % 29 - 14) / 16).half()
    weight_nz = torch_npu.npu_format_cast(weight.to(device), NZ)
    packed_weight = pack_nz_cpu(weight).to(device)
    cases = []
    for rows in TAIL_ROWS:
        r = torch.arange(rows)
        selected_k = (r * 73 + 127) % K
        next_k = (selected_k + 128) % K
        sign = (1 - 2 * (r % 2)).half()
        x = torch.zeros(rows, K, dtype=torch.float16)
        x[r, selected_k] = sign
        x[r, next_k] = -sign
        expected = (weight[:, selected_k].T - weight[:, next_k].T) * sign[:, None]
        # Rows before and after the output are independently checked on CPU.
        backing = torch.full((rows + 2, N), -17, dtype=torch.float16, device=device)
        out = backing[1:-1]
        for name, x_nz, w_nz in (
            ("fractal_nz", torch_npu.npu_format_cast(x.to(device), NZ), weight_nz),
            ("packed", pack_nz_cpu(x).to(device), packed_weight),
        ):
            x_format = int(torch_npu.get_npu_format(x_nz))
            w_format = int(torch_npu.get_npu_format(w_nz))
            print(f"check/layout M={rows} input={name} formats={x_format}/{w_format}", flush=True)
            out.fill_(float("nan"))
            torch.ops.qkv_matmul_310p.run_(x_nz, w_nz, out, cores, fp32_accum)
            actual = backing.cpu()
            torch.testing.assert_close(actual[1:-1], expected, atol=0, rtol=0)
            torch.testing.assert_close(actual[[0, -1]], torch.full((2, N), -17, dtype=x.dtype), atol=0, rtol=0)
            cases.append(
                {
                    "rows": rows,
                    "input": name,
                    "input_formats": [x_format, w_format],
                    "exact": True,
                    "guard_unchanged": True,
                }
            )
    return cases


def check_error(actual: torch.Tensor, expected: torch.Tensor, name: str) -> dict:
    delta = actual.float() - expected.float()
    relative_l2 = (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected.float())).item()
    max_abs = delta.abs().max().item()
    print(f"  {name} relative_l2={relative_l2:.6g} max_abs={max_abs:.6g}", flush=True)
    torch.testing.assert_close(actual, expected, atol=4e-3, rtol=1e-2, msg=lambda details: f"{name}\n{details}")
    if relative_l2 > 4e-3:
        raise AssertionError(f"{name} relative L2 {relative_l2:.6g} exceeds 0.004")
    return {"relative_l2": relative_l2, "max_abs": max_abs}


def run_perf(call, warmup: int, repeats: int, *, timing: bool) -> dict:
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    result = {"warmup": warmup, "repeats": repeats}
    if not timing:
        for _ in range(repeats):
            call()
        torch.npu.synchronize()
        return result

    events = [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for _ in range(repeats)]
    for start, end in events:
        start.record()
        call()
        end.record()
    torch.npu.synchronize()
    times = sorted(start.elapsed_time(end) * 1000 for start, end in events)
    p90_index = (9 * repeats + 9) // 10 - 1
    return {
        **result,
        "p50_us": statistics.median(times),
        "p90_us": times[p90_index],
        "min_us": times[0],
    }


def custom_linear(x: torch.Tensor, weight: torch.Tensor, cores: int, fp32_accum: bool, *, pack: bool) -> torch.Tensor:
    out = torch.empty((x.shape[0], N), device=x.device, dtype=torch.float16)
    x_nz = torch_npu.npu_format_cast(x, NZ) if pack else x
    torch.ops.qkv_matmul_310p.run_(x_nz, weight, out, cores, fp32_accum)
    return out


def run_rows(args: argparse.Namespace, device: torch.device) -> list[dict]:
    implementations = ("native", "custom") if args.impl == "both" else (args.impl,)
    fp32_accum = args.accum_dtype == "fp32"
    need_nz = "custom" in implementations or ("native" in implementations and args.native_input_format == "nz")
    generator = torch.Generator().manual_seed(args.seed)
    weight_cpu = (torch.randn(N, K, generator=generator) * K**-0.5).half()
    weight_nz = torch_npu.npu_format_cast(weight_cpu.to(device), NZ)
    cases = []
    for rows in args.rows:
        input_generator = torch.Generator().manual_seed(args.seed + rows)
        x_cpu = torch.randn(rows, K, generator=input_generator, dtype=torch.float16)
        x = x_cpu.to(device)
        x_nz = torch_npu.npu_format_cast(x, NZ) if need_nz else x
        calls = {
            "native": partial(F.linear, x_nz if args.native_input_format == "nz" else x, weight_nz),
            "custom": partial(custom_linear, x_nz, weight_nz, args.cores, fp32_accum, pack=False),
            "custom-from-nd": partial(custom_linear, x, weight_nz, args.cores, fp32_accum, pack=True),
        }
        case = {"rows": rows, "output_formats": {}}
        outputs = {}
        for name in implementations:
            print(f"{args.stage}/{name} M={rows}", flush=True)
            call = calls[name]
            out = call()
            torch.npu.synchronize()
            out_format = int(torch_npu.get_npu_format(out))
            allowed_formats = (ND, NZ) if name == "native" else (ND,)
            if out.shape != (rows, N) or out.dtype != torch.float16 or out_format not in allowed_formats:
                raise RuntimeError(f"{name} produced shape={out.shape}, dtype={out.dtype}, format={out_format}")
            case["output_formats"][name] = out_format
            print(f"  {name} output_format={out_format}", flush=True)
            if args.stage == "perf":
                case[name] = run_perf(call, args.warmup, args.repeats, timing=not args.no_timing)
                if not args.no_timing:
                    case[name]["tflops_p50"] = 2 * rows * K * N / (case[name]["p50_us"] * 1e6)
                print(json.dumps(case[name]), flush=True)
            else:
                outputs[name] = out.cpu()
        if args.stage == "check":
            ref = F.linear(x_cpu.float(), weight_cpu.float()).half()
            for name, actual in outputs.items():
                case[name] = check_error(actual, ref, f"{name} vs FP32")
            if args.impl == "both":
                case["custom_vs_native"] = check_error(outputs["custom"], outputs["native"], "custom vs native")
        cases.append(case)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--stage", choices=("check", "perf"), default="check")
    parser.add_argument("--impl", choices=("both", "native", "custom", "custom-from-nd"), default="both")
    parser.add_argument("--native-input-format", choices=("nd", "nz"), default="nd")
    parser.add_argument(
        "--accum-dtype",
        choices=("fp32", "fp16"),
        default="fp32",
        help="L0C accumulator for the custom kernel (outputs always FP16)",
    )
    parser.add_argument("--rows", nargs="+", type=int, default=PROFILE_ROWS)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--cores", type=int, default=0, help="0 uses all available Cube cores")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--no-timing", action="store_true", help="Submit perf iterations for msprof without NPU events")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.rows) < 1 or max(args.rows) > MAX_ROWS:
        parser.error(f"--rows must be in [1,{MAX_ROWS}]")
    if args.cores < 0 or args.warmup < 0 or args.repeats < 1:
        parser.error("--cores and --warmup must be nonnegative and --repeats must be positive")

    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = True
    library = None
    if args.impl != "native":
        library = str(args.library.resolve(strict=True))
        torch.ops.load_library(library)

    report = {
        "device": torch.npu.get_device_name(args.device),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "ascend_home_path": os.getenv("ASCEND_HOME_PATH"),
        "library": library,
        "stage": args.stage,
        "implementation": args.impl,
        "native_input_format": args.native_input_format,
        "custom_accum_dtype": args.accum_dtype if args.impl != "native" else None,
        "cores": args.cores,
        "seed": args.seed,
    }
    if args.stage == "perf":
        report["timing"] = "disabled" if args.no_timing else "npu_event"
    with torch.inference_mode():
        if args.stage == "check" and args.impl != "native":
            report["layout"] = layout_cases(device, args.cores, args.accum_dtype == "fp32")
        report["cases"] = run_rows(args, device)
    report["status"] = "PASSED"
    result = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()

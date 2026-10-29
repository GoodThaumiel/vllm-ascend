# SPDX-License-Identifier: Apache-2.0
"""Compare tied and request-owned SD ring addressing on one NPU.

The two paths use identical inputs. Checkpoints live at different physical
block ids while d/k/g stay in the same small request-slot allocation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experimental.replayssm_sd_ascendc.lab import clone_page_strided_case, make_case  # noqa: E402


def run(tp: int, device: int) -> dict:
    import torch_npu  # noqa: F401

    from vllm_ascend.ops.replayssm.ascendc import replayssm_sd_compact, replayssm_sd_verify
    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(device)
    if not enable_custom_op():
        raise RuntimeError("ReplaySSM SD operators unavailable; rebuild the extension")
    npu = torch.device(f"npu:{device}")
    report = {"status": "RUNNING", "tp": tp, "device": device, "cases": []}
    for position, base, flush in ((0, 0, 0), (7, 31, 0), (12, 31, 1), (16, 31, 1)):
        source = make_case(tp=tp, batch=2, position=position, length=4, ring_len=32, tile_v=128, base=base)
        tied = clone_page_strided_case(source, npu)
        detached = {name: tensor.clone() for name, tensor in tied.items() if name not in ("checkpoint", "slots")}
        detached["checkpoint"] = torch.zeros((17, *source["checkpoint"].shape[1:]), dtype=torch.bfloat16, device=npu)
        detached["slots"] = torch.tensor([11, 12], dtype=torch.int32, device=npu)
        ring_slots = tied["slots"].clone()
        for old_slot, new_slot in ((1, 11), (2, 12)):
            detached["checkpoint"][new_slot].copy_(tied["checkpoint"][old_slot])

        for case, request_slots in ((tied, None), (detached, ring_slots)):
            replayssm_sd_verify(
                *(case[name] for name in ("q", "k", "v", "g", "beta", "checkpoint", "d_cache", "k_cache", "g_cache",
                                          "slots", "positions", "bases", "lengths", "out", "status")),
                128**-0.5,
                128,
                40,
                13,
                request_slots,
            )
            case["flush"] = torch.full((2,), flush, dtype=torch.int32, device=npu)
            case["compact_status"] = torch.empty((2 * (48 // tp) * 4,), dtype=torch.int32, device=npu)
            replayssm_sd_compact(
                *(case[name] for name in ("checkpoint", "d_cache", "k_cache", "g_cache", "slots", "positions",
                                          "bases", "flush", "compact_status")),
                32,
                40,
                1,
                request_slots,
            )
        torch.npu.synchronize()
        for name in ("out", "status", "compact_status", "d_cache", "k_cache", "g_cache"):
            torch.testing.assert_close(detached[name].cpu(), tied[name].cpu(), atol=0, rtol=0, equal_nan=True)
        for old_slot, new_slot in ((1, 11), (2, 12)):
            torch.testing.assert_close(
                detached["checkpoint"][new_slot].cpu(), tied["checkpoint"][old_slot].cpu(), atol=0, rtol=0
            )
        if bool(detached["status"].cpu().abs().any()) or bool(detached["compact_status"].cpu().abs().any()):
            raise AssertionError("ReplaySSM SD returned a nonzero status")
        report["cases"].append({"position": position, "base": base, "flush": flush, "status": "PASSED"})
    report["status"] = "PASSED"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, choices=(4, 8), default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    try:
        report.update(run(args.tp, args.device))
    except Exception as exc:
        report.update({"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

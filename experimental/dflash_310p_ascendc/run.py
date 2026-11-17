# SPDX-License-Identifier: Apache-2.0
"""Run the standalone DFlash input-preparation op on Ascend310P3."""

import argparse
import json
import traceback
from pathlib import Path

from .reference import LANES, QUERY_FIELDS, cases, prepare_reference

GUARD = -20261116
MASK_ID = 248077
MAX_LEN = 32


def run_device(args, report):
    import torch
    import torch_npu

    torch.npu.set_device(args.device)
    torch.ops.load_library(str(args.library.resolve()))
    inputs = cases()
    requests, positions, tables, _ = inputs["mixed"]
    groups = len(tables)
    batch = len(requests)
    device = f"npu:{args.device}"

    def upload(value, device=device):
        return torch.tensor(value, dtype=torch.int32, device=device)

    def guarded(shape):
        count = 1
        for dim in shape:
            count *= dim
        backing = torch.full((count + 2 * LANES,), GUARD, dtype=torch.int32, device=device)
        return backing, backing[LANES:-LANES].view(shape)

    prepared = [upload(values) for values in inputs["mixed"]]
    query_guard, query = guarded((groups, batch, QUERY_FIELDS, LANES))
    context_guard, context = guarded((groups, 2, len(positions)))
    op = torch.ops.dflash_310p.prepare_

    def launch(spec, anchor):
        op(*prepared, query, context, spec, MAX_LEN, MASK_ID, anchor)

    def check(name, spec, anchor, graph=None):
        for target, values in zip(prepared, inputs[name]):
            target.copy_(upload(values))
        if graph is None:
            launch(spec, anchor)
        else:
            graph.replay()
        torch.npu.synchronize()
        expected_query, expected_context = prepare_reference(*inputs[name], spec, MAX_LEN, MASK_ID, anchor)
        torch.testing.assert_close(query.cpu(), upload(expected_query, "cpu"), rtol=0, atol=0)
        torch.testing.assert_close(context.cpu(), upload(expected_context, "cpu"), rtol=0, atol=0)
        for backing in (query_guard, context_guard):
            for edge in (backing[:LANES], backing[-LANES:]):
                torch.testing.assert_close(edge.cpu(), upload([GUARD] * LANES, "cpu"), rtol=0, atol=0)
        report["cases"].append({"case": name, "k": spec, "anchor": anchor, "status": "PASSED"})

    report.update(torch=torch.__version__, torch_npu=torch_npu.__version__)
    for spec in args.k:
        for anchor in (False, True):
            launch(spec, anchor)
            torch.npu.synchronize()
            graph = None
            if args.stage == "graph":
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    launch(spec, anchor)
            for name in (*inputs, "mixed"):
                check(name, spec, anchor, graph)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library",
        type=Path,
        default=Path(__file__).parent / "build/libdflash_310p_torch.so",
    )
    parser.add_argument("--stage", choices=("eager", "graph"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--k", type=int, choices=range(1, 8), nargs="+", default=[7])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"stage": args.stage, "device": args.device, "library": str(args.library.resolve()), "cases": []}
    try:
        run_device(args, report)
    except Exception:
        report.update(status="FAILED", traceback=traceback.format_exc())
        raise
    else:
        report["status"] = "PASSED"
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

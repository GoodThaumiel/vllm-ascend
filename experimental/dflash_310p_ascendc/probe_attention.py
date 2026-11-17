# SPDX-License-Identifier: Apache-2.0
"""Probe explicit full and sliding masks with native 310P paged SplitFuse."""

import argparse
import json
import traceback
from pathlib import Path


def run(args, report):
    import torch
    import torch_npu

    from vllm_ascend._310p.attention.dflash_splitfuse import DFlashSplitFuse310, draft_kv_group_order
    from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

    device = torch.device("npu", args.device)
    torch.npu.set_device(args.device)
    page_size, pages_per_row, rows = 128, 2, 2
    width, heads, kv_heads, head_dim = args.k + 1, 16, 4, 128
    pages = pages_per_row * rows
    scale = head_dim**-0.5
    generator = torch.Generator().manual_seed(310)
    query_cpu = torch.randn(rows * width, heads, head_dim, generator=generator).half()
    shape = (pages, page_size, kv_heads, head_dim)
    keys, values = (torch.randn(shape, generator=generator).half() for _ in range(2))

    def pack_cache(tensor):
        packed = (
            tensor.reshape(pages, page_size, kv_heads * head_dim // 16, 16).permute(0, 2, 1, 3).contiguous().to(device)
        )
        return torch_npu.npu_format_cast(packed, ACL_FORMAT_FRACTAL_NZ)

    key_cache, value_cache = pack_cache(keys), pack_cache(values)
    query = query_cpu.to(device)

    def expected(lengths, block_ids, causal, window):
        results = []
        for row, length in enumerate(lengths):
            k, v = [
                tensor[block_ids[row]].reshape(-1, kv_heads, head_dim)[:length].float() for tensor in (keys, values)
            ]
            q = query_cpu[row * width : (row + 1) * width].float()
            q = q.view(width, kv_heads, heads // kv_heads, head_dim)
            logits = torch.einsum("qkgd,skd->kgqs", q, k) * scale
            qpos = torch.arange(length - width, length).unsqueeze(1)
            kpos = torch.arange(length).unsqueeze(0)
            valid = torch.ones(width, length, dtype=torch.bool)
            if causal:
                valid &= kpos <= qpos
            if window is not None:
                valid &= kpos >= qpos - window + 1
                if not causal:
                    valid &= kpos <= qpos + window - 1
            probs = logits.masked_fill(~valid, float("-inf")).softmax(dim=-1)
            result = torch.einsum("kgqs,skd->qkgd", probs, v).reshape(width, heads, head_dim)
            results.append(result.half())
        return torch.cat(results)

    policies = {"full_noncausal": (False, None), "sliding_causal": (True, 32)}
    layers = list(reversed(policies))
    group_ids, layer_to_index = draft_kv_group_order(
        layers,
        [["target.full", "full_noncausal"], ["target.gdn"], ["sliding_causal"]],
    )
    buffers = [
        DFlashSplitFuse310(
            rows=rows,
            query_width=width,
            table_width=pages_per_row,
            page_size=page_size,
            device=device,
            causal=causal,
            window=window,
        )
        for causal, window in policies.values()
    ]
    outputs = {name: torch.empty_like(query) for name in layers}

    def forward():
        for name, index in zip(layers, layer_to_index):
            buffers[index].forward(
                query,
                key_cache,
                value_cache,
                num_kv_heads=kv_heads,
                num_heads=heads,
                scale=scale,
                output=outputs[name],
            )

    graph = None
    for lengths, block_ids in (([width + 4, 139], [[0, 1], [2, 3]]), ([142, 44], [[3, 2], [1, 0]])):
        tables = [block_ids, block_ids[::-1]]
        assert len(group_ids) == len(tables)
        for metadata, table in zip(buffers, tables):
            metadata.refresh(torch.tensor(lengths, dtype=torch.int32), torch.tensor(table, dtype=torch.int32))
        if args.stage == "graph":
            if graph is None:
                forward()
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    forward()
            graph.replay()
        else:
            forward()
        torch.npu.synchronize()
        for name, index in zip(layers, layer_to_index):
            causal, window = policies[name]
            actual = outputs[name].cpu()
            ref = expected(lengths, tables[index], causal, window)
            torch.testing.assert_close(actual, ref, rtol=0.01, atol=0.002)
            report["cases"].append(
                {
                    "case": name,
                    "group_id": group_ids[index],
                    "lengths": lengths,
                    "max_abs": (actual - ref).abs().max().item(),
                }
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("eager", "graph"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--k", type=int, choices=range(1, 8), default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"stage": args.stage, "device": args.device, "k": args.k, "cases": []}
    try:
        run(args, report)
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

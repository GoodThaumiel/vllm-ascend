# SPDX-License-Identifier: Apache-2.0
"""Probe explicit full and sliding masks with native 310P paged SplitFuse."""

import argparse
import json
import traceback
from pathlib import Path


def run(args, report):
    import torch
    import torch_npu

    from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, nd_to_nz_spec

    device = f"npu:{args.device}"
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
    output = torch.empty_like(query)
    qlens = torch.full((rows,), width, dtype=torch.int32)
    seq = torch.empty(rows, dtype=torch.int32, device=device)
    table = torch.empty((rows, pages_per_row), dtype=torch.int32, device=device)
    columns = torch.arange(page_size * pages_per_row, dtype=torch.int32, device=device).reshape(1, 1, -1)
    query_offsets = torch.arange(width, dtype=torch.int32, device=device).reshape(1, -1, 1)
    mask_nd = torch.zeros((rows * width, columns.numel()), dtype=torch.float16, device=device)
    mask_buffer = torch_npu.npu_format_cast(nd_to_nz_spec(mask_nd).contiguous(), ACL_FORMAT_FRACTAL_NZ)

    def update_mask(causal, window):
        lengths = seq[:, None, None]
        positions = lengths - width + query_offsets
        visible = columns < lengths
        if causal:
            visible = visible & (columns <= positions)
        if window is not None:
            visible = visible & (columns >= positions - window + 1)
            if not causal:
                visible = visible & (columns <= positions + window - 1)
        mask = (~visible.expand(rows, width, -1)).reshape(rows * width, -1).to(torch.float16) * -10000.0
        mask_buffer.copy_(torch_npu.npu_format_cast(nd_to_nz_spec(mask).contiguous(), ACL_FORMAT_FRACTAL_NZ))

    def forward():
        torch_npu._npu_paged_attention_splitfuse(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            mask=mask_buffer,
            block_table=table,
            seq_len=qlens,
            context_lens=seq,
            num_kv_heads=kv_heads,
            num_heads=heads,
            scale_value=scale,
            out=output,
        )

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

    def update(lengths, block_ids):
        seq.copy_(torch.tensor(lengths, dtype=torch.int32))
        table.copy_(torch.tensor(block_ids, dtype=torch.int32))

    for name, causal, window in (("full_noncausal", False, None), ("sliding_causal", True, 32)):
        graph = None
        for lengths, block_ids in (([width + 4, 139], [[0, 1], [2, 3]]), ([142, 44], [[3, 2], [1, 0]])):
            update(lengths, block_ids)
            update_mask(causal, window)
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
            actual = output.cpu()
            ref = expected(lengths, block_ids, causal, window)
            torch.testing.assert_close(actual, ref, rtol=0.01, atol=0.002)
            report["cases"].append({"case": name, "lengths": lengths, "max_abs": (actual - ref).abs().max().item()})


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

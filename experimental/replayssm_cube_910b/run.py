# SPDX-License-Identifier: Apache-2.0
"""One same-input graph benchmark. Use --path to isolate msprof captures."""

import argparse
import hashlib
import importlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experimental.replayssm_cube_910b.projections import (  # noqa: E402
    checkpoint_projection,
    compact,
    compact_inputs,
    cube_fold_operands,
    panel,
    verify,
)
from experimental.replayssm_sd_ascendc.lab import make_case, recurrent_golden  # noqa: E402
from scripts.replayssm.profile_sd_operator_compare import (  # noqa: E402
    native_state_page,
    package_versions,
    reconstruct_native_start,
)

VECTOR_PATHS = ("native", "v20", "compact_v1", "vector_pair")
MIXED_PATHS = ("mixed_verify", "mixed_pair")
MIXED_FUSED_FOLD_PATH = "mixed_fused_fold"
MIXED_FOLD_FMA_PATH = "mixed_fold_fma"
MIXED_FOLD_PAIR_PATH = "mixed_fold_pair"
MIXED_COLUMN_RECURRENCE_PATH = "mixed_column_recurrence"
IN_KERNEL_FOLD_PATHS = (MIXED_FUSED_FOLD_PATH, MIXED_FOLD_FMA_PATH, MIXED_FOLD_PAIR_PATH, MIXED_COLUMN_RECURRENCE_PATH)
MIXED_AIV_ABLATIONS = {
    "mixed_history_panel": 1,
    "mixed_triangular": 2,
    "mixed_suffix_reference": 3,
    MIXED_FUSED_FOLD_PATH: 4,
    MIXED_FOLD_FMA_PATH: 5,
    MIXED_FOLD_PAIR_PATH: 6,
    MIXED_COLUMN_RECURRENCE_PATH: 7,
}
MIXED_CUBE_FOLD_PATH = "mixed_cube_fold"
CUBE_FOLD_ONLY_PATHS = ("cube_fold_path", "cube_fold_gemm")
MIXED_FOLD_PATHS = (
    "mixed_packed_compact_v5",
    "mixed_packed_compact_v6",
    *MIXED_AIV_ABLATIONS,
    MIXED_CUBE_FOLD_PATH,
)
PACKED_PANEL_PATHS = ("mixed_packed", *MIXED_FOLD_PATHS)
ALL_MIXED_PATHS = (*MIXED_PATHS, *PACKED_PANEL_PATHS)
CUBE_PATHS = (
    "cube_verify",
    "cube_compact",
    "projection",
    "grouped_projection",
    "fold_gemm",
    "cube_pair",
)
PATHS = (*VECTOR_PATHS, *CUBE_PATHS)
EXTRA_PATHS = ("v13", "v13_pair", "cursor")
COMPACT_ONLY_PATHS = ("compact_v5", "compact_v6")
COMPACT_PATHS = (*COMPACT_ONLY_PATHS, "v20_compact_v5", "v23_compact_v5", *MIXED_FOLD_PATHS)
WIDE_PATHS = ("v23", "v23_compact_v5")
FUSED_PATH = "fused_v24"
FUSED_PACK_PATH = "fused_v24_with_pack"
FUSED_FOLD_FIRST_PATH = "fused_v25"
FUSED_PATHS = (FUSED_PATH, FUSED_PACK_PATH, FUSED_FOLD_FIRST_PATH)
STACK_PATHS = ("native", "vector_pair", "v20_compact_v5", "v23_compact_v5", *MIXED_FOLD_PATHS, *FUSED_PATHS)
DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
PROJECTION_ATOL = 2e-5
PROJECTION_RTOL = 1e-3


def batches(text):
    if text == "all":
        return tuple(range(1, 129))
    try:
        values = tuple(int(x) for x in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated batches in 1..256, or all") from exc
    if not values or min(values) < 1 or max(values) > 256 or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("expected unique batches in 1..256, or all")
    return values


def measure(call, reset, iterations, label, check_graph=None):
    for _ in range(2):
        reset()
        call()
    reset()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        call()
    # The first launches of a newly captured Ascend graph can have a
    # large setup cost even after eager warmup. Measure steady replay.
    graph_warmups = 2
    for _ in range(graph_warmups):
        reset()
        graph.replay()
    torch.npu.synchronize()
    us = []
    print(f"MEASURE_BEGIN {label} replays={iterations}", flush=True)
    with torch.profiler.record_function(f"REPLAYSSM_SD/{label}"):
        for _ in range(iterations):
            reset()
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            us.append(start.elapsed_time(end) * 1000)
    print(f"MEASURE_END {label}", flush=True)
    if check_graph is not None:
        check_graph(graph)
    return {
        "graph_us": us,
        "graph_warmup_replays": graph_warmups,
        "p50_us": statistics.median(us),
        "p90_us": sorted(us)[(9 * len(us) - 1) // 10],
    }


def measure_eager(call, reset, iterations):
    for _ in range(3):
        reset()
        call()
    torch.npu.synchronize()
    event_us, complete_us = [], []
    for _ in range(iterations):
        reset()
        torch.npu.synchronize()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        host_start = time.perf_counter_ns()
        start.record()
        call()
        end.record()
        end.synchronize()
        complete_us.append((time.perf_counter_ns() - host_start) / 1000)
        event_us.append(start.elapsed_time(end) * 1000)
    return {
        "event_us": event_us,
        "event_p50_us": statistics.median(event_us),
        "host_to_complete_p50_us": statistics.median(complete_us),
    }


def last_projections(case, blocks, balanced=False, panel_tokens=4):
    """Reference the last head produced in each AIC's reusable workspace slot."""
    batch, _, hk, _ = case["k"].shape
    hv = case["v"].shape[2]
    if balanced:
        last = (torch.arange(1, blocks + 1) * (batch * hv)) // blocks - 1
        return reference_projections(case, last // hv, last % hv, panel_tokens)
    group = hv // hk
    first = torch.arange(blocks)
    tasks = first + ((batch * hk - 1 - first) // blocks) * blocks
    rows, keys = tasks // hk, tasks % hk
    return reference_projections(case, rows, keys * group + group - 1, panel_tokens)


def reference_projections(case, rows, heads, panel_tokens=4):
    """Build FP32 C tiles for the requested (row, V head) pairs."""
    group = case["v"].shape[2] // case["k"].shape[2]
    keys = heads // group
    checkpoint = case["checkpoint"][rows + 1, heads].float()
    rhs = torch.zeros((rows.numel(), 128, 16), dtype=torch.float32)
    width = case["k"].size(1)
    rhs[:, :, :width] = case["k"][rows, :, keys].transpose(1, 2)
    rhs[:, :, panel_tokens : panel_tokens + width] = case["q"][rows, :, keys].transpose(1, 2)
    return torch.bmm(checkpoint, rhs)


def projection_error_summary(actual, expected, panel_tokens=4):
    """Locate bad lanes in each workspace tile and count zeros and nonfinite values."""
    bad = ~torch.isclose(actual, expected, atol=PROJECTION_ATOL, rtol=PROJECTION_RTOL)
    summary = {}
    for part, cols in (
        ("k", slice(0, panel_tokens)),
        ("q", slice(panel_tokens, 2 * panel_tokens)),
        ("padding", slice(2 * panel_tokens, 16)),
    ):
        errors = bad[..., cols]
        if errors.size(-1) == 0:
            continue
        values = actual[..., cols]
        tiles = errors.reshape(-1, 2, 64, errors.size(-1)).sum(dim=(2, 3))
        indices = tiles.nonzero().tolist()
        summary[part] = dict(
            mismatches=int(tiles.sum()),
            zero_values=int((errors & (values == 0)).sum()),
            nonfinite_values=int((errors & ~values.isfinite()).sum()),
            tile_halves=[(tile, half, int(tiles[tile, half])) for tile, half in indices[:8]],
            affected_tile_halves=len(indices),
        )
    return summary


def assert_stage(actual, expected, stage, *, atol, rtol, dump=None, equal_nan=False):
    """Preserve the failing tensor and identify the stage in the traceback."""
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, equal_nan=equal_nan)
    except AssertionError as exc:
        if dump is not None:
            dump.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                dict(stage=stage, actual=actual, expected=expected, atol=atol, rtol=rtol, equal_nan=equal_nan), dump
            )
        raise AssertionError(f"{stage} ({tuple(actual.shape)}): {exc}") from None


def select_flush_rows(batch, count, layout, seed):
    if count == 0 or layout == "prefix":
        return torch.arange(count)
    if layout == "spread":
        return torch.arange(count) * batch // count
    return torch.randperm(batch, generator=torch.Generator().manual_seed(seed))[:count].sort().values


def native_start_with_mixed_history(case):
    """Roll each request's committed history into the native BF16 entry state."""
    positions = case["positions"]
    state = case["checkpoint"][1:].float()
    group = case["v"].size(2) // case["k"].size(2)
    base = int(case["bases"][0])
    for i in range(int(positions.max())):
        rows = (positions > i).nonzero().flatten()
        slots = rows + 1
        physical = (base + i) & 31
        alpha = case["g_cache"][slots, :, physical].exp()
        delta = case["d_cache"][slots, :, physical].float()
        key = case["k_cache"][slots, :, physical].float().repeat_interleave(group, dim=1)
        state[rows] = alpha[..., None, None] * state[rows] + delta[..., None] * key[..., None, :]
    return state.bfloat16()


def trial(args, b, dev):
    from vllm_ascend.ops.gdn import recurrent_gated_delta_rule
    from vllm_ascend.ops.replayssm.ascendc import replayssm_sd_compact, replayssm_sd_verify

    p, t, base = args.pos, args.length, args.base
    wide = t > 4
    active = {"none": 0, "mixed": max(1, b // 8), "full": b}[args.flush]
    if args.flush_count is not None:
        active = args.flush_count
    flush_rows = select_flush_rows(b, active, args.flush_layout, args.flush_seed)
    nonflush_pos = p if args.nonflush_pos is None else args.nonflush_pos
    cpu = make_case(
        geometry=args.geometry,
        tp=args.tp,
        batch=b,
        position=max(p, nonflush_pos),
        length=t,
        base=base,
        tile_v=128,
        seed=42,
        window=t if wide else 4,
    )
    cpu["positions"].fill_(nonflush_pos)
    cpu["positions"][flush_rows] = p
    key_heads, value_heads = cpu["k"].shape[2], cpu["v"].shape[2]
    cpu["beta"] = cpu["beta"].bfloat16().float()
    c = {n: v.to(dev) for n, v in cpu.items()}
    c["checkpoint"] = torch.cat((torch.zeros_like(c["checkpoint"][:1]).expand(4, -1, -1, -1), c["checkpoint"]))
    initial = c["checkpoint"].clone()
    ring = c["slots"]
    slots = ring + 4
    mask = torch.zeros(b, dtype=torch.int32)
    mask[flush_rows] = 1
    flush = mask.to(dev)
    restore_slots = (flush_rows + 5).to(dev)
    restore_values = None
    if p and active:
        restore_values = (
            initial[5 : 5 + active] if args.flush_layout == "prefix" else initial.index_select(0, restore_slots)
        )
    status = torch.full((b * value_heads * 4,), -1, dtype=torch.int32, device=dev)
    state, state_page_bytes = native_state_page(dev, args.tp, b, t, geometry=args.geometry)
    start_cpu = reconstruct_native_start(cpu, p) if p == nonflush_pos else native_start_with_mixed_history(cpu)
    start_state = start_cpu.to(dev)
    first = state[1::t]
    indices = torch.arange(1, b * t + 1, device=dev, dtype=torch.int32)
    lengths = torch.tensor([0] + [t] * b, device=dev, dtype=torch.int32)
    accepted = torch.ones(b, device=dev, dtype=torch.int32)
    inputs = [c[n][:, :t].contiguous().flatten(0, 1) for n in ("q", "k", "v", "g", "beta")]

    def native(state=state):
        return recurrent_gated_delta_rule(
            *inputs[:3],
            state,
            g=inputs[3],
            beta=inputs[4],
            scale=128**-0.5,
            actual_seq_lengths=lengths,
            ssm_state_indices=indices,
            num_accepted_tokens=accepted,
        )

    inputs[4] = inputs[4].bfloat16()

    def vector(variant=20, cache=c):
        verify_op = torch.ops.replayssm_sd_lab.verify_ if variant == 23 else replayssm_sd_verify
        verify_op(
            *(cache[n] for n in ("q", "k", "v", "g", "beta", "checkpoint", "d_cache", "k_cache", "g_cache")),
            slots,
            c["positions"],
            c["bases"],
            c["lengths"],
            c["out"],
            c["status"],
            128**-0.5,
            128,
            40,
            variant,
            ring,
        )

    def fold(cache=c):
        replayssm_sd_compact(
            cache["checkpoint"],
            cache["d_cache"],
            cache["k_cache"],
            cache["g_cache"],
            slots,
            c["positions"],
            c["bases"],
            flush,
            status,
            32,
            40,
            1,
            ring,
        )

    def reset_state(cache=c):
        # Restore the same starting checkpoint outside the timed region.
        if restore_values is not None:
            if args.flush_layout == "prefix":
                cache["checkpoint"][5 : 5 + active].copy_(restore_values)
            else:
                cache["checkpoint"].index_copy_(0, restore_slots, restore_values)

    # Native materializes the committed ring in its initial state. Replay reads
    # the same history directly and writes its candidates into disjoint slots.
    wide_golden = recurrent_golden(cpu) if wide else None
    if wide_golden is None:
        vector()
        if c["status"].cpu().any():
            raise RuntimeError("V20 returned nonzero status")
        expected_out = c["out"][:, :t].float().cpu()
        fold()
        if status.cpu().any():
            raise RuntimeError("Compact V1 returned nonzero status")
        checkpoint = c["checkpoint"].cpu()
    else:
        expected_out = wide_golden["out"].float()
        checkpoint = initial.cpu()
        if p and active:
            checkpoint[flush_rows + 5] = start_cpu[flush_rows]
    first.copy_(start_state)
    y_native = native().view_as(expected_out).float().cpu()
    torch.testing.assert_close(y_native, expected_out, atol=0.01, rtol=0.03)
    errors = {"native_vs_dense_out" if wide else "native_vs_v20_out": (y_native - expected_out).abs().max().item()}
    unchanged = torch.ones(b + 1, dtype=torch.bool)
    if p and active:
        changed = flush_rows + 1
        torch.testing.assert_close(checkpoint[changed + 4], start_cpu[flush_rows], atol=0.001, rtol=0.015)
        unchanged[changed] = False
    torch.testing.assert_close(checkpoint[4:][unchanged], cpu["checkpoint"][unchanged], atol=0, rtol=0)
    reset_state()

    calls = dict(native=native, v20=vector, compact_v1=fold, vector_pair=lambda: (vector(), fold()))
    calls.update(v13=lambda: vector(13), v13_pair=lambda: (vector(13), fold()))
    calls["v23"] = lambda: vector(23)
    if any(path in FUSED_PATHS for path in args.path):
        names = ("q", "k", "v", "g", "beta")
        identity_pack = torch.arange(b * t, device=dev, dtype=torch.int64) if FUSED_PACK_PATH in args.path else None

        token_offsets = (
            torch.arange(b + 1, device=dev, dtype=torch.int32) * t if FUSED_FOLD_FIRST_PATH in args.path else None
        )

        def fused(cache=c, with_pack=False, fold_first=False):
            inputs = [cache[name] for name in names]
            output = c["out"]
            if fold_first:
                inputs = [tensor.flatten(0, 1) for tensor in inputs]
                output = output.flatten(0, 1)
            if with_pack:
                packed = [tensor.new_zeros(tensor.shape) for tensor in inputs]
                for dest, src in zip(packed, inputs):
                    dest.flatten(0, 1).index_copy_(0, identity_pack, src.flatten(0, 1))
                inputs = packed
            torch.ops.replayssm_sd_lab.fused_(
                *inputs,
                *(cache[n] for n in ("checkpoint", "d_cache", "k_cache", "g_cache")),
                slots,
                c["positions"],
                c["bases"],
                c["lengths"],
                flush,
                output,
                c["status"],
                status,
                128**-0.5,
                128,
                40,
                ring,
                25 if fold_first else 24,
                *((token_offsets,) if fold_first else ()),
            )
            if with_pack:
                return c["out"].flatten(0, 1).index_select(0, identity_pack)
            return c["out"].flatten(0, 1)

        calls[FUSED_PATH] = fused
        calls[FUSED_PACK_PATH] = lambda: fused(with_pack=True)
        calls[FUSED_FOLD_FIRST_PATH] = lambda: fused(fold_first=True)
    if any(path in COMPACT_PATHS for path in args.path):
        status_v5 = torch.empty(b * value_heads * (128 // args.compact_tile), dtype=torch.int32, device=dev)

        def fold_candidate(cache=c, variant=5):
            torch.ops.replayssm_sd_compact_lab.compact_(
                *(cache[n] for n in ("checkpoint", "d_cache", "k_cache", "g_cache")),
                slots,
                c["positions"],
                c["bases"],
                flush,
                status_v5,
                args.compact_tile,
                40,
                variant,
                ring,
            )

        calls.update(
            compact_v5=fold_candidate,
            compact_v6=lambda: fold_candidate(variant=6),
            v20_compact_v5=lambda: (vector(), fold_candidate()),
            v23_compact_v5=lambda: (vector(23), fold_candidate()),
        )
    if any(path in (*CUBE_FOLD_ONLY_PATHS, MIXED_CUBE_FOLD_PATH) for path in args.path):
        ring_ids = ring.index_select(0, flush_rows.to(dev)).long()
        history_records = (torch.arange(p, device=dev, dtype=torch.int64) + base) % 32

        def cube_fold(cache=c):
            alpha, lhs, rhs = cube_fold_operands(cache, ring_ids, history_records, p)
            update = torch.bmm(lhs, rhs).view(active, value_heads, 128, 128)
            state = cache["checkpoint"].index_select(0, restore_slots)
            folded = state.float() * alpha[..., None, None] + update.float()
            cache["checkpoint"].index_copy_(0, restore_slots, folded.to(state.dtype))

        if active and p:
            calls["cube_fold_path"] = cube_fold
            if "cube_fold_gemm" in args.path:
                _, lhs, rhs = cube_fold_operands(c, ring_ids, history_records, p)
                calls["cube_fold_gemm"] = lambda: torch.bmm(lhs, rhs)
        else:
            raise ValueError("Cube folding requires at least one flushing request with committed history")

    if "cursor" in args.path:
        from vllm_ascend.ops.replayssm.ascendc import replayssm_sd_cursor

        cursor_position = torch.full((b + 1,), p, dtype=torch.int32, device=dev)
        cursor_base = torch.full_like(cursor_position, base)
        cursor_flush = torch.zeros(b + 1, dtype=torch.int8, device=dev)
        cursor_status = torch.empty_like(ring)
        cursor_reset = torch.zeros(b, dtype=torch.int8, device=dev)
        cursor_accepted = torch.full_like(ring, min(t, 3))

        def reset_cursor():
            cursor_position[1:].copy_(c["positions"])
            cursor_base.fill_(base)
            cursor_flush[1:].copy_(flush)

        def cursor():
            replayssm_sd_cursor(
                cursor_position,
                cursor_base,
                cursor_flush,
                cursor_accepted,
                ring,
                cursor_reset,
                cursor_status,
                20,
                4,
                32,
            )

        calls["cursor"] = cursor
    compare_demo = any(path in CUBE_PATHS for path in args.path)
    compare_mixed = any(path in ALL_MIXED_PATHS for path in args.path)
    if compare_demo or compare_mixed or any(path in FUSED_PATHS for path in args.path):
        # Poison destinations so a missing Cube candidate write cannot inherit V20 results.
        golden = {
            n: (wide_golden[n] if wide_golden is not None else c[n].cpu())
            for n in ("out", "d_cache", "k_cache", "g_cache")
        }
        cache_rows = torch.arange(1, b + 1, device=dev, dtype=torch.int64)
        candidate_records = [(c["positions"].long() + base + s) & 31 for s in range(t)]

        def poison():
            c["out"].fill_(float("nan"))
            c["status"].fill_(-1)
            for s, record in enumerate(candidate_records):
                for n in ("d_cache", "k_cache", "g_cache"):
                    if p == nonflush_pos:
                        c[n][1:, :, (base + p + s) % 32].fill_(float("nan"))
                    else:
                        c[n][cache_rows, :, record] = float("nan")

        def check_outputs(label, error_prefix="", dump=None):
            if c["status"].cpu().any():
                raise RuntimeError(f"{label} returned nonzero status")
            for n, ref in golden.items():
                actual = c[n].cpu().float()
                expected = ref.float()
                assert_stage(actual, expected, f"{label}/{n}", atol=0.01, rtol=0.03, equal_nan=True, dump=dump)
                errors[f"{error_prefix}{n}"] = (actual - expected).nan_to_num().abs().max().item()

    if compare_demo:
        poison()
        verify(c, t, p, base)
        check_outputs("cube_demo")
        compact(c, active, p, base, status)
        actual = c["checkpoint"].cpu().float()
        torch.testing.assert_close(actual, checkpoint.float(), atol=0.01, rtol=0.03)
        errors["checkpoint"] = (actual - checkpoint.float()).abs().max().item()
        reset_state()
        rhs = panel(c, t)
        shared_rhs = panel(c, t, grouped=True)
        calls.update(
            cube_verify=lambda: verify(c, t, p, base),
            cube_compact=lambda: compact(c, active, p, base, status),
            grouped_projection=lambda: checkpoint_projection(c["checkpoint"][5:], shared_rhs),
            projection=lambda: checkpoint_projection(c["checkpoint"][5:], rhs),
            cube_pair=lambda: (verify(c, t, p, base), compact(c, active, p, base, status)),
        )
        if active and p:
            _, dl, kh = compact_inputs(c, active, p, base)
            calls["fold_gemm"] = lambda: torch.bmm(dl, kh)

    if compare_mixed:
        blocks = min(b * (value_heads if args.mixed_head_grid else key_heads), args.mixed_cores)
        zeros = torch.zeros((16, 128), dtype=torch.bfloat16, device=dev)
        panel_tokens = 8 if wide else 4
        wide_offsets = torch.arange(b + 1, device=dev, dtype=torch.int32) * t if wide else None
        if args.mixed_workspace == "per-head":
            heads = torch.arange(value_heads).repeat(b)
            row_ids = torch.arange(b).repeat_interleave(value_heads)
            expected_projection = reference_projections(cpu, row_ids, heads, panel_tokens)
        else:
            expected_projection = last_projections(cpu, blocks, balanced=args.mixed_balanced, panel_tokens=panel_tokens)
        scratch = torch.full(expected_projection.shape, float("nan"), dtype=torch.float32, device=dev)
        dump = args.output.with_suffix(".mismatch.pt") if args.mixed_dump else None

        def check_projection(label):
            actual = scratch.cpu()
            try:
                assert_stage(
                    actual,
                    expected_projection,
                    f"{label}/projection",
                    atol=PROJECTION_ATOL,
                    rtol=PROJECTION_RTOL,
                    dump=dump,
                )
            except AssertionError as exc:
                summary = projection_error_summary(actual, expected_projection, panel_tokens)
                raise AssertionError(f"{exc}\nProjection mismatch by column group: {json.dumps(summary)}") from None

        grid_options = {"expand_head_grid": True} if args.mixed_head_grid else {}

        def verify_mixed(packed=False, cache=c, ablation=0):
            panel_options = (True,) if packed else ()
            if packed and args.mixed_suffix:
                panel_options += (True,)
            if packed and args.mixed_qkv:
                panel_options += (False, True)
            if args.mixed_balanced:
                panel_options = (packed, packed and args.mixed_suffix, packed and args.mixed_qkv, True)
            if ablation:
                panel_options = (True, False, True, args.mixed_balanced, ablation)
                if ablation >= MIXED_AIV_ABLATIONS[MIXED_FUSED_FOLD_PATH]:
                    panel_options += (flush,)
            qkv = [cache[n] for n in ("q", "k", "v", "g", "beta")]
            output = cache["out"]
            if wide:
                qkv = [tensor.flatten(0, 1) for tensor in qkv]
                output = output.flatten(0, 1)
            torch.ops.replayssm_mix_910b.verify_(
                *qkv,
                *(cache[n] for n in ("checkpoint", "d_cache", "k_cache", "g_cache")),
                slots,
                ring,
                c["positions"],
                c["bases"],
                c["lengths"],
                output,
                c["status"],
                zeros,
                scratch,
                args.mixed_cores,
                *panel_options,
                *((wide_offsets,) if wide else ()),
                **grid_options,
            )

        mixed_checks = [
            (path, path in PACKED_PANEL_PATHS, MIXED_AIV_ABLATIONS.get(path, 0))
            for path in args.path
            if path in ALL_MIXED_PATHS
        ]
        for path, packed, ablation in mixed_checks:
            for iteration in range(args.mixed_check_repeats):
                poison()
                scratch.fill_(float("nan"))
                verify_mixed(packed, ablation=ablation)
                label = f"{path} eager run {iteration + 1}"
                check_projection(label)
                check_outputs(label, "mixed_eager_", dump=dump)
                if path in IN_KERNEL_FOLD_PATHS:
                    torch.testing.assert_close(c["checkpoint"].cpu(), checkpoint, atol=0.001, rtol=0.015)
                    reset_state()
        torch.testing.assert_close(c["checkpoint"], initial, atol=0, rtol=0)
        calls.update(
            mixed_verify=verify_mixed,
            mixed_pair=lambda: (verify_mixed(), fold()),
            mixed_packed=lambda: verify_mixed(True),
        )
        if "mixed_packed_compact_v5" in args.path:
            calls["mixed_packed_compact_v5"] = lambda: (verify_mixed(True), fold_candidate())

        def mixed_with_fold(path, cache=c):
            verify_mixed(True, cache, ablation=MIXED_AIV_ABLATIONS.get(path, 0))
            if path == MIXED_CUBE_FOLD_PATH:
                cube_fold(cache)
            elif path not in IN_KERNEL_FOLD_PATHS:
                fold_candidate(cache, variant=5 if path == "mixed_packed_compact_v5" else 6)

        for path in MIXED_FOLD_PATHS:
            if path in args.path:
                calls[path] = lambda path=path: mixed_with_fold(path)

    def measure_gdn_stack(path, label):
        stack_label = f"{label}/gdn_stack_{args.layers}"
        if path == "native":
            states = [state] + [
                native_state_page(dev, args.tp, b, t, geometry=args.geometry)[0] for _ in range(args.layers - 1)
            ]

            def call_stack():
                for page in states:
                    native(page)

            def reset_stack():
                for page in states:
                    page[1::t].copy_(start_state)

            stack = measure(call_stack, reset_stack, args.iterations, stack_label)
            del states
        else:
            mutable = ("checkpoint", "d_cache", "k_cache", "g_cache")
            reset_state()
            layers = [c] + [{**c, **{name: c[name].clone() for name in mutable}} for _ in range(args.layers - 1)]

            def call_stack():
                for cache in layers:
                    if path in FUSED_PATHS:
                        fused(cache, with_pack=path == FUSED_PACK_PATH, fold_first=path == FUSED_FOLD_FIRST_PATH)
                    elif path in MIXED_FOLD_PATHS:
                        mixed_with_fold(path, cache)
                    else:
                        vector(23 if path == "v23_compact_v5" else 20, cache)
                        if path == "vector_pair":
                            fold(cache)
                        else:
                            fold_candidate(cache)

            def reset_stack():
                for cache in layers:
                    reset_state(cache)

            def check_stack(graph):
                reset_stack()
                graph.replay()
                for cache in (layers[0], layers[-1]):
                    torch.testing.assert_close(cache["checkpoint"].cpu(), checkpoint, atol=0.001, rtol=0.015)

            stack = measure(call_stack, reset_stack, args.iterations, stack_label, check_graph=check_stack)
            del layers
        return {"layers": args.layers, **stack}

    rows = []
    for path in args.path:
        if path not in calls:
            continue
        reset = (
            (lambda: first.copy_(start_state))
            if path == "native"
            else reset_cursor
            if path == "cursor"
            else reset_state
            if path
            in (
                "vector_pair",
                "v13_pair",
                "compact_v1",
                "cube_pair",
                "cube_compact",
                "cube_fold_path",
                "mixed_pair",
                *COMPACT_PATHS,
                *FUSED_PATHS,
            )
            else lambda: None
        )
        reset_state()
        label = f"{args.geometry}_tp{args.tp}_b{b}_t{t}_pos{p}_base{base}_f{active}/{path}"
        mix_metadata = {}
        if path in ALL_MIXED_PATHS:
            compact_status = None
            if path == "mixed_pair":
                compact_status = status
            elif path in MIXED_FOLD_PATHS and path not in (MIXED_CUBE_FOLD_PATH, *IN_KERNEL_FOLD_PATHS):
                compact_status = status_v5
            expected_state = checkpoint if path in (*MIXED_FOLD_PATHS, "mixed_pair") else initial.cpu()

            def check_graph(
                graph, reset=reset, path=path, expected_state=expected_state, compact_status=compact_status
            ):
                for iteration in range(args.mixed_check_repeats):
                    reset()
                    poison()
                    scratch.fill_(float("nan"))
                    if compact_status is not None:
                        compact_status.fill_(-1)
                    graph.replay()
                    label = f"{path} graph run {iteration + 1}"
                    check_projection(label)
                    check_outputs(label, "mixed_graph_", dump=dump)
                    if path in MIXED_FOLD_PATHS and path != "mixed_packed_compact_v5":
                        actual_state = c["checkpoint"].cpu()
                        torch.testing.assert_close(actual_state, expected_state, atol=0.001, rtol=0.015)
                        if path in (MIXED_CUBE_FOLD_PATH, *IN_KERNEL_FOLD_PATHS):
                            name = "cube_fold_checkpoint" if path == MIXED_CUBE_FOLD_PATH else "fused_fold_checkpoint"
                            errors[name] = (actual_state.float() - expected_state.float()).abs().max().item()
                    else:
                        torch.testing.assert_close(c["checkpoint"].cpu(), expected_state, atol=0, rtol=0)
                    if compact_status is not None and compact_status.cpu().any():
                        raise RuntimeError("Captured compact returned nonzero status")
                if path in IN_KERNEL_FOLD_PATHS and b <= 4 and active:
                    # Check the captured graph with device-side routing changed
                    # in place. The previous replay must not be folded twice.
                    saved_flush = flush.clone()
                    try:
                        reset()
                        poison()
                        flush.zero_()
                        graph.replay()
                        check_outputs("mixed fused fold: flush changed after capture", "mixed_graph_", dump=dump)
                        torch.testing.assert_close(c["checkpoint"], initial, atol=0, rtol=0)
                    finally:
                        flush.copy_(saved_flush)

            timings = measure(calls[path], reset, args.iterations, label, check_graph=check_graph)
            mix_metadata = dict(
                aic_blocks=blocks,
                aiv_blocks=2 * blocks,
                workspace_bytes=scratch.nbytes,
                workspace_layout=args.mixed_workspace,
                aiv_panel_cols=8 if path in PACKED_PANEL_PATHS and not wide else 16,
                aiv_history_suffix=(
                    path in ("mixed_history_panel", "mixed_suffix_reference", *IN_KERNEL_FOLD_PATHS)
                    or (path not in MIXED_AIV_ABLATIONS and path in PACKED_PANEL_PATHS and args.mixed_suffix)
                ),
                aiv_bulk_qkv=(path in PACKED_PANEL_PATHS and args.mixed_qkv) or path in MIXED_AIV_ABLATIONS,
                aiv_ablation=MIXED_AIV_ABLATIONS.get(path, 0),
                mixed_balanced=args.mixed_balanced,
            )
        elif path in (*COMPACT_PATHS, "v23", *FUSED_PATHS):

            def check_graph(graph, path=path):
                reset_state()
                compact_status = status if path in FUSED_PATHS else status_v5 if path in COMPACT_PATHS else None
                if compact_status is not None:
                    compact_status.fill_(-1)
                if path in FUSED_PATHS:
                    poison()
                elif path not in COMPACT_ONLY_PATHS:
                    c["status"].fill_(-1)
                    c["out"].fill_(float("nan"))
                graph.replay()
                if compact_status is not None:
                    if compact_status.cpu().any():
                        raise RuntimeError(f"Captured {path} returned nonzero compact status")
                    actual = c["checkpoint"].cpu()
                    torch.testing.assert_close(actual, checkpoint, atol=0.001, rtol=0.015)
                    errors["compact_candidate_vs_v1_checkpoint"] = (
                        (actual.float() - checkpoint.float()).abs().max().item()
                    )
                if path not in COMPACT_ONLY_PATHS:
                    if c["status"].cpu().any():
                        raise RuntimeError(f"Captured {path} returned nonzero verify status")
                    if path != FUSED_FOLD_FIRST_PATH:
                        torch.testing.assert_close(c["out"][:, :t].float().cpu(), expected_out, atol=0, rtol=0)
                if path in FUSED_PATHS:
                    check_outputs(f"Captured {path}", f"{path}_")

            timings = measure(calls[path], reset, args.iterations, label, check_graph=check_graph)
            if path in COMPACT_PATHS:
                mix_metadata = {"compact_tile": args.compact_tile}
        elif path == "cube_fold_path":

            def check_cube_fold(graph):
                reset_state()
                graph.replay()
                actual = c["checkpoint"].cpu()
                torch.testing.assert_close(actual, checkpoint, atol=0.001, rtol=0.015)
                errors["cube_fold_checkpoint"] = (actual.float() - checkpoint.float()).abs().max().item()

            timings = measure(calls[path], reset, args.iterations, label, check_graph=check_cube_fold)
        else:
            timings = measure(calls[path], reset, args.iterations, label)
        if args.compare_eager:
            timings["eager"] = measure_eager(calls[path], reset, args.iterations)
        if args.layers > 1:
            timings["gdn_stack"] = measure_gdn_stack(path, label)
        rows.append(
            dict(
                batch=b,
                pos=p,
                length=t,
                base=base,
                active_flush=active,
                flush_layout=args.flush_layout,
                flush_seed=args.flush_seed,
                nonflush_pos=nonflush_pos,
                key_heads=key_heads,
                value_heads=value_heads,
                native_state_page_bytes=state_page_bytes,
                path=path,
                errors=dict(errors),
                **timings,
                **mix_metadata,
            )
        )
    return rows


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", choices=("all", "vector", "mixed"), default="all")
    ap.add_argument("--geometry", choices=("27b", "9b"), default="27b")
    ap.add_argument("--batches", type=batches)
    ap.add_argument("--length", type=int, choices=(1, 2, 3, 4, 6, 8), default=4)
    ap.add_argument("--pos", type=int, choices=range(17), default=15)
    ap.add_argument("--base", type=int, choices=range(32), default=31)
    ap.add_argument("--flush", choices=("none", "mixed", "full"), default="full")
    ap.add_argument("--flush-count", type=int, help="Hold the number of flushing rows fixed across batches")
    ap.add_argument("--flush-layout", choices=("prefix", "spread", "random"), default="prefix")
    ap.add_argument("--flush-seed", type=int, default=42)
    ap.add_argument("--nonflush-pos", type=int, choices=range(17), help="Use a shorter history for non-flushing rows")
    ap.add_argument(
        "--path",
        nargs="+",
        choices=tuple(
            dict.fromkeys(
                (
                    *PATHS,
                    *ALL_MIXED_PATHS,
                    *EXTRA_PATHS,
                    *COMPACT_PATHS,
                    *WIDE_PATHS,
                    *FUSED_PATHS,
                    *CUBE_FOLD_ONLY_PATHS,
                )
            )
        ),
    )
    ap.add_argument("--compact-library", type=Path, help="Path to the standalone Compact Torch binding")
    ap.add_argument("--verify-library", type=Path, help="Path to the standalone V23 Torch binding")
    ap.add_argument("--compact-tile", type=int, choices=(32, 64), default=64)
    ap.add_argument("--mixed-library", type=Path, help="Path to libreplayssm_sd_mix_torch.so")
    ap.add_argument("--mixed-cores", type=int, choices=range(1, 21), default=20)
    ap.add_argument("--mixed-workspace", choices=("per-core", "per-head"), default="per-core")
    ap.add_argument("--mixed-balanced", action="store_true", help="Balance entire value-head ranges across Cube blocks")
    ap.add_argument("--mixed-head-grid", action="store_true", help="Let balanced tasks launch one block per value head")
    ablations = ap.add_mutually_exclusive_group()
    ablations.add_argument(
        "--mixed-suffix", action="store_true", help="Pre-scale packed AIV history rows by gate suffix"
    )
    ablations.add_argument("--mixed-qkv", action="store_true", help="Batch candidate Q/K/V DMA on AIV")
    ap.add_argument("--mixed-check-repeats", type=int, default=1, help="Untimed eager and graph checks per case")
    ap.add_argument("--mixed-dump", action="store_true", help="Save the first mismatch beside --output")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--compare-eager", action="store_true", help="Also time eager launches on the same input")
    ap.add_argument("--layers", type=int, default=1, help="Also replay a GDN stack with distinct cache pages per layer")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.iterations < 1 or args.mixed_check_repeats < 1:
        ap.error("iterations and mixed check repeats must be positive")
    vector_only = args.suite == "vector" or (args.geometry == "9b" and args.suite == "all")
    if args.suite == "mixed":
        defaults = (*VECTOR_PATHS, *MIXED_PATHS)
    else:
        defaults = VECTOR_PATHS if vector_only else PATHS
    args.batches = args.batches or (DEFAULT_BATCHES if defaults == PATHS else batches("all"))
    args.path = args.path or defaults
    if args.length > 4 and set(args.path) - {"native", MIXED_FUSED_FOLD_PATH, MIXED_COLUMN_RECURRENCE_PATH}:
        ap.error("six/eight-token windows require --path native, mixed_fused_fold or mixed_column_recurrence")
    if args.flush_count is not None and not 0 <= args.flush_count <= min(args.batches):
        ap.error("flush-count must fit the smallest requested batch")
    if (args.suite != "all" or args.geometry == "9b") and any(path in CUBE_PATHS for path in args.path):
        ap.error("the PyTorch Cube demo uses the 27B TP4 all suite")
    if any(path in ALL_MIXED_PATHS for path in args.path) and not args.mixed_library:
        ap.error("mixed paths require --mixed-library")
    if (args.mixed_suffix or args.mixed_qkv) and not any(path in PACKED_PANEL_PATHS for path in args.path):
        ap.error("mixed-suffix and mixed-qkv require a packed AIV path")
    if args.mixed_balanced and not any(path in ALL_MIXED_PATHS for path in args.path):
        ap.error("mixed-balanced requires a mixed kernel path")
    if args.mixed_head_grid and not args.mixed_balanced:
        ap.error("mixed-head-grid requires --mixed-balanced")
    if (
        any(path in COMPACT_PATHS and path not in IN_KERNEL_FOLD_PATHS for path in args.path)
        and not args.compact_library
    ):
        ap.error("Compact candidate paths require --compact-library")
    if any(path in (*WIDE_PATHS, *FUSED_PATHS) for path in args.path) and not args.verify_library:
        ap.error("V23 and V24 paths require --verify-library")
    if any(path in args.path for path in (FUSED_FOLD_FIRST_PATH, FUSED_PACK_PATH)) and args.length != 4:
        ap.error("the identity pack or V25 view path requires four tokens per request")
    if args.suite == "vector" and any(path in ALL_MIXED_PATHS for path in args.path):
        ap.error("use --suite mixed for AIC/AIV paths")
    if max(args.batches) > 128 and any(path in CUBE_PATHS for path in args.path):
        ap.error("batches above 128 are supported by the Ascend C paths")
    if any(path in CUBE_PATHS for path in args.path) and (
        args.flush_layout != "prefix" or args.nonflush_pos is not None
    ):
        ap.error("mixed flush placement or per-row history requires Ascend C paths")
    if not 1 <= args.layers <= 48 or (args.layers > 1 and any(path not in STACK_PATHS for path in args.path)):
        ap.error("GDN stack supports 1..48 layers; choose native and/or paired Ascend C paths")
    if len(set(args.path)) != len(args.path):
        ap.error("paths must be unique")
    args.tp = 4 if args.geometry == "27b" else 1
    return args


def main():
    args = parse_args()
    import torch_npu  # noqa: F401

    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(args.device)
    if not enable_custom_op():
        raise RuntimeError("framework v20 + compact v1 extension required")
    library = Path(importlib.import_module("vllm_ascend.vllm_ascend_C").__file__).resolve()
    report = dict(
        status="RUNNING",
        commit=(
            (ROOT / "SOURCE_COMMIT").read_text().strip()
            if (ROOT / "SOURCE_COMMIT").exists()
            else subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        ),
        packages=package_versions(),
        config=vars(args),
        soc=torch.npu.get_device_name(args.device),
        library=str(library),
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        rows=[],
    )
    if any(path in ALL_MIXED_PATHS for path in args.path):
        mixed_library = args.mixed_library.resolve(strict=True)
        torch.ops.load_library(str(mixed_library))
        mixed_ops = torch.ops.replayssm_mix_910b
        max_tokens = mixed_ops.max_tokens() if hasattr(mixed_ops, "max_tokens") else 4
        if max_tokens != (8 if args.length > 4 else 4):
            raise ValueError("Build the mixed library with REPLAYSSM_VERIFY_MAX_TOKENS=8 for six/eight-token windows")
        report.update(
            mixed_library=str(mixed_library),
            mixed_library_sha256=hashlib.sha256(mixed_library.read_bytes()).hexdigest(),
        )
    for name in ("compact_library", "verify_library"):
        path = getattr(args, name)
        if path is None:
            continue
        extension = path.resolve(strict=True)
        torch.ops.load_library(str(extension))
        report[name] = str(extension)
        report[f"{name}_sha256"] = hashlib.sha256(extension.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for b in args.batches:
        try:
            report["rows"].extend(trial(args, b, torch.device(f"npu:{args.device}")))
            if b == args.batches[-1]:
                report["status"] = "PASSED"
        except Exception as exc:
            report.update(status="FAILED", failed_batch=b, error=repr(exc))
            raise
        finally:
            args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()

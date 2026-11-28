# SPDX-License-Identifier: Apache-2.0
"""Compare fixed-slot, compact vector and mixed verify on the same NPU graph inputs."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experimental.replayssm_cube_910b.run import (  # noqa: E402
    MIXED_AIV_ABLATIONS,
    MIXED_FOLD_FMA_PATH,
    MIXED_FOLD_PAIR_PATH,
    MIXED_FUSED_FOLD_PATH,
    batches,
    measure,
    select_flush_rows,
)
from experimental.replayssm_sd_ascendc.lab import CACHE_NAMES, MAX_TOKENS, make_case  # noqa: E402
from scripts.replayssm.profile_sd_operator_compare import package_versions  # noqa: E402

INPUTS = ("q", "k", "v", "g", "beta")
PATHS = ("fixed", "pack_unpack", "compact")
FOLD_FIRST_PATH = "fold_first"
MIXED_PATH = "mixed_fold"
PACKED_PATHS = ("compact", FOLD_FIRST_PATH, MIXED_PATH)
RING_LEN = 32
TILE_V = 128
CORES = 40
FLUSH_POSITION = 15
NONFLUSH_POSITION = 7


def layout_tensors(lengths, device):
    offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
    pick = (torch.arange(MAX_TOKENS) < lengths[:, None]).flatten().nonzero().flatten()
    return offsets.to(device), pick.to(device)


def trial(args, batch, device):
    tp = 4 if args.geometry == "27b" else 1
    cpu = make_case(
        geometry=args.geometry,
        tp=tp,
        batch=batch,
        position=FLUSH_POSITION,
        length=MAX_TOKENS,
        base=31,
        tile_v=TILE_V,
        seed=42,
    )
    lengths = cpu["lengths"]
    if args.layout != "uniform":
        lengths.copy_((torch.arange(batch) % MAX_TOKENS + 1).to(torch.int32))
    if args.layout == "padded":
        # Include an interior pad and a trailing pad when the batch allows it.
        lengths[1::5].zero_()
        if batch > 1:
            lengths[-1] = 0
    padding = lengths == 0
    flush_count = min(int(batch * args.flush_fraction), int((~padding).sum()))
    flush_rows = (~padding).nonzero().flatten()[select_flush_rows(int((~padding).sum()), flush_count, "random", 42)]
    cpu["positions"].fill_(NONFLUSH_POSITION)
    cpu["positions"][flush_rows] = FLUSH_POSITION
    cpu["positions"][padding] = 0
    cpu["bases"][padding] = 0
    cpu["slots"][padding] = 0

    case = {name: cpu[name].to(device) for name in ("slots", "positions", "bases", "lengths", *CACHE_NAMES)}
    # Exercise independent checkpoint and ring IDs.
    case["checkpoint"] = torch.cat((case["checkpoint"][:1], case["checkpoint"]))
    slots = torch.where(case["slots"] > 0, case["slots"] + 1, 0)
    value_heads = cpu["v"].size(2)
    fixed_inputs = [cpu[name].to(device) for name in INPUTS]
    offsets, pick = layout_tensors(lengths, device)
    tokens = pick.numel()
    compact_inputs = [tensor.flatten(0, 1).index_select(0, pick) for tensor in fixed_inputs]
    if args.layout == "padded":
        # A captured graph may reserve physical token capacity beyond offsets[-1].
        compact_inputs = [
            torch.cat((src, src.new_full((batch * MAX_TOKENS - tokens, *src.shape[1:]), float("nan"))))
            for src in compact_inputs
        ]
    flush = torch.zeros(batch, dtype=torch.int32, device=device)
    flush[flush_rows.to(device)] = 1
    restore_slots = slots[flush_rows.to(device)].long()
    restore_state = case["checkpoint"].index_select(0, restore_slots)
    verify_status = torch.empty(batch * value_heads, dtype=torch.int32, device=device)
    compact_status = torch.empty(batch * value_heads * 4, dtype=torch.int32, device=device)
    fixed_out = torch.empty_like(fixed_inputs[2])
    compact_out = torch.empty_like(compact_inputs[2])
    op = torch.ops.replayssm_sd_lab.fused_

    def invoke(cache, inputs, out, token_offsets=None, *, variant=24):
        op(
            *inputs,
            *(cache[name] for name in CACHE_NAMES),
            slots,
            case["positions"],
            case["bases"],
            case["lengths"],
            flush,
            out,
            verify_status,
            compact_status,
            TILE_V**-0.5,
            TILE_V,
            CORES,
            case["slots"],
            variant,
            token_offsets,
        )

    if MIXED_PATH in args.path:
        rhs_zeros = torch.zeros((16, 128), dtype=fixed_inputs[0].dtype, device=device)
        blocks = min(20, batch * fixed_inputs[0].size(2))
        scratch = torch.empty((blocks, 128, 16), dtype=torch.float32, device=device)

    fold_path = MIXED_FUSED_FOLD_PATH
    if args.mixed_fold_pair:
        fold_path = MIXED_FOLD_PAIR_PATH
    elif args.mixed_fold_fma:
        fold_path = MIXED_FOLD_FMA_PATH

    def invoke_mixed(cache, inputs, out, token_offsets=None, *, ablation=fold_path):
        torch.ops.replayssm_mix_910b.verify_(
            *inputs,
            *(cache[name] for name in CACHE_NAMES),
            slots,
            case["slots"],
            case["positions"],
            case["bases"],
            case["lengths"],
            out,
            verify_status,
            rhs_zeros,
            scratch,
            20,
            True,
            False,
            True,
            args.mixed_balanced,
            MIXED_AIV_ABLATIONS[ablation],
            flush,
            token_offsets,
        )

    def reset_cache(cache):
        if flush_count:
            cache["checkpoint"].index_copy_(0, restore_slots, restore_state)

    def reset_status():
        verify_status.fill_(-1)
        compact_status.fill_(-1)

    def check_status(path):
        torch.testing.assert_close(verify_status.cpu(), torch.zeros_like(verify_status, device="cpu"))
        if path != MIXED_PATH:
            torch.testing.assert_close(compact_status.cpu(), torch.zeros_like(compact_status, device="cpu"))

    def capture_expected(valid_indices=pick, path="fixed"):
        reset_cache(case)
        reset_status()
        if path == MIXED_PATH:
            reference_fold = MIXED_FUSED_FOLD_PATH if args.mixed_fold_pair else fold_path
            invoke_mixed(case, fixed_inputs, fixed_out, ablation=reference_fold)
        else:
            invoke(case, fixed_inputs, fixed_out)
        check_status(path)
        expected_out = fixed_out.flatten(0, 1).index_select(0, valid_indices).cpu()
        expected_cache = {name: case[name].cpu() for name in CACHE_NAMES}
        reset_cache(case)
        return expected_out, expected_cache

    input_ring = {name: cpu[name] for name in CACHE_NAMES[1:]}

    def make_layers(path):
        layers = []
        for _ in range(args.layers):
            cache = {name: case[name].clone() for name in CACHE_NAMES[1:]}
            initial = case["checkpoint"]
            if path == MIXED_PATH:
                # Add a gap between slots to model shared KV pages.
                strides = (initial.stride(0) + 128, *initial.stride()[1:])
                cache["checkpoint"] = torch.empty_strided(
                    initial.shape, strides, dtype=initial.dtype, device=device
                ).copy_(initial)
            else:
                cache["checkpoint"] = initial.clone()
            layers.append(cache)
        return layers

    def check_arithmetic(actual, reference, changed, name, errors):
        torch.testing.assert_close(actual[~changed], reference[~changed], atol=0, rtol=0, equal_nan=True, msg=name)
        actual, reference = actual[changed], reference[changed]
        torch.testing.assert_close(actual, reference, atol=2e-5, rtol=0.008, msg=name)
        if actual.numel():
            errors[name] = max(errors.get(name, 0), (actual.float() - reference.float()).abs().max().item())

    def check_cache(cache, reference, path, query_lengths, errors):
        for name in CACHE_NAMES:
            if path == FOLD_FIRST_PATH and name == "d_cache" and flush_count:
                # Only the newly generated D in a flushed row can change.
                changed = torch.zeros(cache[name].shape[0], RING_LEN, dtype=torch.bool)
                for row in flush_rows.tolist():
                    start = int(cpu["bases"][row]) + int(cpu["positions"][row])
                    positions = (torch.arange(int(query_lengths[row])) + start) % RING_LEN
                    changed[cpu["slots"][row], positions] = True
                check_arithmetic(
                    cache[name].cpu().transpose(1, 2), reference[name].transpose(1, 2), changed, name, errors
                )
            else:
                torch.testing.assert_close(cache[name].cpu(), reference[name], atol=0, rtol=0, equal_nan=True, msg=name)

    def check_compact(out, reference, path, query_lengths, errors):
        check_status(path)
        actual = out[:tokens].cpu()
        if path == FOLD_FIRST_PATH and flush_count:
            rows = torch.repeat_interleave(torch.arange(batch), query_lengths)
            check_arithmetic(actual, reference, torch.isin(rows, flush_rows), "out", errors)
        else:
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        if out.size(0) > tokens:
            assert torch.isnan(out[tokens:]).all().item(), "compact output wrote into padding"

    def test_path(path):
        errors = {}
        outputs = []
        for name, initial in input_ring.items():
            case[name].copy_(initial)
        layers = make_layers(path)

        def reset_stack():
            for cache in layers:
                reset_cache(cache)

        expected_out, expected_cache = capture_expected(path=path)
        if path == MIXED_PATH:

            def call_stack():
                for cache in layers:
                    invoke_mixed(cache, compact_inputs, compact_out, offsets)

        elif path in PACKED_PATHS:

            def call_stack(variant=25 if path == FOLD_FIRST_PATH else 24):
                for cache in layers:
                    invoke(cache, compact_inputs, compact_out, offsets, variant=variant)

        elif path == "fixed":

            def call_stack():
                for cache in layers:
                    invoke(cache, fixed_inputs, fixed_out)

        else:

            def call_stack():
                for index, cache in enumerate(layers):
                    packed = [tensor.new_zeros(tensor.shape) for tensor in fixed_inputs]
                    for dest, src in zip(packed, compact_inputs):
                        dest.flatten(0, 1).index_copy_(0, pick, src[:tokens])
                    invoke(cache, packed, fixed_out)
                    unpacked = fixed_out.flatten(0, 1).index_select(0, pick)
                    if len(outputs) <= index:
                        outputs.append(unpacked)
                    else:
                        outputs[index] = unpacked

        def check_stack(graph, path=path, errors=errors):
            reset_stack()
            reset_status()
            compact_out.fill_(float("nan"))
            fixed_out.fill_(float("nan"))
            graph.replay()
            if path in PACKED_PATHS:
                check_compact(compact_out, expected_out, path, lengths, errors)
            else:
                check_status(path)
                result = outputs[-1] if path == "pack_unpack" else fixed_out.flatten(0, 1).index_select(0, pick)
                torch.testing.assert_close(result.cpu(), expected_out, atol=0, rtol=0)
            check_cache(layers[0], expected_cache, path, lengths, errors)
            if len(layers) > 1:
                check_cache(layers[-1], expected_cache, path, lengths, errors)

            if path in PACKED_PATHS and args.layout == "ragged" and batch > 1:
                # Keep the same number of tokens and buffer addresses, but change
                # their request boundaries after capture. Rebuild the fixed oracle
                # using those same flat Q/K/V/g/beta values.
                changed_lengths = lengths.roll(1)
                changed_offsets, changed_pick = layout_tensors(changed_lengths, device)
                offsets.copy_(changed_offsets)
                case["lengths"].copy_(changed_lengths)
                for dest, src in zip(fixed_inputs, compact_inputs):
                    dest.flatten(0, 1).index_copy_(0, changed_pick, src)
                # The oracle and replay must see the same ring bytes even
                # outside the newly assigned candidate window.
                for name, initial in input_ring.items():
                    case[name].copy_(initial)
                    for cache in layers:
                        cache[name].copy_(initial)
                changed_expected, changed_cache = capture_expected(changed_pick, path)
                reset_stack()
                reset_status()
                compact_out.fill_(float("nan"))
                graph.replay()
                check_compact(compact_out, changed_expected, path, changed_lengths, errors)
                check_cache(layers[0], changed_cache, path, changed_lengths, errors)
                if len(layers) > 1:
                    check_cache(layers[-1], changed_cache, path, changed_lengths, errors)

                if batch == 4:
                    # Make only the last row invalid. It must write a status
                    # without touching its output or checkpoint/ring slots.
                    bad_ring_slot = int(cpu["slots"][-1])
                    ring_snapshot = {name: layers[-1][name][bad_ring_slot].cpu() for name in CACHE_NAMES[1:]}
                    offsets[-1] = tokens + 1
                    reset_stack()
                    reset_status()
                    compact_out.fill_(float("nan"))
                    graph.replay()
                    statuses = verify_status.cpu().view(batch, value_heads)
                    assert (statuses[:-1] == 0).all() and (statuses[-1] == 6).all()
                    if path != MIXED_PATH:
                        statuses = compact_status.cpu().view(batch, value_heads * 4)
                        assert (statuses[:-1] == 0).all() and (statuses[-1] == 6).all()
                    last_start = int(changed_lengths[:-1].sum())
                    assert torch.isnan(compact_out[last_start:]).all().item()
                    bad_checkpoint_slot = int(cpu["slots"][-1]) + 1
                    last_initial = cpu["checkpoint"][bad_checkpoint_slot - 1]
                    torch.testing.assert_close(layers[-1]["checkpoint"][bad_checkpoint_slot].cpu(), last_initial)
                    for name in CACHE_NAMES[1:]:
                        torch.testing.assert_close(
                            layers[-1][name][bad_ring_slot].cpu(),
                            ring_snapshot[name],
                            atol=0,
                            rtol=0,
                            equal_nan=True,
                        )

        label = f"{args.geometry}/b{batch}/{args.layout}/f{flush_count}/{path}/layers{args.layers}"
        timing = measure(call_stack, reset_stack, args.iterations, label, check_graph=check_stack)
        if path in PACKED_PATHS and args.layout == "ragged" and batch > 1:
            # Each variant's timing must start from the same request boundaries.
            initial_offsets, _ = layout_tensors(lengths, device)
            offsets.copy_(initial_offsets)
            case["lengths"].copy_(lengths)
            for name, original in zip(INPUTS, fixed_inputs):
                original.copy_(cpu[name])
        return dict(path=path, batch=batch, tokens=tokens, flush_rows=flush_count, errors=errors, **timing)

    return [test_path(path) for path in args.path]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=("27b", "9b"), required=True)
    parser.add_argument("--batches", type=batches, required=True)
    parser.add_argument("--layout", choices=("uniform", "ragged", "padded"), default="ragged")
    parser.add_argument("--flush-fraction", type=float, default=0.125)
    parser.add_argument("--path", nargs="+", choices=(*PATHS, FOLD_FIRST_PATH, MIXED_PATH), default=PATHS)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--verify-library", type=Path, required=True)
    parser.add_argument("--mixed-library", type=Path)
    parser.add_argument("--mixed-balanced", action="store_true", help="Split key-head groups across Cube blocks")
    fold_options = parser.add_mutually_exclusive_group()
    fold_options.add_argument("--mixed-fold-fma", action="store_true", help="Fuse multiply-add in checkpoint folding")
    fold_options.add_argument("--mixed-fold-pair", action="store_true", help="Batch pairs of history row broadcasts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.flush_fraction <= 1 or not 1 <= args.layers <= 48 or args.iterations < 1:
        parser.error("Expected flush-fraction in [0,1], 1..48 layers, and at least one iteration")

    if MIXED_PATH in args.path and args.mixed_library is None:
        parser.error("--path mixed_fold requires --mixed-library")

    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    library = args.verify_library.resolve(strict=True)
    torch.ops.load_library(str(library))
    mixed_library = None
    if args.mixed_library is not None:
        mixed_library = args.mixed_library.resolve(strict=True)
        torch.ops.load_library(str(mixed_library))
    report = dict(
        status="RUNNING",
        config=vars(args),
        packages=package_versions(),
        device_name=torch.npu.get_device_name(args.device),
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        rows=[],
    )
    if mixed_library is not None:
        report["mixed_library_sha256"] = hashlib.sha256(mixed_library.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for batch in args.batches:
            report["rows"].extend(trial(args, batch, device))
        report["status"] = "PASSED"
    except Exception as exc:
        report.update(status="FAILED", failed_batch=batch, error=repr(exc))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()

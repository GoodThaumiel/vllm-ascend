# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Standalone CPU golden and NPU binding loader. Does not import vLLM."""

from pathlib import Path

import torch

HEAD_DIM = 128
BUFFER_NAMES = ("checkpoint", "d_cache", "k_cache", "g_cache")
INPUT_NAMES = ("q", "k", "v", "g", "beta", *BUFFER_NAMES, "slots", "positions", "flush", "out", "status")


def ub_bytes(capacity: int, tile_v: int) -> int:
    """Allocated UB bytes, including 32-byte rounding of each buffer."""
    sizes = (
        2 * HEAD_DIM * max(capacity, tile_v),
        2 * tile_v * HEAD_DIM,
        4 * tile_v * HEAD_DIM,
        4 * tile_v * HEAD_DIM,
        4 * capacity * HEAD_DIM,
        4 * capacity * tile_v,
        4 * 64,
        4 * (2 * HEAD_DIM + 3 * tile_v + 8),
        4096,
    )
    return sum((size + 31) // 32 * 32 for size in sizes)


def load_library(path: Path):
    # Register the NPU backend before loading the C++ TORCH_LIBRARY extension.
    import torch_npu  # noqa: F401

    torch.ops.load_library(str(path.resolve(strict=True)))
    return torch.ops.replayssm_lab.ar_step_


def invoke(op, case: dict, tile_v: int = 32, cores: int = 1):
    op(*(case[name] for name in INPUT_NAMES), HEAD_DIM**-0.5, tile_v, cores)


def validate_metadata(case: dict):
    """CPU diagnostic contract. Never called from the serving hot path."""
    slots, positions, flushes = (case[name].tolist() for name in ("slots", "positions", "flush"))
    capacity = case["d_cache"].shape[2]
    active = []
    for slot, pos, flush in zip(slots, positions, flushes):
        if not 0 <= slot < case["checkpoint"].shape[0]:
            raise ValueError("slot outside allocation")
        if not 0 <= pos < capacity:
            raise ValueError("position outside allocation")
        if flush not in (0, 1) or (pos == capacity - 1 and slot != 0 and not flush):
            raise ValueError("last position must flush; flush must be 0 or 1")
        if slot == 0 and (pos or flush):
            raise ValueError("padding must have zero position/flush")
        if slot:
            active.append(slot)
    if len(active) != len(set(active)):
        raise ValueError("active slots must be unique in a launch")


def make_case(tp=4, batch=2, capacity=16, position=0, flush=False, tile_v=32, seed=0, padding=False):
    if tp not in (4, 8) or tile_v not in (16, 32, 64) or batch < 1 or not 1 <= capacity <= 64:
        raise ValueError("invalid experiment geometry")
    if not 0 <= position < capacity:
        raise ValueError("invalid position")
    generator = torch.Generator().manual_seed(seed)
    heads, value_heads, num_slots = 16 // tp, 48 // tp, batch + 2

    def rand(*shape):
        return torch.randn(*shape, generator=generator)

    def norm(x):
        # Match the native path: normalize in FP32, then store BF16 q/k.
        x = x.to(torch.bfloat16).float()
        return (x / (x.square().sum(-1, keepdim=True) + 1e-6).sqrt()).to(torch.bfloat16)

    case = {
        "q": norm(rand(batch, heads, HEAD_DIM)),
        "k": norm(rand(batch, heads, HEAD_DIM)),
        "v": (rand(batch, value_heads, HEAD_DIM) * 0.3).to(torch.bfloat16),
        "g": -torch.nn.functional.softplus(rand(batch, value_heads)) * 0.1,
        "beta": torch.sigmoid(rand(batch, value_heads)).to(torch.bfloat16).float(),
        "checkpoint": (rand(num_slots, value_heads, HEAD_DIM, HEAD_DIM) * 0.1).to(torch.bfloat16),
        "d_cache": (rand(num_slots, value_heads, capacity, HEAD_DIM) * 0.1).to(torch.float16),
        "k_cache": (rand(num_slots, heads, capacity, HEAD_DIM) * 0.1).to(torch.float16),
        "g_cache": -torch.nn.functional.softplus(rand(num_slots, value_heads, capacity)) * 0.1,
        "slots": torch.arange(1, batch + 1, dtype=torch.int32),
        "positions": torch.full((batch,), position, dtype=torch.int32),
        "flush": torch.full((batch,), int(flush), dtype=torch.int32),
        "out": torch.full((batch, value_heads, HEAD_DIM), float("nan"), dtype=torch.bfloat16),
        "status": torch.full((batch * value_heads * (HEAD_DIM // tile_v),), -99, dtype=torch.int32),
    }
    # Poison all uncommitted cache tails. A masked/short-prefix bug becomes a
    # deterministic NaN instead of being concealed by a zero initialization.
    for name in ("d_cache", "k_cache", "g_cache"):
        case[name][:, :, position:] = float("nan")
    if padding:
        case["slots"][-1] = 0
        case["positions"][-1] = 0
        case["flush"][-1] = 0
    validate_metadata(case)
    return case


def clone_case(case, device="cpu"):
    return {name: tensor.to(device).clone() for name, tensor in case.items()}


def golden(case: dict):
    """FP64 CPU replay with quantization only at persistent-buffer boundaries.

    The history formula is evaluated as a closed form, independently of the
    device's sequential outer-update loop. No FP16 rounding of decay-scaled d
    occurs here: stored FP16 d/k are promoted before all arithmetic.
    """
    validate_metadata(case)
    result = clone_case(case)
    result["status"].zero_()
    heads, value_heads = case["q"].shape[1], case["v"].shape[1]
    for row, slot in enumerate(case["slots"].tolist()):
        if slot == 0:
            result["out"][row].zero_()
            continue
        pos = int(case["positions"][row])
        flush = bool(case["flush"][row])
        for head in range(value_heads):
            kh = head // (value_heads // heads)
            q, k, v = (case[name][row, index].double() for name, index in (("q", kh), ("k", kh), ("v", head)))
            state = case["checkpoint"][slot, head].double()
            if pos:
                logs = case["g_cache"][slot, head, :pos].double()
                decay = torch.exp(logs.sum() - logs.cumsum(0))
                d = case["d_cache"][slot, head, :pos].double()
                keys = case["k_cache"][slot, kh, :pos].double()
                state = logs.sum().exp() * state + (d * decay[:, None]).T @ keys
            decayed = case["g"][row, head].double().exp() * state
            delta = case["beta"][row, head].double() * (v - decayed @ k)
            updated = decayed + delta[:, None] * k[None, :]
            result["out"][row, head] = (updated @ (q * HEAD_DIM**-0.5)).to(torch.bfloat16)
            if flush:
                result["checkpoint"][slot, head] = updated.to(torch.bfloat16)
            else:
                result["d_cache"][slot, head, pos] = delta.to(torch.float16)
                result["k_cache"][slot, kh, pos] = k.to(torch.float16)
                result["g_cache"][slot, head, pos] = case["g"][row, head]
    return result


def error_metrics(actual: torch.Tensor, expected: torch.Tensor):
    actual, expected = actual.float(), expected.float()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("nonfinite values in computed output")
    error = (actual - expected).abs()
    reference_norm = float(torch.linalg.vector_norm(expected))
    relative_l2 = float(torch.linalg.vector_norm(error)) / max(reference_norm, 1e-12)
    return {"max_abs": float(error.max()), "mean_abs": float(error.mean()), "relative_l2": relative_l2}


def check_result(actual, expected, before, preserved=None):
    """Check output, every changed buffer and exact preservation elsewhere."""
    metrics = {}
    torch.testing.assert_close(actual["status"], expected["status"], rtol=0, atol=0)
    metrics["out"] = error_metrics(actual["out"], expected["out"])
    torch.testing.assert_close(actual["out"], expected["out"], rtol=2**-6, atol=2**-10)
    if metrics["out"]["relative_l2"] > 2e-2:
        raise AssertionError(f"output relative L2 error exceeds 2%: {metrics['out']}")
    preserved = before if preserved is None else preserved
    for name in BUFFER_NAMES:
        changed = torch.zeros_like(before[name], dtype=torch.bool)
        for row, slot in enumerate(before["slots"].tolist()):
            if slot == 0:
                continue
            if bool(before["flush"][row]):
                if name == "checkpoint":
                    changed[slot] = True
            elif name != "checkpoint":
                changed[slot, :, int(before["positions"][row])] = True
        unchanged = ~changed
        torch.testing.assert_close(actual[name][unchanged], preserved[name][unchanged], rtol=0, atol=0, equal_nan=True)
        if changed.any():
            metrics[name] = error_metrics(actual[name][changed], expected[name][changed])
            tolerance = 2**-6 if name == "checkpoint" else 2**-9
            if name == "g_cache":
                tolerance = 0
            torch.testing.assert_close(actual[name][changed], expected[name][changed], rtol=tolerance, atol=tolerance)
    return metrics

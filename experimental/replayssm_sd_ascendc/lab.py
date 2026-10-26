# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU contracts for the isolated, non-flush GDN speculative verify pilot."""

from __future__ import annotations

import math

import torch

MAX_TOKENS = 4
MAX_HISTORY = 16
MAX_WINDOW = MAX_HISTORY + MAX_TOKENS
HEAD_DIM = 128
CACHE_NAMES = ("checkpoint", "d_cache", "k_cache", "g_cache")


def make_case(
    *,
    tp: int = 4,
    batch: int = 2,
    position: int = 7,
    length: int = 4,
    base: int = 0,
    ring_len: int = 32,
    tile_v: int = 32,
    seed: int = 0,
    padding: bool = False,
) -> dict[str, torch.Tensor]:
    if tp not in (4, 8) or batch < 1 or ring_len not in (16, 32, 64):
        raise ValueError("Invalid TP, batch, or ring length")
    if not 0 <= position <= MAX_HISTORY or not 1 <= length <= MAX_TOKENS:
        raise ValueError("Invalid history position or verify length")
    if position + length > min(MAX_WINDOW, ring_len) or not 0 <= base < ring_len:
        raise ValueError("This pilot requires the verify window to fit the physical ring")
    if tile_v not in (16, 32, 64, 128):
        raise ValueError("Invalid tile_v")
    generator = torch.Generator().manual_seed(seed)
    heads, value_heads = 16 // tp, 48 // tp
    slots = batch + 1

    def rand(*shape, dtype=torch.float32, multiplier=1.0):
        return (torch.randn(*shape, generator=generator) * multiplier).to(dtype)

    case = {
        "q": rand(batch, MAX_TOKENS, heads, HEAD_DIM, dtype=torch.bfloat16, multiplier=0.09),
        "k": rand(batch, MAX_TOKENS, heads, HEAD_DIM, dtype=torch.bfloat16, multiplier=0.09),
        "v": rand(batch, MAX_TOKENS, value_heads, HEAD_DIM, dtype=torch.bfloat16, multiplier=0.09),
        "g": -(torch.rand(batch, MAX_TOKENS, value_heads, generator=generator) * 0.04),
        "beta": torch.rand(batch, MAX_TOKENS, value_heads, generator=generator) * 0.8,
        "checkpoint": rand(slots, value_heads, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, multiplier=0.02),
        "d_cache": torch.full((slots, value_heads, ring_len, HEAD_DIM), float("nan"), dtype=torch.float16),
        "k_cache": torch.full((slots, heads, ring_len, HEAD_DIM), float("nan"), dtype=torch.float16),
        "g_cache": torch.full((slots, value_heads, ring_len), float("nan"), dtype=torch.float32),
        "slots": torch.arange(1, slots, dtype=torch.int32),
        "positions": torch.full((batch,), position, dtype=torch.int32),
        "bases": torch.full((batch,), base, dtype=torch.int32),
        "lengths": torch.full((batch,), length, dtype=torch.int32),
        "out": torch.full((batch, MAX_TOKENS, value_heads, HEAD_DIM), float("nan"), dtype=torch.bfloat16),
        "status": torch.full((batch * value_heads * (HEAD_DIM // tile_v),), -1, dtype=torch.int32),
    }
    for row in range(batch):
        slot = int(case["slots"][row])
        for i in range(position):
            phys = (base + i) & (ring_len - 1)
            case["d_cache"][slot, :, phys] = rand(value_heads, HEAD_DIM, dtype=torch.float16, multiplier=0.02)
            case["k_cache"][slot, :, phys] = rand(heads, HEAD_DIM, dtype=torch.float16, multiplier=0.09)
            case["g_cache"][slot, :, phys] = -(torch.rand(value_heads, generator=generator) * 0.04)
    if padding:
        case["slots"][-1] = 0
        case["positions"][-1] = 0
        case["bases"][-1] = 0
        case["lengths"][-1] = 0
    return case


def clone_case(case: dict[str, torch.Tensor], device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    return {name: tensor.to(device).clone() for name, tensor in case.items()}


def clone_page_strided_case(
    case: dict[str, torch.Tensor], device: str | torch.device = "cpu"
) -> dict[str, torch.Tensor]:
    """Put all four state views on one padded physical page per slot."""
    result = {name: tensor.to(device).clone() for name, tensor in case.items() if name not in CACHE_NAMES}
    slots = case["checkpoint"].shape[0]
    region_bytes = {name: case[name][0].numel() * case[name].element_size() for name in CACHE_NAMES}
    # Reserve a conv-state region before the SSM data, as in the hybrid cache.
    conv_bytes = 16 * 1024
    page_alignment = 256 * 1024
    total_bytes = conv_bytes + sum(region_bytes.values())
    page_bytes = ((total_bytes + page_alignment - 1) // page_alignment) * page_alignment
    raw = torch.empty(slots * page_bytes, dtype=torch.uint8, device=device)
    offset_bytes = conv_bytes
    for name in CACHE_NAMES:
        source = case[name]
        typed = raw.view(source.dtype)
        stride = (page_bytes // source.element_size(), *source.stride()[1:])
        view = torch.as_strided(
            typed,
            size=source.shape,
            stride=stride,
            storage_offset=offset_bytes // source.element_size(),
        )
        view.copy_(source.to(device))
        result[name] = view
        offset_bytes += region_bytes[name]
    return result


def _validate_row(case: dict[str, torch.Tensor], row: int) -> tuple[int, int, int, int]:
    slot, pos, base, length = (int(case[name][row]) for name in ("slots", "positions", "bases", "lengths"))
    ring_len = case["d_cache"].shape[2]
    if slot <= 0 or slot >= case["checkpoint"].shape[0]:
        raise ValueError("Golden requires a valid non-padding slot")
    if pos < 0 or pos > MAX_HISTORY or length < 1 or length > MAX_TOKENS or pos + length > min(MAX_WINDOW, ring_len):
        raise ValueError("Golden requires a verify window that fits the physical ring")
    if base < 0 or base >= ring_len:
        raise ValueError("Invalid circular base")
    return slot, pos, base, length


def recurrent_golden(case: dict[str, torch.Tensor], scale: float = HEAD_DIM**-0.5) -> dict[str, torch.Tensor]:
    """Independent dense-state recurrence; updates only candidate ring entries."""
    result = clone_case(case)
    result["out"].zero_()
    result["status"].zero_()
    heads = case["q"].shape[2]
    value_heads = case["v"].shape[2]
    group = value_heads // heads
    ring_len = case["d_cache"].shape[2]
    for row in range(case["q"].shape[0]):
        if int(case["slots"][row]) == 0:
            continue
        slot, pos, base, length = _validate_row(case, row)
        for h in range(value_heads):
            kh = h // group
            state = case["checkpoint"][slot, h].double()
            for i in range(pos):
                phys = (base + i) & (ring_len - 1)
                alpha = case["g_cache"][slot, h, phys].double().exp()
                d = case["d_cache"][slot, h, phys].double()
                key = case["k_cache"][slot, kh, phys].double()
                state = alpha * state + d[:, None] * key[None, :]
            for s in range(length):
                q = case["q"][row, s, kh].double() * scale
                key = case["k"][row, s, kh].double()
                value = case["v"][row, s, h].double()
                alpha = case["g"][row, s, h].double().exp()
                beta = case["beta"][row, s, h].double()
                decayed = alpha * state
                d = beta * (value - decayed @ key)
                state = decayed + d[:, None] * key[None, :]
                result["out"][row, s, h] = (state @ q).bfloat16()
                phys = (base + pos + s) & (ring_len - 1)
                result["d_cache"][slot, h, phys] = d.half()
                result["k_cache"][slot, kh, phys] = key.half()
                result["g_cache"][slot, h, phys] = case["g"][row, s, h]
    return result


def output_only_reference(case: dict[str, torch.Tensor], scale: float = HEAD_DIM**-0.5) -> torch.Tensor:
    """Reconstruct only S@q and S@k; never form a draft intermediate state."""
    out = torch.zeros_like(case["out"])
    value_heads = case["v"].shape[2]
    group = value_heads // case["q"].shape[2]
    ring_len = case["d_cache"].shape[2]
    for row in range(case["q"].shape[0]):
        if int(case["slots"][row]) == 0:
            continue
        slot, pos, base, length = _validate_row(case, row)
        for h in range(value_heads):
            kh = h // group
            checkpoint = case["checkpoint"][slot, h].double()
            records: list[tuple[torch.Tensor, torch.Tensor, float]] = []
            for i in range(pos):
                phys = (base + i) & (ring_len - 1)
                records.append(
                    (
                        case["d_cache"][slot, h, phys].double(),
                        case["k_cache"][slot, kh, phys].double(),
                        math.exp(float(case["g_cache"][slot, h, phys])),
                    )
                )
            for s in range(length):
                q = case["q"][row, s, kh].double() * scale
                key = case["k"][row, s, kh].double()
                alpha = math.exp(float(case["g"][row, s, h]))
                beta = float(case["beta"][row, s, h])
                decay = math.prod(item[2] for item in records)
                prev_q = decay * (checkpoint @ q)
                prev_k = decay * (checkpoint @ key)
                suffix = 1.0
                for d_old, key_old, alpha_old in reversed(records):
                    prev_q += suffix * d_old * (key_old @ q)
                    prev_k += suffix * d_old * (key_old @ key)
                    suffix *= alpha_old
                d = beta * (case["v"][row, s, h].double() - alpha * prev_k)
                out[row, s, h] = (alpha * prev_q + d * (key @ q)).bfloat16()
                records.append((d, key, alpha))
    return out


def check_result(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> dict[str, float]:
    errors: dict[str, float] = {}
    for name in ("out", *CACHE_NAMES):
        a, e = actual[name].cpu(), expected[name].cpu()
        torch.testing.assert_close(a, e, atol=0.01, rtol=0.03, equal_nan=True)
        finite = torch.isfinite(e)
        errors[name] = float((a[finite].float() - e[finite].float()).abs().max()) if finite.any() else 0.0
    torch.testing.assert_close(actual["status"].cpu(), expected["status"], rtol=0, atol=0)
    return errors


def invoke(
    case: dict[str, torch.Tensor],
    *,
    tile_v: int = 32,
    cores: int = 40,
    variant: int = 0,
    backend: str = "lab",
) -> None:
    if backend == "lab":
        op = torch.ops.replayssm_sd_lab.verify_
    elif backend == "framework":
        from vllm_ascend.ops.replayssm.ascendc import replayssm_sd_verify

        op = replayssm_sd_verify
    else:
        raise ValueError(f"Unknown backend: {backend}")
    op(
        *(
            case[name]
            for name in (
                "q",
                "k",
                "v",
                "g",
                "beta",
                "checkpoint",
                "d_cache",
                "k_cache",
                "g_cache",
                "slots",
                "positions",
                "bases",
                "lengths",
                "out",
                "status",
            )
        ),
        HEAD_DIM**-0.5,
        tile_v,
        cores,
        variant,
    )

"""CPU reference and deterministic Qwen3.5-35B-A3B TP2 operator cases."""

from __future__ import annotations

import torch

HISTORY = 16
RING = 32
KEY_HEADS = 8
VALUE_HEADS = 16
DIM = 128
CACHE = ("checkpoint", "d_cache", "k_cache", "g_cache")


def make_case(
    *,
    mtp: int = 3,
    batch: int = 1,
    pos: int = 7,
    base: int = 29,
    tile_v: int = 32,
    seed: int = 42,
    padding: bool = False,
) -> dict[str, torch.Tensor]:
    if mtp not in (3, 5) or batch < 1 or not 0 <= pos <= HISTORY:
        raise ValueError("Expected MTP3/5, positive batch and history in [0,16]")
    if not 0 <= base < RING or tile_v not in (32, 64):
        raise ValueError("Expected a 32-slot ring and tile_v=32/64")
    tokens = mtp + 1
    gen = torch.Generator().manual_seed(seed)

    def rand(*shape: int, dtype: torch.dtype = torch.float32, scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(*shape, generator=gen) * scale).to(dtype)

    slots = batch + 1
    case = {
        "q": rand(batch, tokens, KEY_HEADS, DIM, dtype=torch.float16, scale=0.09),
        "k": rand(batch, tokens, KEY_HEADS, DIM, dtype=torch.float16, scale=0.09),
        "v": rand(batch, tokens, VALUE_HEADS, DIM, dtype=torch.float16, scale=0.09),
        "g": -(torch.rand(batch, tokens, VALUE_HEADS, generator=gen) * 0.04),
        "beta": (torch.rand(batch, tokens, VALUE_HEADS, generator=gen) * 0.8).half(),
        "checkpoint": rand(slots, VALUE_HEADS, DIM, DIM, dtype=torch.float16, scale=0.02),
        "d_cache": torch.full((slots, VALUE_HEADS, RING, DIM), float("nan"), dtype=torch.float16),
        "k_cache": torch.full((slots, KEY_HEADS, RING, DIM), float("nan"), dtype=torch.float16),
        "g_cache": torch.full((slots, VALUE_HEADS, RING), float("nan"), dtype=torch.float32),
        "slots": torch.arange(1, slots, dtype=torch.int32),
        "ring_slots": torch.arange(1, slots, dtype=torch.int32),
        "positions": torch.full((batch,), pos, dtype=torch.int32),
        "bases": torch.full((batch,), base, dtype=torch.int32),
        "lengths": torch.full((batch,), tokens, dtype=torch.int32),
        "flush": torch.zeros(batch, dtype=torch.int32),
        "out": torch.zeros((batch, tokens, VALUE_HEADS, DIM), dtype=torch.float16),
        "verify_status": torch.full((batch * VALUE_HEADS * (DIM // tile_v),), -1, dtype=torch.int32),
        "compact_status": torch.full((batch * VALUE_HEADS * (DIM // 32),), -1, dtype=torch.int32),
    }
    for row in range(batch):
        slot = row + 1
        for i in range(pos):
            phys = (base + i) & (RING - 1)
            case["d_cache"][slot, :, phys] = rand(VALUE_HEADS, DIM, dtype=torch.float16, scale=0.02)
            case["k_cache"][slot, :, phys] = rand(KEY_HEADS, DIM, dtype=torch.float16, scale=0.09)
            case["g_cache"][slot, :, phys] = -(torch.rand(VALUE_HEADS, generator=gen) * 0.04)
    if padding:
        case["slots"][-1] = 0
        case["ring_slots"][-1] = 0
        case["positions"][-1] = 0
        case["bases"][-1] = 0
        case["lengths"][-1] = 0
    return case


def clone(case: dict[str, torch.Tensor], device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    return {key: value.to(device).clone() for key, value in case.items()}


def verify_reference(case: dict[str, torch.Tensor], scale: float = DIM**-0.5) -> dict[str, torch.Tensor]:
    result = clone(case)
    result["out"].zero_()
    result["verify_status"].zero_()
    for row in range(case["slots"].numel()):
        slot = int(case["slots"][row])
        if slot == 0:
            continue
        ring_slot = int(case["ring_slots"][row])
        pos, base, length = (int(case[name][row]) for name in ("positions", "bases", "lengths"))
        for head in range(VALUE_HEADS):
            key_head = head // (VALUE_HEADS // KEY_HEADS)
            state = case["checkpoint"][slot, head].double()
            for i in range(pos):
                phys = (base + i) & (RING - 1)
                alpha = case["g_cache"][ring_slot, head, phys].double().exp()
                delta = case["d_cache"][ring_slot, head, phys].double()
                key = case["k_cache"][ring_slot, key_head, phys].double()
                state = alpha * state + delta[:, None] * key[None, :]
            for token in range(length):
                query = case["q"][row, token, key_head].double() * scale
                key = case["k"][row, token, key_head].double()
                value = case["v"][row, token, head].double()
                alpha = case["g"][row, token, head].double().exp()
                beta = case["beta"][row, token, head].double()
                decayed = alpha * state
                delta = beta * (value - decayed @ key)
                state = decayed + delta[:, None] * key[None, :]
                result["out"][row, token, head] = (state @ query).half()
                phys = (base + pos + token) & (RING - 1)
                result["d_cache"][ring_slot, head, phys] = delta.half()
                result["k_cache"][ring_slot, key_head, phys] = key.half()
                result["g_cache"][ring_slot, head, phys] = case["g"][row, token, head]
    return result


def compact_reference(case: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = clone(case)
    result["compact_status"].zero_()
    for row in range(case["slots"].numel()):
        slot = int(case["slots"][row])
        if slot == 0 or not int(case["flush"][row]):
            continue
        ring_slot = int(case["ring_slots"][row])
        pos, base = int(case["positions"][row]), int(case["bases"][row])
        for head in range(VALUE_HEADS):
            key_head = head // (VALUE_HEADS // KEY_HEADS)
            state = case["checkpoint"][slot, head].float()
            for i in range(pos):
                phys = (base + i) & (RING - 1)
                alpha = case["g_cache"][ring_slot, head, phys].exp()
                delta = case["d_cache"][ring_slot, head, phys].float()
                key = case["k_cache"][ring_slot, key_head, phys].float()
                state = alpha * state + delta[:, None] * key[None, :]
            result["checkpoint"][slot, head] = state.half()
    return result


def cursor_reference(
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    accepted: torch.Tensor,
    slots: torch.Tensor,
    do_reset: torch.Tensor,
    mtp: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    wp, base, flush = write_pos.clone(), cache_base.clone(), is_flush.clone()
    status = torch.zeros_like(accepted)
    for row in range(slots.numel()):
        slot = int(slots[row])
        if slot == 0:
            continue
        if slot < 0 or slot >= wp.numel():
            status[row] = 1
            continue
        if int(do_reset[row]):
            wp[slot], base[slot], flush[slot] = 0, 0, int(2 * mtp > HISTORY)
            continue
        old_wp, old_base, old_flush = int(wp[slot]), int(base[slot]), int(flush[slot])
        n = int(accepted[row])
        if n < 0 or n > mtp:
            status[row] = 2
        elif not (0 <= old_wp <= HISTORY and 0 <= old_base < RING and old_flush in (0, 1)):
            status[row] = 3
        elif n:
            new_wp = (0 if old_flush else old_wp) + n
            if new_wp > HISTORY:
                status[row] = 4
                continue
            base[slot] = (old_base + old_wp) & (RING - 1) if old_flush else old_base
            wp[slot] = new_wp
            flush[slot] = int(new_wp + 2 * mtp > HISTORY)
    return wp, base, flush, status


def fp16_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual, expected = actual.float(), expected.float()
    finite = torch.isfinite(expected)
    if not torch.equal(torch.isfinite(actual), finite):
        raise AssertionError("Finite mask mismatch")
    diff = (actual[finite] - expected[finite]).abs()
    matched = diff <= (2**-9 + 2**-9 * expected[finite].abs())
    ratio = float(matched.float().mean()) if diff.numel() else 1.0
    maximum = float(diff.max()) if diff.numel() else 0.0
    if ratio < 0.99 or maximum > 0.1:
        raise AssertionError(f"FP16 mismatch: matched={ratio:.6f}, max_abs={maximum:.6g}")
    return {"matched_ratio": ratio, "max_abs": maximum}

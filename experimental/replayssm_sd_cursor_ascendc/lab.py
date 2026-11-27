# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU contract for one SD accepted-prefix commit and first-decode reset."""

from __future__ import annotations

import torch

MUTABLE = ("write_pos", "cache_base", "is_flush", "status")
FLUSH_BIT = 1
FIRST_FLUSH_PENDING = 2


def make_case(
    *,
    batch: int = 4,
    slots: int = 17,
    logical_limit: int = 20,
    max_spec_len: int = 4,
    ring_len: int = 32,
) -> dict[str, torch.Tensor]:
    if batch < 1 or slots <= batch or logical_limit < 1 or max_spec_len < 1:
        raise ValueError("invalid batch/slot/window geometry")
    if ring_len < logical_limit or ring_len & (ring_len - 1):
        raise ValueError("ring_len must be a power of two covering logical_limit")
    case = {
        "write_pos": torch.zeros(slots, dtype=torch.int32),
        "cache_base": torch.zeros(slots, dtype=torch.int32),
        "is_flush": torch.zeros(slots, dtype=torch.int8),
        "accepted": torch.ones(batch, dtype=torch.int32),
        "slots": torch.arange(1, batch + 1, dtype=torch.int32),
        "do_reset": torch.zeros(batch, dtype=torch.int8),
        "status": torch.full((batch,), -1, dtype=torch.int32),
    }
    return case


def clone_case(case: dict[str, torch.Tensor], device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    return {name: value.to(device).clone() for name, value in case.items()}


def reference_step(
    case: dict[str, torch.Tensor],
    *,
    logical_limit: int = 20,
    max_spec_len: int = 4,
    ring_len: int = 32,
    stagger_first: bool = False,
) -> dict[str, torch.Tensor]:
    """Match the device kernel; operate only on CPU tensors."""
    if stagger_first and logical_limit < 2 * max_spec_len:
        raise ValueError("Staggering requires room for two verify windows")

    def next_flags(slot: int, pos: int, pending: bool) -> int:
        limit = logical_limit
        if pending:
            phases = 1 + (logical_limit - 2 * max_spec_len) // max_spec_len
            index = slot - 1
            limit -= ((index + index // phases) % phases) * max_spec_len
        return (FIRST_FLUSH_PENDING if pending else 0) | int(pos + 2 * max_spec_len > limit)

    out = clone_case(case)
    out["status"].zero_()
    num_slots = out["write_pos"].numel()
    for row in range(out["slots"].numel()):
        slot = int(out["slots"][row])
        if slot == 0:
            continue
        if slot < 0 or slot >= num_slots:
            out["status"][row] = 1
            continue
        if int(out["do_reset"][row]):
            out["write_pos"][slot] = 0
            out["cache_base"][slot] = 0
            out["is_flush"][slot] = next_flags(slot, 0, stagger_first)
            continue
        old_pos = int(out["write_pos"][slot])
        old_base = int(out["cache_base"][slot])
        old_flush = int(out["is_flush"][slot])
        accepted = int(out["accepted"][row])
        if accepted < 0 or accepted > max_spec_len:
            out["status"][row] = 2
            continue
        max_flags = FLUSH_BIT | FIRST_FLUSH_PENDING if stagger_first else FLUSH_BIT
        if not 0 <= old_pos <= logical_limit or not 0 <= old_base < ring_len or not 0 <= old_flush <= max_flags:
            out["status"][row] = 3
            continue
        if accepted == 0:
            continue
        folded = bool(old_flush & FLUSH_BIT)
        next_pos = (0 if folded else old_pos) + accepted
        if next_pos > logical_limit:
            out["status"][row] = 4
            continue
        out["cache_base"][slot] = ((old_base + old_pos) & (ring_len - 1)) if folded else old_base
        out["write_pos"][slot] = next_pos
        out["is_flush"][slot] = next_flags(slot, next_pos, not folded and bool(old_flush & FIRST_FLUSH_PENDING))
    return out


def invoke(
    case: dict[str, torch.Tensor],
    logical_limit: int = 20,
    max_spec_len: int = 4,
    ring_len: int = 32,
    backend: str = "lab",
    stagger_first: bool = False,
) -> None:
    if backend == "lab":
        op = torch.ops.replayssm_sd_cursor_lab.step_
    elif backend == "framework":
        from vllm_ascend.ops.replayssm.ascendc import replayssm_sd_cursor

        op = replayssm_sd_cursor
    else:
        raise ValueError(f"Unknown backend: {backend}")
    if stagger_first and backend != "lab":
        raise ValueError("Staggered cursor is available through the experimental binding")
    kwargs = {"stagger_first": True} if stagger_first else {}
    op(
        case["write_pos"],
        case["cache_base"],
        case["is_flush"],
        case["accepted"],
        case["slots"],
        case["do_reset"],
        case["status"],
        logical_limit,
        max_spec_len,
        ring_len,
        **kwargs,
    )

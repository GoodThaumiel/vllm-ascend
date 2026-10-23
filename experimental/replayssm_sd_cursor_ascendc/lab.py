# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU contract for one SD accepted-prefix commit and first-decode reset."""

from __future__ import annotations

import torch

MUTABLE = ("write_pos", "cache_base", "is_flush", "status")


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
    case: dict[str, torch.Tensor], *, logical_limit: int = 20, max_spec_len: int = 4, ring_len: int = 32
) -> dict[str, torch.Tensor]:
    """Match the device kernel; operate only on CPU tensors."""
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
            out["is_flush"][slot] = int(2 * max_spec_len > logical_limit)
            continue
        old_pos = int(out["write_pos"][slot])
        old_base = int(out["cache_base"][slot])
        old_flush = int(out["is_flush"][slot])
        accepted = int(out["accepted"][row])
        if accepted < 0 or accepted > max_spec_len:
            out["status"][row] = 2
            continue
        if not 0 <= old_pos <= logical_limit or not 0 <= old_base < ring_len or old_flush not in (0, 1):
            out["status"][row] = 3
            continue
        if accepted == 0:
            continue
        next_pos = (0 if old_flush else old_pos) + accepted
        if next_pos > logical_limit:
            out["status"][row] = 4
            continue
        out["cache_base"][slot] = ((old_base + old_pos) & (ring_len - 1)) if old_flush else old_base
        out["write_pos"][slot] = next_pos
        out["is_flush"][slot] = int(next_pos + 2 * max_spec_len > logical_limit)
    return out


def invoke(
    case: dict[str, torch.Tensor],
    logical_limit: int = 20,
    max_spec_len: int = 4,
    ring_len: int = 32,
    backend: str = "lab",
) -> None:
    if backend == "lab":
        op = torch.ops.replayssm_sd_cursor_lab.step_
    elif backend == "framework":
        from vllm_ascend.ops.replayssm.ascendc import replayssm_sd_cursor

        op = replayssm_sd_cursor
    else:
        raise ValueError(f"Unknown backend: {backend}")
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
    )

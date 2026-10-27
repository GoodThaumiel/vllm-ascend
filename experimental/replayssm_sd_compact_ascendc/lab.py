# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU contract for folding committed Replay SD history into the checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experimental.replayssm_sd_ascendc.lab import (  # noqa: E402
    CACHE_NAMES,
    clone_case,
    clone_page_strided_case,
)
from experimental.replayssm_sd_ascendc.lab import (  # noqa: E402
    make_case as make_verify_case,
)

__all__ = ["CACHE_NAMES", "clone_case", "clone_page_strided_case"]


def make_case(
    *,
    tp: int = 4,
    batch: int = 2,
    position: int = 13,
    base: int = 31,
    ring_len: int = 32,
    tile_v: int = 32,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    if position < 0 or position > 16 or position + 4 > ring_len:
        raise ValueError("Need 0<=position<=16 and room for four draft slots")
    case = make_verify_case(
        tp=tp,
        batch=batch,
        position=position,
        length=4,
        base=base,
        ring_len=ring_len,
        tile_v=tile_v,
        seed=seed,
    )
    case["flush"] = torch.ones(batch, dtype=torch.int32)
    return case


def compact_golden(case: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = clone_case(case)
    ring_len = case["d_cache"].shape[2]
    group = case["checkpoint"].shape[1] // case["k_cache"].shape[1]
    for row in range(case["slots"].numel()):
        slot, pos, base, flush = (int(case[name][row]) for name in ("slots", "positions", "bases", "flush"))
        if slot == 0:
            if pos or base or flush:
                raise ValueError("Padding row metadata must be zero")
            continue
        if not 0 <= slot < case["checkpoint"].shape[0]:
            raise ValueError("Invalid slot")
        if not 0 <= pos <= 16 or not 0 <= base < ring_len or flush not in (0, 1):
            raise ValueError("Invalid compact metadata")
        if not flush:
            continue
        for head in range(case["checkpoint"].shape[1]):
            state = case["checkpoint"][slot, head].float().clone()
            key_head = head // group
            for i in range(pos):
                phys = (base + i) & (ring_len - 1)
                alpha = case["g_cache"][slot, head, phys].exp()
                d = case["d_cache"][slot, head, phys].float()
                k = case["k_cache"][slot, key_head, phys].float()
                state = alpha * state + d[:, None] * k[None, :]
            result["checkpoint"][slot, head] = state.bfloat16()
    result["status"].zero_()
    return result


def invoke(
    case: dict[str, torch.Tensor], *, tile_v: int = 32, cores: int = 40, variant: int = 0, backend: str = "lab"
) -> None:
    if backend == "lab":
        op = torch.ops.replayssm_sd_compact_lab.compact_
    elif backend == "framework":
        from vllm_ascend.ops.replayssm.ascendc import replayssm_sd_compact

        op = replayssm_sd_compact
    else:
        raise ValueError(f"Unknown backend: {backend}")
    op(
        case["checkpoint"],
        case["d_cache"],
        case["k_cache"],
        case["g_cache"],
        case["slots"],
        case["positions"],
        case["bases"],
        case["flush"],
        case["status"],
        tile_v,
        cores,
        variant,
    )


def compare(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> dict[str, float]:
    torch.testing.assert_close(actual["status"].cpu(), expected["status"], atol=0, rtol=0)
    errors = {}
    for name in CACHE_NAMES:
        left, right = actual[name].cpu(), expected[name].cpu()
        torch.testing.assert_close(
            left,
            right,
            atol=0.01 if name == "checkpoint" else 0,
            rtol=0.03 if name == "checkpoint" else 0,
            equal_nan=True,
        )
        mask = torch.isfinite(right)
        errors[name] = float((left[mask].float() - right[mask].float()).abs().max()) if mask.any() else 0.0
    return errors

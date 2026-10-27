# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Preallocated Replay AR operator; registration is owned by vllm_ascend_C."""

import torch


def replayssm_ar_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    checkpoint: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    slots: torch.Tensor,
    positions: torch.Tensor,
    flush: torch.Tensor,
    out: torch.Tensor,
    status: torch.Tensor,
    scale: float,
    tile_v: int = 32,
    cores: int = 1,
) -> None:
    """Mutate out/status and Replay buffers on the current NPU stream.

    The platform must register its compiled extension before tracing/capture.
    No allocations, host metadata reads, environment changes or lazy imports
    occur in this function. Slot uniqueness/lifecycle belong to the caller.
    """
    torch.ops._C_ascend.replayssm_ar_step_(
        q,
        k,
        v,
        g,
        beta,
        checkpoint,
        d_cache,
        k_cache,
        g_cache,
        slots,
        positions,
        flush,
        out,
        status,
        scale,
        tile_v,
        cores,
    )


def replayssm_sd_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    checkpoint: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    slots: torch.Tensor,
    positions: torch.Tensor,
    bases: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
    status: torch.Tensor,
    scale: float,
    tile_v: int = 32,
    cores: int = 40,
    variant: int = 0,
) -> None:
    """Verify a non-flush MTP window through the framework-owned SD op.

    All outputs and ring buffers are supplied by the caller. The fixed four-token
    layout and runtime lengths make this call graph-capturable; accepted-token
    commit and flush are separate stages and are not performed here.
    """
    torch.ops._C_ascend.replayssm_sd_verify_(
        q,
        k,
        v,
        g,
        beta,
        checkpoint,
        d_cache,
        k_cache,
        g_cache,
        slots,
        positions,
        bases,
        lengths,
        out,
        status,
        scale,
        tile_v,
        cores,
        variant,
    )


def replayssm_sd_compact(
    checkpoint: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    slots: torch.Tensor,
    positions: torch.Tensor,
    bases: torch.Tensor,
    flush: torch.Tensor,
    status: torch.Tensor,
    tile_v: int = 32,
    cores: int = 40,
    variant: int = 0,
) -> None:
    """Fold committed SD history into BF16 checkpoint on flush rows."""
    torch.ops._C_ascend.replayssm_sd_compact_(
        checkpoint,
        d_cache,
        k_cache,
        g_cache,
        slots,
        positions,
        bases,
        flush,
        status,
        tile_v,
        cores,
        variant,
    )


def replayssm_sd_cursor(
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    accepted: torch.Tensor,
    slots: torch.Tensor,
    do_reset: torch.Tensor,
    status: torch.Tensor,
    logical_limit: int = 20,
    max_spec_len: int = 4,
    ring_len: int = 32,
) -> None:
    """Commit accepted SD tokens or reset first-decode block-keyed cursors."""
    torch.ops._C_ascend.replayssm_sd_cursor_(
        write_pos,
        cache_base,
        is_flush,
        accepted,
        slots,
        do_reset,
        status,
        logical_limit,
        max_spec_len,
        ring_len,
    )

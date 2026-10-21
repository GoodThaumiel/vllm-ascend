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

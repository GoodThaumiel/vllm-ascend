# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""ReplaySSM cache contracts shared by config, metadata, and GDN dispatch."""

from __future__ import annotations

import torch

REPLAYSSM_AR_MAX_BUFFER_LEN = 64
REPLAYSSM_AR_HEAD_DIM = 128
REPLAYSSM_AR_TILE_V = 32
REPLAYSSM_AR_CORES = 8
REPLAYSSM_SD_MAX_TOKENS = 4
REPLAYSSM_SD_MAX_HISTORY = 16
REPLAYSSM_SD_VERIFY_TILE_V = 128
REPLAYSSM_SD_VERIFY_VARIANT = 13


def replayssm_sd_enabled(vllm_config) -> bool:
    """Ascend-only opt-in, applied after upstream vLLM config validation."""
    value = (getattr(vllm_config, "additional_config", None) or {}).get("replayssm_sd", False)
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError("additional_config.replayssm_sd must be a boolean")


def replayssm_sd_ring_len(buffer_len: int, num_spec: int) -> int:
    max_tokens = num_spec + 1
    if buffer_len != REPLAYSSM_SD_MAX_HISTORY or max_tokens != REPLAYSSM_SD_MAX_TOKENS:
        raise ValueError("Ascend ReplaySSM SD currently requires buffer_len=16 and MTP3 (T=4)")
    return 1 << (buffer_len + max_tokens - 1).bit_length()


def replayssm_sd_pack_indices(query_lengths: torch.Tensor, max_tokens: int = 4) -> torch.Tensor:
    """Map packed speculative tokens to fixed-width [request, token] rows."""
    if query_lengths.device.type != "cpu" or query_lengths.dim() != 1:
        raise ValueError("ReplaySSM SD query lengths must be a CPU vector")
    if bool(((query_lengths < 1) | (query_lengths > max_tokens)).any()):
        raise ValueError("ReplaySSM SD query length must be in [1, 4]")
    return torch.cat(
        [torch.arange(int(length), dtype=torch.int64) + row * max_tokens for row, length in enumerate(query_lengths)]
    )


def prepare_replayssm_decode_base_cpu(
    decode_base_cpu: torch.Tensor,
    *,
    is_dummy_run: bool,
) -> torch.Tensor:
    """Return the request ring origin used by Ascend attention metadata.

    Real execution must preserve the runner-owned pinned buffer. Graph warmup
    and capture use synthetic requests, so their ring origin is always zero
    and must not inherit stale request state from the input batch.
    """
    if decode_base_cpu.device.type != "cpu" or decode_base_cpu.dtype != torch.int32 or decode_base_cpu.dim() != 1:
        raise ValueError("ReplaySSM decode base must be a 1D CPU int32 tensor")
    if is_dummy_run:
        return torch.zeros_like(decode_base_cpu)
    return decode_base_cpu


def append_gdn_replayssm_ar_dtypes(
    base_dtypes: tuple[torch.dtype, ...],
    model_dtype: torch.dtype,
) -> tuple[torch.dtype, ...]:
    """Append ``d``, ``k``, and ``g`` ring dtypes to a GDN cache spec."""
    if model_dtype != torch.bfloat16:
        raise ValueError("ReplaySSM AR on Ascend currently requires a BF16 model dtype")
    if len(base_dtypes) != 2 or base_dtypes[1] != torch.bfloat16:
        raise ValueError("ReplaySSM AR on Ascend requires a BF16 GDN checkpoint; set --mamba-ssm-cache-dtype bfloat16")
    if base_dtypes[0] != torch.bfloat16:
        raise ValueError("ReplaySSM AR on Ascend requires a BF16 convolution cache; set --mamba-cache-dtype bfloat16")
    return (*base_dtypes, torch.float16, torch.float16, torch.float32)


def append_gdn_replayssm_ar_shapes(
    base_shapes: tuple[tuple[int, ...], ...],
    *,
    num_k_heads: int,
    tp_size: int,
    buffer_len: int,
) -> tuple[tuple[int, ...], ...]:
    """Append the GDN ``d``, ``k``, and ``g`` ring shapes."""
    if len(base_shapes) != 2 or len(base_shapes[1]) != 3:
        raise ValueError(f"Unexpected GDN state shapes: {base_shapes}")
    if not 1 <= buffer_len <= REPLAYSSM_AR_MAX_BUFFER_LEN:
        raise ValueError(f"ReplaySSM AR buffer length must be in [1, {REPLAYSSM_AR_MAX_BUFFER_LEN}], got {buffer_len}")
    if num_k_heads % tp_size != 0:
        raise ValueError(f"ReplaySSM AR requires num_k_heads ({num_k_heads}) divisible by TP ({tp_size})")
    local_v_heads, head_v_dim, head_k_dim = base_shapes[1]
    if head_k_dim != REPLAYSSM_AR_HEAD_DIM or head_v_dim != REPLAYSSM_AR_HEAD_DIM:
        raise ValueError(
            "ReplaySSM AR AscendC currently supports only key/value head dimension 128, "
            f"got K={head_k_dim}, V={head_v_dim}"
        )
    local_k_heads = num_k_heads // tp_size
    return (
        *base_shapes,
        (local_v_heads, buffer_len, head_v_dim),
        (local_k_heads, buffer_len, head_k_dim),
        (local_v_heads, buffer_len),
    )


def reject_replayssm_ar_speculative_decode(vllm_config) -> None:
    """Keep AR and SD lifecycles separate until ReplaySSM SD is connected."""
    if not vllm_config.cache_config.use_replayssm:
        return
    cache_mode = vllm_config.cache_config.mamba_cache_mode
    if cache_mode not in ("none", "align"):
        raise ValueError(
            "ReplaySSM AR supports --mamba-cache-mode none or align. "
            "Mode 'all' moves the checkpoint every token without copying the Replay ring."
        )
    speculative_config = vllm_config.speculative_config
    num_spec = speculative_config.num_speculative_tokens if speculative_config is not None else 0
    if num_spec and not replayssm_sd_enabled(vllm_config):
        raise ValueError(
            "ReplaySSM AR is connected for target autoregressive decode only. "
            "Disable speculative decoding until the ReplaySSM SD commit/rollback path is enabled."
        )


def derive_replayssm_ar_cursors(
    *,
    num_computed_tokens: torch.Tensor,
    decode_base: torch.Tensor,
    query_lens: torch.Tensor,
    buffer_len: int,
    block_size: int,
    align_mode: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive host-side write positions and flush flags for one AR step.

    ``decode_base`` is reset by vLLM when a request enters or resumes decode.
    In align mode, block starts are also checkpoints; the step completing a
    block is forced to flush so the next block can start with an empty ring.
    """
    if not 1 <= buffer_len <= REPLAYSSM_AR_MAX_BUFFER_LEN:
        raise ValueError(f"ReplaySSM AR buffer length must be in [1, {REPLAYSSM_AR_MAX_BUFFER_LEN}], got {buffer_len}")
    if not (num_computed_tokens.shape == decode_base.shape == query_lens.shape and num_computed_tokens.dim() == 1):
        raise ValueError("ReplaySSM AR cursor inputs must be equally sized 1D tensors")
    if align_mode and block_size <= 0:
        raise ValueError("ReplaySSM AR align mode requires a positive block size")

    effective_base = decode_base
    if align_mode:
        effective_base = torch.maximum(
            decode_base,
            torch.div(num_computed_tokens, block_size, rounding_mode="floor") * block_size,
        )
    decode_steps = num_computed_tokens - effective_base
    valid_rows = query_lens > 0
    # A one-token extend can be routed through a captured decode graph before
    # its new decode base becomes visible. Rebuild it directly from the current
    # checkpoint and materialize the result immediately.
    checkpoint_only_rows = valid_rows & (decode_steps < 0)
    decode_steps = torch.where(
        valid_rows & ~checkpoint_only_rows,
        decode_steps,
        torch.zeros_like(decode_steps),
    )
    write_pos = torch.remainder(decode_steps, buffer_len).to(torch.int32)
    is_flush = (write_pos == buffer_len - 1) | checkpoint_only_rows
    if align_mode:
        is_flush |= valid_rows & (torch.remainder(num_computed_tokens + query_lens, block_size) == 0)
    return write_pos, is_flush.to(torch.int32)

# SPDX-License-Identifier: Apache-2.0
"""Serve adapter for the V12 310P ReplaySSM kernels."""

from pathlib import Path

import torch

from vllm_ascend._310p.ops.replayssm_sd_trace import active_trace
from vllm_ascend.ops.replayssm.cache import replayssm_sd_enabled

VERIFY_TILE_V = 64
COMPACT_TILE_V = 32
CORES = 7
STATUS_VARIANT = 1
METADATA_ALIGNMENT_BYTES = 32


def load_replayssm_sd_310p_ops() -> None:
    root = Path(__file__).resolve().parents[3]
    library = root / "experimental/replayssm_sd_310p_ascendc/build/libreplayssm_sd_310p_torch.so"
    if not library.is_file():
        raise RuntimeError(f"Build the 310P ReplaySSM extension at {library} before starting the serve pilot")
    torch.ops.load_library(str(library))


def aligned_metadata_size(size: int, element_size: int) -> int:
    # The M200 kernels load and update complete 32-byte metadata blocks.
    alignment = METADATA_ALIGNMENT_BYTES // element_size
    return ((size + alignment - 1) // alignment) * alignment


def replay_token_gather(pack: torch.Tensor, rows: int, width: int) -> torch.Tensor:
    """Invert packed-to-dense indices. Invalid lanes may reuse input token zero."""
    gather = pack.new_zeros(rows * width)
    gather[pack] = torch.arange(pack.numel(), dtype=pack.dtype, device=pack.device)
    return gather


def initialize_replayssm_sd_310p(layer, vllm_config) -> None:
    layer.use_replayssm_sd = replayssm_sd_enabled(vllm_config)
    if not layer.use_replayssm_sd:
        return
    layer._replayssm_sd_verify_width = vllm_config.speculative_config.num_speculative_tokens + 1
    key_heads = layer.num_k_heads // layer.tp_size
    value_heads = layer.num_v_heads // layer.tp_size
    if (key_heads, value_heads) != (8, 16) or layer.head_k_dim != 128 or layer.head_v_dim != 128:
        raise ValueError("310P ReplaySSM SD pilot requires local GDN geometry HK=8, HV=16, DK=DV=128")
    rows = max(
        vllm_config.scheduler_config.max_num_seqs,
        vllm_config.compilation_config.max_cudagraph_capture_size or 0,
    )
    for name, tile in (
        ("_replayssm_sd_verify_status", COMPACT_TILE_V),
        ("_replayssm_sd_compact_status", COMPACT_TILE_V),
    ):
        layer.register_buffer(
            name,
            torch.empty(rows * value_heads * (layer.head_v_dim // tile), dtype=torch.int32, device=layer.A_log.device),
            persistent=False,
        )


def replayssm_sd_310p_decode(layer, query, key, value, g, beta, metadata) -> torch.Tensor:
    """Consume normalized q/k and replace only the speculative recurrence."""
    slots = metadata.replayssm_sd_block_slots
    ring_slots = metadata.replayssm_sd_ring_slots
    positions = metadata.replayssm_sd_positions
    bases = metadata.replayssm_sd_bases
    rows = slots.numel()
    width = layer._replayssm_sd_verify_width
    uniform = value.shape[1] == rows * width
    pack = metadata.replayssm_sd_pack_indices

    def dense(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        flat = tensor.squeeze(0).to(dtype)
        if not uniform:
            flat = flat.index_select(0, metadata.replayssm_sd_token_gather)
        return flat.contiguous().view(rows, width, *flat.shape[1:])

    q, k, v, gate, rate = (
        dense(query, torch.float16),
        dense(key, torch.float16),
        dense(value, torch.float16),
        dense(g, torch.float32),
        dense(beta, torch.float16),
    )
    output = torch.empty_like(v)
    checkpoint = layer.kv_cache[1]
    rings = layer._replayssm_sd_d_ring, layer._replayssm_sd_k_ring, layer._replayssm_sd_g_ring
    tasks = rows * v.shape[2] * v.shape[3]
    ops = torch.ops.replayssm_sd_310p
    trace = active_trace()
    tile = VERIFY_TILE_V if trace is None else trace.tile
    variant = STATUS_VARIANT if trace is None else trace.variant
    if trace is not None:
        slot = int(slots.cpu()[0])
        ring_slot = int(ring_slots.cpu()[0])
        length = trace.record["length"]
        trace.layer(
            layer.prefix,
            q=q[0, :length],
            k=k[0, :length],
            v=v[0, :length],
            g=gate[0, :length],
            beta=rate[0, :length],
            state_before=checkpoint[slot],
            d_before=rings[0][ring_slot],
            k_before=rings[1][ring_slot],
            g_before=rings[2][ring_slot],
            slots=slots,
            ring_slots=ring_slots,
            positions=positions,
            bases=bases,
            lengths=metadata.replayssm_sd_lengths,
            flush=metadata.replayssm_sd_flush,
            cursor_status=metadata.replayssm_sd_cursor_status,
            reset=metadata.replayssm_sd_reset,
        )
    ops.verify_(
        q,
        k,
        v,
        gate,
        rate,
        checkpoint,
        *rings,
        slots,
        positions,
        bases,
        metadata.replayssm_sd_lengths,
        output,
        layer._replayssm_sd_verify_status[: tasks // tile],
        layer.head_k_dim**-0.5,
        tile,
        CORES,
        ring_slots,
        variant,
    )
    ops.compact_(
        checkpoint,
        *rings,
        slots,
        positions,
        bases,
        metadata.replayssm_sd_flush,
        layer._replayssm_sd_compact_status[: tasks // COMPACT_TILE_V],
        COMPACT_TILE_V,
        CORES,
        ring_slots,
    )
    if trace is not None:
        trace.layer(
            layer.prefix,
            state_after=checkpoint[slot],
            d_after=rings[0][ring_slot],
            k_after=rings[1][ring_slot],
            g_after=rings[2][ring_slot],
            verify_status=layer._replayssm_sd_verify_status[: tasks // tile],
            compact_status=layer._replayssm_sd_compact_status[: tasks // COMPACT_TILE_V],
        )
    output = output.flatten(0, 1)
    if not uniform:
        output = output.index_select(0, pack)
    if trace is not None:
        trace.layer(layer.prefix, gdn_out=output)
    return output.unsqueeze(0)

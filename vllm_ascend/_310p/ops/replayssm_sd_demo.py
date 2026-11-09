# SPDX-License-Identifier: Apache-2.0
"""310P ReplaySSM SD request-ring prototype and memory accounting.

The demo exercises the request-owned ring lifecycle without changing vLLM's
active cache spec or GDN dispatch. A serve integration must bind the ring to
the actual model layers before it can remove speculative Mamba blocks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from vllm_ascend.ops.replayssm.cache import ReplaySSMSDRingSlots


@dataclass(frozen=True)
class ReplaySSM310PGeometry:
    linear_layers: int
    local_key_heads: int
    local_value_heads: int
    max_num_seqs: int
    mtp_tokens: int
    history_len: int = 16
    head_dim: int = 128
    state_element_bytes: int = 2

    def __post_init__(self) -> None:
        if self.linear_layers < 1 or self.local_key_heads < 1 or self.max_num_seqs < 1:
            raise ValueError("linear_layers, local_key_heads and max_num_seqs must be positive")
        if self.local_value_heads < 1 or self.local_value_heads % self.local_key_heads:
            raise ValueError("local_value_heads must be a positive multiple of local_key_heads")
        if self.local_value_heads > 32 or self.head_dim != 128:
            raise ValueError("310P verify supports at most 32 local value heads of dimension 128")
        if self.mtp_tokens not in (3, 5) or self.history_len != 16 or self.state_element_bytes != 2:
            raise ValueError("310P pilot supports MTP3/5, history 16 and FP16 checkpoint")

    @property
    def ring_len(self) -> int:
        return 1 << (self.history_len + self.mtp_tokens).bit_length()


def geometry_from_hf_config(config: dict, *, tp: int, max_num_seqs: int, mtp_tokens: int) -> ReplaySSM310PGeometry:
    """Read the actual Qwen3.5 GDN layer/head counts from config.json."""
    text = config.get("text_config", config)
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list):
        raise ValueError("config.json must contain text_config.layer_types")
    linear_layers = sum("linear" in kind.lower() for kind in layer_types if isinstance(kind, str))
    key_heads = text.get("linear_num_key_heads")
    value_heads = text.get("linear_num_value_heads")
    if not isinstance(key_heads, int) or not isinstance(value_heads, int) or tp < 1:
        raise ValueError("config.json must contain integer linear head counts and tp must be positive")
    if key_heads % tp or value_heads % tp:
        raise ValueError("linear key/value head counts must be divisible by TP")
    return ReplaySSM310PGeometry(
        linear_layers=linear_layers,
        local_key_heads=key_heads // tp,
        local_value_heads=value_heads // tp,
        max_num_seqs=max_num_seqs,
        mtp_tokens=mtp_tokens,
    )


def estimate_memory(geometry: ReplaySSM310PGeometry, *, active_requests: int | None = None) -> dict:
    """Compare tensor payloads at equal active request count, per TP rank.

    Native holds 1+MTP FP16 states per request. Replay retains one checkpoint
    and allocates one D/K/G ring per request slot plus reserved slot zero.
    Attention KV, convolution state, allocator fragmentation and workspace are
    excluded; these are a capacity projection, not a scheduler measurement.
    """
    active = geometry.max_num_seqs if active_requests is None else active_requests
    if not 0 <= active <= geometry.max_num_seqs:
        raise ValueError("active_requests must be in [0, max_num_seqs]")
    layer_state = geometry.local_value_heads * geometry.head_dim * geometry.head_dim * geometry.state_element_bytes
    checkpoint_per_request = geometry.linear_layers * layer_state
    layer_ring_d = geometry.local_value_heads * geometry.ring_len * geometry.head_dim * 2
    layer_ring_k = geometry.local_key_heads * geometry.ring_len * geometry.head_dim * 2
    layer_ring_g = geometry.local_value_heads * geometry.ring_len * 4
    slots = geometry.max_num_seqs + 1
    ring_bytes = geometry.linear_layers * slots * (layer_ring_d + layer_ring_k + layer_ring_g)
    native_bytes = active * (geometry.mtp_tokens + 1) * checkpoint_per_request
    replay_bytes = active * checkpoint_per_request + ring_bytes
    return {
        "geometry": asdict(geometry),
        "active_requests": active,
        "checkpoint_bytes_per_request": checkpoint_per_request,
        "ring_bytes_per_slot": geometry.linear_layers * (layer_ring_d + layer_ring_k + layer_ring_g),
        "ring_capacity_including_reserved": slots,
        "ring_bytes_allocated": ring_bytes,
        "native_ssm_bytes": native_bytes,
        "replay_ssm_bytes": replay_bytes,
        "projected_saved_bytes": native_bytes - replay_bytes,
        "scope": "FP16 SSM tensor payload per TP rank; conv/KV/workspace excluded",
    }


class ReplaySSM310PRingDemo:
    """Request identity and ring allocation contract for the 310P demo."""

    def __init__(self, geometry: ReplaySSM310PGeometry):
        self.geometry = geometry
        self.slots = ReplaySSMSDRingSlots(geometry.max_num_seqs)
        self._active_request_ids: set[str] = set()

    def update(self, active_request_ids: list[str], finished_request_ids: list[str] = ()) -> dict[str, list[int]]:
        if len(set(active_request_ids)) != len(active_request_ids):
            raise ValueError("active_request_ids must be unique")
        self.slots.release(finished_request_ids)
        self._active_request_ids.difference_update(finished_request_ids)
        reset = [int(req_id not in self._active_request_ids) for req_id in active_request_ids]
        self.slots.assign(active_request_ids)
        self._active_request_ids.update(active_request_ids)
        return {"ring_slots": self.slots.slots_for(active_request_ids), "do_reset": reset}

    def allocate_rings(self, device: str = "cpu") -> list[tuple]:
        """Allocate the actual D/K/G ring shapes used by the 310P kernels."""
        import torch

        g = self.geometry
        slots = g.max_num_seqs + 1
        return [
            (
                torch.zeros((slots, g.local_value_heads, g.ring_len, g.head_dim), dtype=torch.float16, device=device),
                torch.zeros((slots, g.local_key_heads, g.ring_len, g.head_dim), dtype=torch.float16, device=device),
                torch.zeros((slots, g.local_value_heads, g.ring_len), dtype=torch.float32, device=device),
            )
            for _ in range(g.linear_layers)
        ]

    @staticmethod
    def allocated_bytes(rings: list[tuple]) -> int:
        return sum(t.numel() * t.element_size() for layer in rings for t in layer)

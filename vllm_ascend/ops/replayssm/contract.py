# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Dependency-free ReplaySSM autoregressive cache contract."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ReplaySSMARLayout:
    """Shapes and cursor rules for one GDN ReplaySSM AR layer.

    Slot zero is reserved for padded rows. Active decode rows use positive
    state indices, matching vLLM's recurrent cache convention.
    """

    batch: int
    num_slots: int
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    capacity: int

    def __post_init__(self) -> None:
        values = (
            self.batch,
            self.num_slots,
            self.key_heads,
            self.value_heads,
            self.key_dim,
            self.value_dim,
            self.capacity,
        )
        if any(value <= 0 for value in values):
            raise ValueError("ReplaySSM dimensions must be positive")
        if self.num_slots < 2:
            raise ValueError("ReplaySSM requires slot zero plus at least one active slot")
        if self.value_heads % self.key_heads != 0:
            raise ValueError("value_heads must be divisible by key_heads")

    @property
    def packed_qkv_dim(self) -> int:
        return 2 * self.key_heads * self.key_dim + self.value_heads * self.value_dim

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "mixed_qkv": (self.batch, self.packed_qkv_dim),
            "a": (self.batch, self.value_heads),
            "b": (self.batch, self.value_heads),
            "checkpoint": (self.num_slots, self.value_heads, self.value_dim, self.key_dim),
            "d_cache": (self.num_slots, self.value_heads, self.capacity, self.value_dim),
            "k_cache": (self.num_slots, self.key_heads, self.capacity, self.key_dim),
            "g_cache": (self.num_slots, self.value_heads, self.capacity),
            "out": (self.batch, 1, self.value_heads, self.value_dim),
            "state_indices": (self.batch,),
            "write_pos": (self.batch,),
            "is_flush": (self.batch,),
        }

    def validate_metadata(
        self,
        state_indices: Sequence[int],
        write_pos: Sequence[int],
        is_flush: Sequence[bool | int],
    ) -> None:
        if not (len(state_indices) == len(write_pos) == len(is_flush) == self.batch):
            raise ValueError("ReplaySSM metadata lengths must equal batch")
        for row, (slot, position, flush) in enumerate(zip(state_indices, write_pos, is_flush, strict=True)):
            if slot < 0 or slot >= self.num_slots:
                raise ValueError(f"state_indices[{row}] is outside allocated slots")
            if position < 0 or position >= self.capacity:
                raise ValueError(f"write_pos[{row}] is outside the logical buffer")
            if slot == 0 and position != 0:
                raise ValueError(f"padded row {row} must have write_pos zero")
            if slot > 0 and position == self.capacity - 1 and not bool(flush):
                raise ValueError(f"row {row} must flush at the final logical position")

    def advance(self, write_pos: int, is_flush: bool) -> int:
        if write_pos < 0 or write_pos >= self.capacity:
            raise ValueError("write_pos is outside the logical buffer")
        if is_flush:
            return 0
        if write_pos == self.capacity - 1:
            raise ValueError("the final logical position must flush")
        return write_pos + 1

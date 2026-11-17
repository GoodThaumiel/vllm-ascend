# SPDX-License-Identifier: Apache-2.0
"""310P DFlash attention inputs owned by one KV group and graph descriptor."""

import torch


def draft_kv_group_order(
    draft_layer_names: list[str],
    kv_group_layer_names: list[list[str]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return physical group IDs and each drafter layer's index in that list."""
    draft_names = set(draft_layer_names)
    layer_to_group = {}
    for group_id, group_layers in enumerate(kv_group_layer_names):
        for layer in draft_names.intersection(group_layers):
            if layer in layer_to_group:
                raise ValueError(f"DFlash layer {layer!r} belongs to multiple KV groups")
            layer_to_group[layer] = group_id
    missing = draft_names - layer_to_group.keys()
    if not draft_layer_names or missing:
        raise ValueError(f"DFlash KV groups do not cover the draft layers: {sorted(missing)}")
    groups = tuple(sorted(set(layer_to_group.values())))
    group_to_index = {group: index for index, group in enumerate(groups)}
    return groups, tuple(group_to_index[layer_to_group[layer]] for layer in draft_layer_names)


def draft_additive_mask(
    seq_lens: torch.Tensor,
    columns: torch.Tensor,
    query_offsets: torch.Tensor,
    *,
    causal: bool,
    window: int | None,
) -> torch.Tensor:
    """Build the explicit SplitFuse mask before capture or replay."""
    lengths = seq_lens[:, None, None]
    width = query_offsets.numel()
    positions = lengths - width + query_offsets
    visible = columns < lengths
    if causal:
        visible = visible & (columns <= positions)
    if window is not None:
        visible = visible & (columns >= positions - window + 1)
        if not causal:
            visible = visible & (columns <= positions + window - 1)
    denied = ~visible.expand(seq_lens.numel(), width, -1)
    return denied.reshape(seq_lens.numel() * width, -1).to(torch.float16) * -10000.0


class DFlashSplitFuse310:
    """Persist exactly the tensors read by native SplitFuse across replays.

    The caller orders refresh against outstanding replay before mutating these
    buffers. Host query lengths are fixed at the descriptor width. Padding
    rows, when present, must already refer to readable scratch cache pages.
    """

    def __init__(
        self,
        *,
        rows: int,
        query_width: int,
        table_width: int,
        page_size: int,
        device: torch.device,
        causal: bool,
        window: int | None,
    ) -> None:
        if min(rows, query_width, table_width, page_size) <= 0 or (window is not None and window <= 0):
            raise ValueError("Invalid DFlash SplitFuse descriptor")

        import torch_npu

        from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, nd_to_nz_spec

        self._op = torch_npu._npu_paged_attention_splitfuse
        self._format_cast = torch_npu.npu_format_cast
        self._nz_format = ACL_FORMAT_FRACTAL_NZ
        self._nd_to_nz = nd_to_nz_spec
        self.causal = causal
        self.window = window
        self.num_query_tokens = rows * query_width
        self.query_lens_cpu = torch.full((rows,), query_width, dtype=torch.int32)
        self.seq_lens = torch.empty(rows, dtype=torch.int32, device=device)
        self.block_tables = torch.empty((rows, table_width), dtype=torch.int32, device=device)
        self._columns = torch.arange(table_width * page_size, dtype=torch.int32, device=device).view(1, 1, -1)
        self._query_offsets = torch.arange(query_width, dtype=torch.int32, device=device).view(1, -1, 1)
        mask_nd = torch.zeros((self.num_query_tokens, self._columns.numel()), dtype=torch.float16, device=device)
        self.mask = self._as_nz(mask_nd)

    def _as_nz(self, mask: torch.Tensor) -> torch.Tensor:
        return self._format_cast(self._nd_to_nz(mask).contiguous(), self._nz_format)

    def refresh(self, seq_lens: torch.Tensor, block_tables: torch.Tensor) -> None:
        """Update metadata in place outside graph capture."""
        if seq_lens.shape != self.seq_lens.shape or block_tables.shape != self.block_tables.shape:
            raise ValueError("DFlash SplitFuse metadata must match the fixed descriptor")
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("DFlash SplitFuse mask refresh must run outside ACLGraph capture")
        self.seq_lens.copy_(seq_lens)
        self.block_tables.copy_(block_tables)
        mask = draft_additive_mask(
            self.seq_lens,
            self._columns,
            self._query_offsets,
            causal=self.causal,
            window=self.window,
        )
        self.mask.copy_(self._as_nz(mask))

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        num_kv_heads: int,
        num_heads: int,
        scale: float,
        output: torch.Tensor,
    ) -> None:
        if query.shape[0] != self.num_query_tokens or output.shape != query.shape:
            raise ValueError("DFlash SplitFuse query and output must match the fixed descriptor")
        self._op(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            mask=self.mask,
            block_table=self.block_tables,
            seq_len=self.query_lens_cpu,
            context_lens=self.seq_lens,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            scale_value=scale,
            out=output,
        )

# SPDX-License-Identifier: Apache-2.0
"""Scope the 310P MRv1 compatibility change to DFlash draft model construction."""

from contextvars import ContextVar

DFLASH_MRV1_LOADING: ContextVar[bool] = ContextVar("dflash_mrv1_loading", default=False)


def dflash_layer_attention(config, layer_index, upstream, layer_causal):
    layers = getattr(config, "layer_types", None) or ()
    sliding_count = layers.count("sliding_attention")
    if not DFLASH_MRV1_LOADING.get() or not 0 < sliding_count < len(layers):
        return upstream(config, layer_index)
    if layers[layer_index] != "sliding_attention":
        return None, layer_causal(config, layer_index)
    options = getattr(config, "dflash_config", None) or {}
    window = options.get("swa_window_size", getattr(config, "sliding_window", None))
    if window is None or window <= 0:
        raise ValueError("310P DFlash sliding layers require a positive window")
    return window, layer_causal(config, layer_index)


def usable_decode_sizes(capture_sizes, max_requests: int, max_tokens: int, width: int) -> frozenset[int]:
    """Select exact, unpadded DFlash decode shapes from the configured sizes."""
    limit = min(max_requests * width, max_tokens)
    return frozenset(size for size in capture_sizes or () if 0 < size <= limit and size % width == 0)


def retained_page_range(length: int, width: int, page_size: int, window: int | None) -> tuple[int, int]:
    """Keep every page visible to any query token in the draft block."""
    if length < width:
        raise ValueError("310P DFlash context is shorter than its query window")
    earliest_token = 0 if window is None else max(length - width - window + 1, 0)
    return earliest_token // page_size, (length + page_size - 1) // page_size


def validate_packed_context(offsets: list[int], rows: int, tokens: int) -> None:
    if (
        len(offsets) != rows + 1
        or offsets[0] != 0
        or offsets[-1] != tokens
        or any(left >= right for left, right in zip(offsets, offsets[1:]))
    ):
        raise ValueError(f"310P DFlash invalid packed context bounds: rows={rows}, tokens={tokens}")

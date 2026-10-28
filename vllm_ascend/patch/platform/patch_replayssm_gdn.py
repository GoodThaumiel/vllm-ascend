# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Expose the Qwen GDN ReplaySSM cache contract from the Ascend plugin."""

from __future__ import annotations

from collections.abc import Callable

from vllm.config import ModelConfig

from vllm_ascend.ops.replayssm.cache import (
    append_gdn_replayssm_ar_dtypes,
    append_gdn_replayssm_ar_shapes,
    reject_replayssm_ar_speculative_decode,
    replayssm_sd_enabled,
    replayssm_sd_ring_len,
)

_REPLAYSSM_GDN_ARCHITECTURES = frozenset(
    {
        "Qwen3NextForCausalLM",
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }
)


def _patch_config_methods(model_cls: type) -> None:
    if "get_mamba_state_shape_from_config" not in model_cls.__dict__:
        return
    original_shape = model_cls.get_mamba_state_shape_from_config.__func__
    original_dtype = model_cls.get_mamba_state_dtype_from_config.__func__
    if getattr(original_shape, "_ascend_replayssm_gdn", False):
        return

    def get_state_shape(cls, vllm_config, _original: Callable = original_shape):
        base_shapes = _original(cls, vllm_config)
        if not vllm_config.cache_config.use_replayssm:
            return base_shapes
        reject_replayssm_ar_speculative_decode(vllm_config)
        hf_config = vllm_config.model_config.hf_text_config
        buffer_len = vllm_config.cache_config.replayssm_buffer_len
        if replayssm_sd_enabled(vllm_config):
            buffer_len = replayssm_sd_ring_len(buffer_len, vllm_config.speculative_config.num_speculative_tokens)
        return append_gdn_replayssm_ar_shapes(
            base_shapes,
            num_k_heads=hf_config.linear_num_key_heads,
            tp_size=vllm_config.parallel_config.tensor_parallel_size,
            buffer_len=buffer_len,
        )

    def get_state_dtype(cls, vllm_config, _original: Callable = original_dtype):
        base_dtypes = _original(cls, vllm_config)
        if not vllm_config.cache_config.use_replayssm:
            return base_dtypes
        reject_replayssm_ar_speculative_decode(vllm_config)
        return append_gdn_replayssm_ar_dtypes(
            base_dtypes,
            vllm_config.model_config.dtype,
        )

    get_state_shape._ascend_replayssm_gdn = True
    get_state_dtype._ascend_replayssm_gdn = True
    model_cls.get_mamba_state_shape_from_config = classmethod(get_state_shape)
    model_cls.get_mamba_state_dtype_from_config = classmethod(get_state_dtype)


def _patch_qwen_gdn_models() -> None:
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForCausalLMBase,
        Qwen3_5ForConditionalGeneration,
    )
    from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM

    for model_cls in (
        Qwen3NextForCausalLM,
        Qwen3_5ForCausalLMBase,
        Qwen3_5ForConditionalGeneration,
    ):
        _patch_config_methods(model_cls)


_ORIGINAL_SUPPORTS_REPLAYSSM = ModelConfig.supports_replayssm.fget


def _supports_replayssm_on_ascend(self: ModelConfig) -> bool:
    assert _ORIGINAL_SUPPORTS_REPLAYSSM is not None
    return _ORIGINAL_SUPPORTS_REPLAYSSM(self) or self.architecture in _REPLAYSSM_GDN_ARCHITECTURES


ModelConfig.supports_replayssm = property(_supports_replayssm_on_ascend)
_patch_qwen_gdn_models()

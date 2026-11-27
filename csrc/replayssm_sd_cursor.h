// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
#pragma once

#include <ATen/ATen.h>

#define VLLM_ASCEND_REPLAYSSM_SD_CURSOR_SCHEMA                                     \
  "(Tensor(a!) write_pos, Tensor(b!) cache_base, Tensor(c!) is_flush, "         \
  "Tensor accepted, Tensor slots, Tensor do_reset, Tensor(d!) status, "         \
  "int logical_limit=20, int max_spec_len=4, int ring_len=32, bool stagger_first=False) -> ()"

namespace vllm_ascend {
// Experimental staggering uses is_flush bit 1 to track the first window.
// The consumer reads bit 0 for the actual flush decision.
void replayssm_sd_cursor(const at::Tensor& write_pos, const at::Tensor& cache_base,
                         const at::Tensor& is_flush, const at::Tensor& accepted,
                         const at::Tensor& slots, const at::Tensor& do_reset,
                         const at::Tensor& status, int64_t logical_limit,
                         int64_t max_spec_len, int64_t ring_len, bool stagger_first);

inline void replayssm_sd_cursor_meta(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                     const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                     const at::Tensor&, int64_t, int64_t, int64_t, bool) {}
}  // namespace vllm_ascend

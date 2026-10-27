// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
#pragma once

#include <ATen/ATen.h>

#define VLLM_ASCEND_REPLAYSSM_SD_COMPACT_SCHEMA                                               \
  "(Tensor(a!) checkpoint, Tensor d_cache, Tensor k_cache, Tensor g_cache, Tensor slots, "   \
  "Tensor positions, Tensor bases, Tensor flush, Tensor(b!) status, int tile_v=32, "         \
  "int cores=40, int variant=0) -> ()"

namespace vllm_ascend {
void replayssm_sd_compact(const at::Tensor& checkpoint, const at::Tensor& d_cache,
                          const at::Tensor& k_cache, const at::Tensor& g_cache,
                          const at::Tensor& slots, const at::Tensor& positions,
                          const at::Tensor& bases, const at::Tensor& flush,
                          const at::Tensor& status, int64_t tile_v, int64_t cores,
                          int64_t variant);

inline void replayssm_sd_compact_meta(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                      const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                      const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t,
                                      int64_t, int64_t) {}
}  // namespace vllm_ascend

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
#pragma once

#include <ATen/ATen.h>

#define VLLM_ASCEND_REPLAYSSM_SD_SCHEMA                                                     \
  "(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor checkpoint, "                \
  "Tensor(a!) d_cache, Tensor(b!) k_cache, Tensor(c!) g_cache, Tensor slots, "              \
  "Tensor positions, Tensor bases, Tensor lengths, Tensor(d!) out, Tensor(e!) status, "      \
  "float scale, int tile_v=32, int cores=40) -> ()"

namespace vllm_ascend {
void replayssm_sd_verify(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                         const at::Tensor& g, const at::Tensor& beta, const at::Tensor& checkpoint,
                         const at::Tensor& d_cache, const at::Tensor& k_cache, const at::Tensor& g_cache,
                         const at::Tensor& slots, const at::Tensor& positions, const at::Tensor& bases,
                         const at::Tensor& lengths, const at::Tensor& out, const at::Tensor& status,
                         double scale, int64_t tile_v, int64_t cores);

inline void replayssm_sd_verify_meta(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                     const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                     const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                     const at::Tensor&, const at::Tensor&, const at::Tensor&, double, int64_t,
                                     int64_t) {}
}  // namespace vllm_ascend

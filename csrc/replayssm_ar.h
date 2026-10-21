// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
#pragma once
#include <ATen/ATen.h>

#define VLLM_ASCEND_REPLAYSSM_AR_SCHEMA                                                 \
  "(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "                              \
  "Tensor(a!) checkpoint, Tensor(b!) d_cache, Tensor(c!) k_cache, Tensor(d!) g_cache, " \
  "Tensor slots, Tensor positions, Tensor flush, Tensor(e!) out, Tensor(f!) status, "   \
  "float scale, int tile_v=32, int cores=1) -> ()"

namespace vllm_ascend {
void replayssm_ar_step(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
                       const at::Tensor& beta, const at::Tensor& checkpoint, const at::Tensor& d_cache,
                       const at::Tensor& k_cache, const at::Tensor& g_cache, const at::Tensor& slots,
                       const at::Tensor& positions, const at::Tensor& flush, const at::Tensor& out,
                       const at::Tensor& status, double scale, int64_t tile_v, int64_t cores);

// All outputs are preallocated mutation arguments. Tracing allocates nothing
// and must not read device metadata or execute the NPU launcher.
inline void replayssm_ar_step_meta(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                   const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                   const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                   const at::Tensor&, const at::Tensor&, double, int64_t, int64_t) {}
}  // namespace vllm_ascend

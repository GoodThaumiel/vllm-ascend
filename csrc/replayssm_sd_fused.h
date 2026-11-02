// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
#pragma once

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

#define VLLM_ASCEND_REPLAYSSM_SD_FUSED_SCHEMA                                                    \
  "(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor(a!) checkpoint, "               \
  "Tensor(b!) d_cache, Tensor(c!) k_cache, Tensor(d!) g_cache, Tensor slots, "                   \
  "Tensor positions, Tensor bases, Tensor lengths, Tensor flush, Tensor(e!) out, "               \
  "Tensor(f!) verify_status, Tensor(g!) compact_status, float scale, "                           \
  "int tile_v=128, int cores=40, Tensor? ring_slots=None) -> ()"

namespace vllm_ascend {
void replayssm_sd_verify_compact_fused(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& g, const at::Tensor& beta, const at::Tensor& checkpoint,
    const at::Tensor& d_cache, const at::Tensor& k_cache, const at::Tensor& g_cache,
    const at::Tensor& slots, const at::Tensor& positions, const at::Tensor& bases,
    const at::Tensor& lengths, const at::Tensor& flush, const at::Tensor& out,
    const at::Tensor& verify_status, const at::Tensor& compact_status,
    double scale, int64_t tile_v, int64_t cores,
    c10::optional<at::Tensor> ring_slots);

inline void replayssm_sd_verify_compact_fused_meta(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, double, int64_t, int64_t, c10::optional<at::Tensor>) {}
}  // namespace vllm_ascend

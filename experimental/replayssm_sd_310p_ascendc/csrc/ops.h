// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

namespace replayssm_sd_310p {
void io_probe(const at::Tensor& input, const at::Tensor& output, int64_t mode);
void cast_probe(const at::Tensor& input, const at::Tensor& output, int64_t mode, int64_t count);
void verify(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
            const at::Tensor& g, const at::Tensor& beta,
            const at::Tensor& checkpoint, const at::Tensor& d_cache,
            const at::Tensor& k_cache, const at::Tensor& g_cache,
            const at::Tensor& slots, const at::Tensor& positions,
            const at::Tensor& bases, const at::Tensor& lengths,
            const at::Tensor& out, const at::Tensor& status, double scale,
            int64_t tile_v, int64_t cores, c10::optional<at::Tensor> ring_slots);

void compact(const at::Tensor& checkpoint, const at::Tensor& d_cache,
             const at::Tensor& k_cache, const at::Tensor& g_cache,
             const at::Tensor& slots, const at::Tensor& positions,
             const at::Tensor& bases, const at::Tensor& flush,
             const at::Tensor& status, int64_t tile_v, int64_t cores,
             c10::optional<at::Tensor> ring_slots);

void cursor(const at::Tensor& write_pos, const at::Tensor& cache_base,
            const at::Tensor& is_flush, const at::Tensor& accepted,
            const at::Tensor& slots, const at::Tensor& do_reset,
            const at::Tensor& status, int64_t logical_limit,
            int64_t max_spec_len, int64_t ring_len);
}  // namespace replayssm_sd_310p

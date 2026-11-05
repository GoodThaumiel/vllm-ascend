// SPDX-License-Identifier: Apache-2.0
#include "ops.h"

#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <cmath>

#include "aclrtlaunch_replayssm_sd_310p_verify4.h"
#include "aclrtlaunch_replayssm_sd_310p_verify6.h"

namespace replayssm_sd_310p {
void verify(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
            const at::Tensor& g, const at::Tensor& beta,
            const at::Tensor& checkpoint, const at::Tensor& d_cache,
            const at::Tensor& k_cache, const at::Tensor& g_cache,
            const at::Tensor& slots, const at::Tensor& positions,
            const at::Tensor& bases, const at::Tensor& lengths,
            const at::Tensor& out, const at::Tensor& status, double scale,
            int64_t tile_v, int64_t cores, c10::optional<at::Tensor> ring_slots) {
  const at::Tensor& ring_ids = ring_slots.has_value() ? *ring_slots : slots;
  TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "NPU tensors required");
  for (const auto& t : {k, v, g, beta, checkpoint, d_cache, k_cache, g_cache,
                        slots, ring_ids, positions, bases, lengths, out, status}) {
    TORCH_CHECK(t.device() == q.device(), "All tensors must be on one NPU");
  }
  for (const auto& t : {q, k, v, g, beta, slots, ring_ids, positions,
                        bases, lengths, out, status}) {
    TORCH_CHECK(t.is_contiguous(), "Non-cache tensors must be contiguous");
  }
  TORCH_CHECK(q.dim() == 4 && (q.size(1) == 4 || q.size(1) == 6) &&
                  q.size(3) == 128 && k.sizes() == q.sizes(),
              "q/k must be [B,4|6,H,128]");
  const int64_t batch = q.size(0), tokens = q.size(1), heads = q.size(2);
  TORCH_CHECK(v.dim() == 4 && v.size(0) == batch && v.size(1) == tokens &&
                  v.size(3) == 128, "v must be [B,4|6,HV,128]");
  const int64_t value_heads = v.size(2);
  TORCH_CHECK(batch > 0 && batch <= 1024 && heads > 0 && value_heads > 0 &&
                  value_heads <= 32 && value_heads % heads == 0,
              "Invalid batch/head geometry");
  TORCH_CHECK(g.sizes() == at::IntArrayRef({batch, tokens, value_heads}) &&
                  beta.sizes() == g.sizes() && out.sizes() == v.sizes(),
              "g/beta/out shape mismatch");
  TORCH_CHECK(checkpoint.dim() == 4 && checkpoint.size(0) > 1 &&
                  checkpoint.size(1) == value_heads && checkpoint.size(2) == 128 &&
                  checkpoint.size(3) == 128, "checkpoint shape mismatch");
  const int64_t num_slots = checkpoint.size(0);
  TORCH_CHECK(d_cache.dim() == 4 && d_cache.size(0) > 1 &&
                  d_cache.size(1) == value_heads && d_cache.size(2) == 32 &&
                  d_cache.size(3) == 128, "d_cache must be [ring_slots,HV,32,128]");
  const int64_t num_ring_slots = d_cache.size(0);
  TORCH_CHECK(k_cache.sizes() == at::IntArrayRef({num_ring_slots, heads, 32, 128}) &&
                  g_cache.sizes() == at::IntArrayRef({num_ring_slots, value_heads, 32}),
              "k/g ring shape mismatch");
  TORCH_CHECK(checkpoint.stride(3) == 1 && checkpoint.stride(2) == 128 &&
                  checkpoint.stride(1) == 128 * 128 &&
                  checkpoint.stride(0) >= value_heads * 128 * 128,
              "checkpoint slot layout invalid");
  TORCH_CHECK(d_cache.stride(3) == 1 && d_cache.stride(2) == 128 &&
                  d_cache.stride(1) == 32 * 128 &&
                  d_cache.stride(0) >= value_heads * 32 * 128,
              "d ring slot layout invalid");
  TORCH_CHECK(k_cache.stride(3) == 1 && k_cache.stride(2) == 128 &&
                  k_cache.stride(1) == 32 * 128 &&
                  k_cache.stride(0) >= heads * 32 * 128,
              "k ring slot layout invalid");
  TORCH_CHECK(g_cache.stride(2) == 1 && g_cache.stride(1) == 32 &&
                  g_cache.stride(0) >= value_heads * 32,
              "g ring slot layout invalid");
  for (const auto& t : {slots, ring_ids, positions, bases, lengths}) {
    TORCH_CHECK(t.dim() == 1 && t.numel() == batch && t.scalar_type() == at::kInt,
                "metadata must be int32[B]");
  }
  for (const auto& t : {q, k, v, beta, checkpoint, d_cache, k_cache, out}) {
    TORCH_CHECK(t.scalar_type() == at::kHalf, "q/k/v/beta/checkpoint/d/k/out must be FP16");
  }
  TORCH_CHECK(g.scalar_type() == at::kFloat && g_cache.scalar_type() == at::kFloat,
              "g/g_cache must be FP32");
  TORCH_CHECK(tile_v == 32 || tile_v == 64, "tile_v must be 32 or 64");
  TORCH_CHECK(status.dim() == 1 && status.numel() == batch * value_heads * (128 / tile_v) &&
                  status.scalar_type() == at::kInt,
              "status must be int32[B*HV*(128/tile_v)]");
  const int64_t heads_per_status_block = 8 / (128 / tile_v);
  TORCH_CHECK(value_heads % heads_per_status_block == 0,
              "value heads must align to a 32-byte status block");
  TORCH_CHECK(cores >= 1 && cores <= 7, "310P pilot cores must be in [1,7]");
  TORCH_CHECK(std::isfinite(scale) && scale > 0, "scale must be finite and positive");

  const c10_npu::OptionalNPUGuard guard(q.device());
  const auto stream = c10_npu::getCurrentNPUStream().stream();
  // A status block is owned by one core; the device task loop groups all
  // heads sharing that block before distributing groups across cores.
  const auto block_dim = static_cast<uint32_t>(std::min<int64_t>(
      cores, batch * value_heads / heads_per_status_block));
#define REPLAY_310P_ARGS \
  block_dim, stream, q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), \
      beta.data_ptr(), checkpoint.data_ptr(), d_cache.data_ptr(), k_cache.data_ptr(), \
      g_cache.data_ptr(), slots.data_ptr(), ring_ids.data_ptr(), positions.data_ptr(), \
      bases.data_ptr(), lengths.data_ptr(), out.data_ptr(), status.data_ptr(), \
      static_cast<uint32_t>(batch), static_cast<uint32_t>(num_slots), \
      static_cast<uint32_t>(num_ring_slots), static_cast<uint32_t>(heads), \
      static_cast<uint32_t>(value_heads), 32u, \
      static_cast<uint64_t>(checkpoint.stride(0)), static_cast<uint64_t>(d_cache.stride(0)), \
      static_cast<uint64_t>(k_cache.stride(0)), static_cast<uint64_t>(g_cache.stride(0)), \
      static_cast<uint32_t>(tile_v), static_cast<float>(scale)
  const aclError result = tokens == 4
      ? aclrtlaunch_replayssm_sd_310p_verify4(REPLAY_310P_ARGS)
      : aclrtlaunch_replayssm_sd_310p_verify6(REPLAY_310P_ARGS);
#undef REPLAY_310P_ARGS
  TORCH_CHECK(result == ACL_SUCCESS, "ReplaySSM SD 310P verify launch failed: ", result);
}
}  // namespace replayssm_sd_310p

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

#ifdef VLLM_ASCEND_ENABLE_REPLAYSSM_SD
#include "replayssm_sd_fused.h"
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <acl/acl.h>
#include "aclrtlaunch_replayssm_sd_verify_compact_fused.h"

#include <algorithm>
#include <cmath>

namespace vllm_ascend {
void replayssm_sd_verify_compact_fused(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& g, const at::Tensor& beta, const at::Tensor& checkpoint,
    const at::Tensor& d_cache, const at::Tensor& k_cache, const at::Tensor& g_cache,
    const at::Tensor& slots, const at::Tensor& positions, const at::Tensor& bases,
    const at::Tensor& lengths, const at::Tensor& flush, const at::Tensor& out,
    const at::Tensor& verify_status, const at::Tensor& compact_status,
    double scale, int64_t tile_v, int64_t cores,
    c10::optional<at::Tensor> ring_slots) {
  const at::Tensor& ring_ids = ring_slots.has_value() ? *ring_slots : slots;
  TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "NPU tensors required");
  for (const auto& t : {k, v, g, beta, checkpoint, d_cache, k_cache, g_cache,
                        slots, ring_ids, positions, bases, lengths, flush, out,
                        verify_status, compact_status}) {
    TORCH_CHECK(t.device() == q.device(), "All tensors must be on one NPU");
  }
  TORCH_CHECK(q.dim() == 4 && q.size(1) == 4 && q.size(3) == 128 &&
                  k.sizes() == q.sizes() && q.scalar_type() == at::kBFloat16 &&
                  k.scalar_type() == at::kBFloat16 && q.is_contiguous() && k.is_contiguous(),
              "q/k must be contiguous BF16[B,4,H,128]");
  const int64_t batch = q.size(0), heads = q.size(2);
  TORCH_CHECK(v.dim() == 4 && v.size(0) == batch && v.size(1) == 4 &&
                  v.size(3) == 128 && v.scalar_type() == at::kBFloat16 && v.is_contiguous(),
              "v must be contiguous BF16[B,4,HV,128]");
  const int64_t value_heads = v.size(2);
  TORCH_CHECK(batch > 0 && batch <= 1024 && heads > 0 && value_heads > 0 &&
                  value_heads <= 32 && value_heads % heads == 0,
              "Invalid batch/head dimensions");
  TORCH_CHECK((heads == 4 && value_heads == 12) || (heads == 2 && value_heads == 6),
              "This fusion ablation currently supports 27B TP4/TP8 only");
  TORCH_CHECK(g.sizes() == at::IntArrayRef({batch, 4, value_heads}) &&
                  beta.sizes() == g.sizes() && g.scalar_type() == at::kFloat &&
                  beta.scalar_type() == at::kFloat && g.is_contiguous() && beta.is_contiguous(),
              "g/beta must be contiguous FP32[B,4,HV]");
  TORCH_CHECK(out.sizes() == v.sizes() && out.scalar_type() == at::kBFloat16 &&
                  out.is_contiguous(), "out must match v");
  TORCH_CHECK(checkpoint.dim() == 4 && checkpoint.size(0) > 1 &&
                  checkpoint.size(1) == value_heads && checkpoint.size(2) == 128 &&
                  checkpoint.size(3) == 128 && checkpoint.scalar_type() == at::kBFloat16 &&
                  checkpoint.stride(3) == 1 && checkpoint.stride(2) == 128 &&
                  checkpoint.stride(1) == 128 * 128 &&
                  checkpoint.stride(0) >= value_heads * 128 * 128,
              "checkpoint layout must be BF16[slots,HV,128,128]");
  TORCH_CHECK(d_cache.dim() == 4 && d_cache.size(0) > 1 &&
                  d_cache.size(1) == value_heads && d_cache.size(3) == 128 &&
                  d_cache.scalar_type() == at::kHalf,
              "d_cache must be FP16[ring_slots,HV,R,128]");
  const int64_t ring_len = d_cache.size(2), num_ring_slots = d_cache.size(0);
  TORCH_CHECK(ring_len == 32 &&
                  k_cache.sizes() == at::IntArrayRef({num_ring_slots, heads, ring_len, 128}) &&
                  g_cache.sizes() == at::IntArrayRef({num_ring_slots, value_heads, ring_len}) &&
                  k_cache.scalar_type() == at::kHalf && g_cache.scalar_type() == at::kFloat,
              "MTP3 fusion requires FP16 d/k, FP32 g, and a 32-entry ring");
  TORCH_CHECK(d_cache.stride(3) == 1 && d_cache.stride(2) == 128 &&
                  d_cache.stride(1) == ring_len * 128 &&
                  d_cache.stride(0) >= value_heads * ring_len * 128 &&
                  k_cache.stride(3) == 1 && k_cache.stride(2) == 128 &&
                  k_cache.stride(1) == ring_len * 128 &&
                  k_cache.stride(0) >= heads * ring_len * 128 &&
                  g_cache.stride(2) == 1 && g_cache.stride(1) == ring_len &&
                  g_cache.stride(0) >= value_heads * ring_len,
              "ring slot layout invalid");
  for (const auto& t : {slots, ring_ids, positions, bases, lengths, flush}) {
    TORCH_CHECK(t.dim() == 1 && t.numel() == batch && t.scalar_type() == at::kInt &&
                    t.is_contiguous(), "metadata must be contiguous int32[B]");
  }
  TORCH_CHECK(tile_v == 128 && cores >= 1 && cores <= 40,
              "MTP3 fusion requires tile_v=128 and cores in [1,40]");
  TORCH_CHECK(verify_status.dim() == 1 && verify_status.numel() == batch * value_heads &&
                  verify_status.scalar_type() == at::kInt && verify_status.is_contiguous(),
              "verify_status must be int32[B*HV]");
  TORCH_CHECK(compact_status.dim() == 1 &&
                  compact_status.numel() == batch * value_heads * 4 &&
                  compact_status.scalar_type() == at::kInt && compact_status.is_contiguous(),
              "compact_status must be int32[B*HV*4]");
  TORCH_CHECK(std::isfinite(scale) && scale > 0, "scale must be finite and positive");

  const c10_npu::OptionalNPUGuard guard(q.device());
  const auto stream = c10_npu::getCurrentNPUStream().stream();
  const auto result = aclrtlaunch_replayssm_sd_verify_compact_fused(
      static_cast<uint32_t>(std::min(cores, batch * value_heads)), stream,
      q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
      checkpoint.data_ptr(), d_cache.data_ptr(), k_cache.data_ptr(), g_cache.data_ptr(),
      slots.data_ptr(), ring_ids.data_ptr(), positions.data_ptr(), bases.data_ptr(),
      lengths.data_ptr(), flush.data_ptr(), out.data_ptr(), verify_status.data_ptr(),
      compact_status.data_ptr(), static_cast<uint32_t>(batch),
      static_cast<uint32_t>(checkpoint.size(0)), static_cast<uint32_t>(num_ring_slots),
      static_cast<uint32_t>(heads), static_cast<uint32_t>(value_heads),
      static_cast<uint32_t>(ring_len), static_cast<uint64_t>(checkpoint.stride(0)),
      static_cast<uint64_t>(d_cache.stride(0)), static_cast<uint64_t>(k_cache.stride(0)),
      static_cast<uint64_t>(g_cache.stride(0)), static_cast<uint32_t>(tile_v),
      static_cast<float>(scale));
  TORCH_CHECK(result == ACL_SUCCESS, "Replay SD fused launch failed: ", result);
}
}  // namespace vllm_ascend
#endif  // VLLM_ASCEND_ENABLE_REPLAYSSM_SD

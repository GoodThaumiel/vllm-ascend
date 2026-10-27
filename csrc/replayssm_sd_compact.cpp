// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

#ifdef VLLM_ASCEND_ENABLE_REPLAYSSM_SD_COMPACT
#include "replayssm_sd_compact.h"
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <acl/acl.h>
#include "aclrtlaunch_replayssm_sd_compact.h"

#include <algorithm>

namespace vllm_ascend {
void replayssm_sd_compact(const at::Tensor& checkpoint, const at::Tensor& d_cache,
                          const at::Tensor& k_cache, const at::Tensor& g_cache,
                          const at::Tensor& slots, const at::Tensor& positions,
                          const at::Tensor& bases, const at::Tensor& flush,
                          const at::Tensor& status, int64_t tile_v, int64_t cores,
                          int64_t variant) {
  TORCH_CHECK(checkpoint.device().type() == c10::DeviceType::PrivateUse1, "NPU tensors required");
  for (const auto& t : {d_cache, k_cache, g_cache, slots, positions, bases, flush, status}) {
    TORCH_CHECK(t.device() == checkpoint.device(), "All tensors must be on one NPU");
  }
  TORCH_CHECK(checkpoint.dim() == 4 && checkpoint.size(0) > 1 && checkpoint.size(2) == 128 &&
                  checkpoint.size(3) == 128 && checkpoint.scalar_type() == at::kBFloat16,
              "checkpoint must be BF16[slots,HV,128,128]");
  const int64_t num_slots = checkpoint.size(0);
  const int64_t value_heads = checkpoint.size(1);
  TORCH_CHECK(d_cache.dim() == 4 && d_cache.size(0) == num_slots &&
                  d_cache.size(1) == value_heads && d_cache.size(3) == 128 &&
                  d_cache.scalar_type() == at::kHalf,
              "d_cache must be FP16[slots,HV,R,128]");
  const int64_t ring_len = d_cache.size(2);
  TORCH_CHECK(ring_len == 16 || ring_len == 32 || ring_len == 64, "R must be 16/32/64");
  TORCH_CHECK(k_cache.dim() == 4 && k_cache.size(0) == num_slots &&
                  k_cache.size(2) == ring_len && k_cache.size(3) == 128 &&
                  k_cache.scalar_type() == at::kHalf,
              "k_cache must be FP16[slots,H,R,128]");
  const int64_t heads = k_cache.size(1);
  TORCH_CHECK(heads > 0 && value_heads > 0 && value_heads % heads == 0,
              "Invalid head geometry");
  TORCH_CHECK(g_cache.sizes() == at::IntArrayRef({num_slots, value_heads, ring_len}) &&
                  g_cache.scalar_type() == at::kFloat,
              "g_cache must be FP32[slots,HV,R]");
  TORCH_CHECK(checkpoint.stride(3) == 1 && checkpoint.stride(2) == 128 &&
                  checkpoint.stride(1) == 128 * 128 &&
                  checkpoint.stride(0) >= value_heads * 128 * 128,
              "checkpoint slot layout invalid");
  TORCH_CHECK(d_cache.stride(3) == 1 && d_cache.stride(2) == 128 &&
                  d_cache.stride(1) == ring_len * 128 &&
                  d_cache.stride(0) >= value_heads * ring_len * 128,
              "d_cache slot layout invalid");
  TORCH_CHECK(k_cache.stride(3) == 1 && k_cache.stride(2) == 128 &&
                  k_cache.stride(1) == ring_len * 128 &&
                  k_cache.stride(0) >= heads * ring_len * 128,
              "k_cache slot layout invalid");
  TORCH_CHECK(g_cache.stride(2) == 1 && g_cache.stride(1) == ring_len &&
                  g_cache.stride(0) >= value_heads * ring_len,
              "g_cache slot layout invalid");
  TORCH_CHECK(slots.dim() == 1 && slots.numel() > 0 && slots.numel() <= 1024,
              "slots must be [B], B=1..1024");
  const int64_t batch = slots.numel();
  for (const auto& t : {slots, positions, bases, flush}) {
    TORCH_CHECK(t.dim() == 1 && t.numel() == batch && t.scalar_type() == at::kInt &&
                    t.is_contiguous(),
                "slots/positions/bases/flush must be contiguous int32[B]");
  }
  TORCH_CHECK(tile_v == 16 || tile_v == 32, "tile_v must be 16/32");
  const int64_t tasks = batch * value_heads * (128 / tile_v);
  TORCH_CHECK(status.dim() == 1 && status.numel() == tasks && status.scalar_type() == at::kInt &&
                  status.is_contiguous(),
              "status must be contiguous int32[B*HV*(128/tile_v)]");
  TORCH_CHECK(cores >= 1 && cores <= 40, "cores must be in [1,40]");
  TORCH_CHECK(variant >= 0 && variant <= 2, "variant must be 0 (original), 1 (grouped), or 2 (hybrid)");
  TORCH_CHECK(variant == 0 || (value_heads * (128 / tile_v)) % 8 == 0,
              "optimized compact requires a 32-byte aligned status row");
  TORCH_CHECK(variant == 0 || value_heads * (128 / tile_v) * sizeof(int32_t) <=
                                   tile_v * 128 * sizeof(at::BFloat16),
              "optimized compact status row exceeds the output UB buffer");

  const c10_npu::OptionalNPUGuard guard(checkpoint.device());
  const auto stream = c10_npu::getCurrentNPUStream().stream();
  const int64_t active_tasks = variant == 1 ? batch * value_heads : tasks;
  const auto result = aclrtlaunch_replayssm_sd_compact(
      static_cast<uint32_t>(std::min(cores, active_tasks)), stream, checkpoint.data_ptr(),
      d_cache.data_ptr(), k_cache.data_ptr(), g_cache.data_ptr(), slots.data_ptr(),
      positions.data_ptr(), bases.data_ptr(), flush.data_ptr(), status.data_ptr(),
      static_cast<uint32_t>(batch), static_cast<uint32_t>(num_slots),
      static_cast<uint32_t>(heads), static_cast<uint32_t>(value_heads),
      static_cast<uint32_t>(ring_len), static_cast<uint64_t>(checkpoint.stride(0)),
      static_cast<uint64_t>(d_cache.stride(0)), static_cast<uint64_t>(k_cache.stride(0)),
      static_cast<uint64_t>(g_cache.stride(0)), static_cast<uint32_t>(tile_v),
      static_cast<uint32_t>(variant));
  TORCH_CHECK(result == ACL_SUCCESS, "Replay SD compact launch failed: ", result);
}
}  // namespace vllm_ascend
#endif  // VLLM_ASCEND_ENABLE_REPLAYSSM_SD_COMPACT

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

#ifdef VLLM_ASCEND_ENABLE_REPLAYSSM_SD
#include "replayssm_sd.h"
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <acl/acl.h>
#include "aclrtlaunch_replayssm_sd_verify.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace vllm_ascend {
namespace {
constexpr int64_t K_DIM = 128;
constexpr int64_t MAX_TOKENS = 4;
constexpr int64_t MAX_HISTORY = 16;
}  // namespace

void replayssm_sd_verify(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
            const at::Tensor& beta, const at::Tensor& checkpoint, const at::Tensor& d_cache,
            const at::Tensor& k_cache, const at::Tensor& g_cache, const at::Tensor& slots,
            const at::Tensor& positions, const at::Tensor& bases, const at::Tensor& lengths,
            const at::Tensor& out, const at::Tensor& status, double scale, int64_t tile_v, int64_t cores) {
  TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "NPU tensors required");
  const std::vector<at::Tensor> tensors = {q,      k,         v,      g,      beta,   checkpoint, d_cache,
                                            k_cache, g_cache,   slots,  positions, bases,  lengths,    out, status};
  for (const auto& t : tensors) {
    TORCH_CHECK(t.device() == q.device(), "All tensors must be on one NPU");
  }
  for (size_t i : {0u, 1u, 2u, 3u, 4u, 9u, 10u, 11u, 12u, 13u, 14u}) {
    TORCH_CHECK(tensors[i].is_contiguous(), "Non-cache tensors must be contiguous");
  }
  TORCH_CHECK(q.dim() == 4 && k.sizes() == q.sizes() && q.size(1) == MAX_TOKENS && q.size(3) == K_DIM,
              "q/k must be [B,4,H,128]");
  TORCH_CHECK(v.dim() == 4 && v.size(0) == q.size(0) && v.size(1) == MAX_TOKENS && v.size(3) == K_DIM,
              "v must be [B,4,HV,128]");
  const int64_t batch = q.size(0), heads = q.size(2), value_heads = v.size(2);
  TORCH_CHECK(batch > 0 && batch <= 1024 && heads > 0 && value_heads > 0 && value_heads % heads == 0,
              "Invalid batch/head dimensions");
  TORCH_CHECK(g.sizes() == at::IntArrayRef({batch, MAX_TOKENS, value_heads}) && beta.sizes() == g.sizes(),
              "g/beta must be [B,4,HV]");
  TORCH_CHECK(out.sizes() == v.sizes(), "out must match v");
  TORCH_CHECK(checkpoint.dim() == 4 && checkpoint.size(0) > 1 && checkpoint.size(1) == value_heads &&
                  checkpoint.size(2) == K_DIM && checkpoint.size(3) == K_DIM,
              "checkpoint must be [slots,HV,128,128]");
  const int64_t num_slots = checkpoint.size(0);
  TORCH_CHECK(d_cache.dim() == 4 && d_cache.size(0) == num_slots && d_cache.size(1) == value_heads &&
                  d_cache.size(3) == K_DIM,
              "d_cache must be [slots,HV,R,128]");
  const int64_t ring_len = d_cache.size(2);
  TORCH_CHECK(ring_len == 16 || ring_len == 32 || ring_len == 64, "R must be 16/32/64");
  TORCH_CHECK(k_cache.sizes() == at::IntArrayRef({num_slots, heads, ring_len, K_DIM}) &&
                  g_cache.sizes() == at::IntArrayRef({num_slots, value_heads, ring_len}),
              "k/g cache shape mismatch");
  TORCH_CHECK(checkpoint.stride(3) == 1 && checkpoint.stride(2) == K_DIM &&
                  checkpoint.stride(1) == K_DIM * K_DIM &&
                  checkpoint.stride(0) >= value_heads * K_DIM * K_DIM,
              "checkpoint slot layout invalid");
  TORCH_CHECK(d_cache.stride(3) == 1 && d_cache.stride(2) == K_DIM &&
                  d_cache.stride(1) == ring_len * K_DIM &&
                  d_cache.stride(0) >= value_heads * ring_len * K_DIM,
              "d_cache slot layout invalid");
  TORCH_CHECK(k_cache.stride(3) == 1 && k_cache.stride(2) == K_DIM &&
                  k_cache.stride(1) == ring_len * K_DIM && k_cache.stride(0) >= heads * ring_len * K_DIM,
              "k_cache slot layout invalid");
  TORCH_CHECK(g_cache.stride(2) == 1 && g_cache.stride(1) == ring_len &&
                  g_cache.stride(0) >= value_heads * ring_len,
              "g_cache slot layout invalid");
  for (const auto& t : {slots, positions, bases, lengths}) {
    TORCH_CHECK(t.dim() == 1 && t.numel() == batch && t.scalar_type() == at::kInt,
                "slots/positions/bases/lengths must be int32[B]");
  }
  for (const auto& t : {q, k, v, checkpoint, out}) {
    TORCH_CHECK(t.scalar_type() == at::kBFloat16, "q/k/v/checkpoint/out must be BF16");
  }
  TORCH_CHECK(d_cache.scalar_type() == at::kHalf && k_cache.scalar_type() == at::kHalf,
              "d/k cache must be FP16");
  TORCH_CHECK(g.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat &&
                  g_cache.scalar_type() == at::kFloat,
              "g/beta/g_cache must be FP32");
  TORCH_CHECK(tile_v == 16 || tile_v == 32, "tile_v must be 16/32 for this UB budget");
  const int64_t tasks = batch * value_heads * (K_DIM / tile_v);
  TORCH_CHECK(status.dim() == 1 && status.numel() == tasks && status.scalar_type() == at::kInt,
              "status must be int32[B*HV*(128/tile_v)]");
  TORCH_CHECK(cores >= 1 && cores <= 40, "cores must be in [1,40]");
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0, "scale must be finite and positive");
  const c10_npu::OptionalNPUGuard guard(q.device());
  const auto stream = c10_npu::getCurrentNPUStream().stream();
  const auto result = aclrtlaunch_replayssm_sd_verify(
      static_cast<uint32_t>(std::min(cores, tasks)), stream, q.data_ptr(), k.data_ptr(), v.data_ptr(),
      g.data_ptr(), beta.data_ptr(), checkpoint.data_ptr(), d_cache.data_ptr(), k_cache.data_ptr(),
      g_cache.data_ptr(), slots.data_ptr(), positions.data_ptr(), bases.data_ptr(), lengths.data_ptr(),
      out.data_ptr(), status.data_ptr(), static_cast<uint32_t>(batch), static_cast<uint32_t>(num_slots),
      static_cast<uint32_t>(heads), static_cast<uint32_t>(value_heads), static_cast<uint32_t>(ring_len),
      static_cast<uint64_t>(checkpoint.stride(0)), static_cast<uint64_t>(d_cache.stride(0)),
      static_cast<uint64_t>(k_cache.stride(0)), static_cast<uint64_t>(g_cache.stride(0)),
      static_cast<uint32_t>(tile_v), static_cast<float>(scale));
  TORCH_CHECK(result == ACL_SUCCESS, "Replay SD verify launch failed: ", result);
}

}  // namespace vllm_ascend
#endif  // VLLM_ASCEND_ENABLE_REPLAYSSM_SD

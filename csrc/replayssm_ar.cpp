// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

#ifdef VLLM_ASCEND_ENABLE_REPLAYSSM_AR
  #include <ATen/ATen.h>
  #include <torch/library.h>
  #include <torch_npu/csrc/core/npu/NPUGuard.h>
  #include <torch_npu/csrc/core/npu/NPUStream.h>
  #include <acl/acl.h>
  #include "aclrtlaunch_replayssm_ar.h"
  #include "replayssm_ar.h"

  #include <algorithm>
  #include <cmath>
  #include <limits>
  #include <vector>

namespace vllm_ascend {
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t MAX_CAPACITY = 64;

void replayssm_ar_step(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
                       const at::Tensor& beta, const at::Tensor& checkpoint, const at::Tensor& d_cache,
                       const at::Tensor& k_cache, const at::Tensor& g_cache, const at::Tensor& slots,
                       const at::Tensor& positions, const at::Tensor& flush, const at::Tensor& out,
                       const at::Tensor& status, double scale, int64_t tile_v, int64_t cores) {
  TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "NPU tensors required");
  std::vector<at::Tensor> tensors = {q,       k,       v,     g,         beta,  checkpoint, d_cache,
                                     k_cache, g_cache, slots, positions, flush, out,        status};
  for (const auto& t : tensors) {
    TORCH_CHECK(t.device() == q.device(), "All tensors must be on the same NPU");
  }
  // vLLM packs every state belonging to one physical slot into a shared page.
  // The four Replay views therefore have compact inner dimensions but a padded
  // stride(0). All other inputs remain dense.
  for (size_t i : {0u, 1u, 2u, 3u, 4u, 9u, 10u, 11u, 12u, 13u}) {
    TORCH_CHECK(tensors[i].is_contiguous(), "Replay AR non-cache tensors must be contiguous");
  }
  TORCH_CHECK(q.dim() == 3 && k.sizes() == q.sizes(), "q/k must have shape [B,H,128]");
  TORCH_CHECK(v.dim() == 3 && v.size(0) == q.size(0), "v must have shape [B,HV,128]");
  const int64_t batch = q.size(0), heads = q.size(1), value_heads = v.size(1);
  TORCH_CHECK(batch > 0 && batch <= 1024 && heads > 0 && value_heads > 0 && value_heads % heads == 0,
              "Invalid batch/head dimensions");
  TORCH_CHECK(q.size(2) == HEAD_DIM && v.size(2) == HEAD_DIM, "Only K=V=128 is supported");
  TORCH_CHECK(checkpoint.dim() == 4 && checkpoint.size(0) > 1 && checkpoint.size(1) == value_heads &&
                  checkpoint.size(2) == HEAD_DIM && checkpoint.size(3) == HEAD_DIM,
              "checkpoint must be [slots,HV,V,K]");
  TORCH_CHECK(d_cache.dim() == 4, "d_cache must be [slots,HV,L,V]");
  const int64_t num_slots = checkpoint.size(0), capacity = d_cache.size(2);
  TORCH_CHECK(capacity >= 1 && capacity <= MAX_CAPACITY, "L must be in [1,64]");
  TORCH_CHECK(d_cache.sizes() == at::IntArrayRef({num_slots, value_heads, capacity, HEAD_DIM}),
              "d_cache shape mismatch");
  TORCH_CHECK(k_cache.sizes() == at::IntArrayRef({num_slots, heads, capacity, HEAD_DIM}), "k_cache shape mismatch");
  TORCH_CHECK(g_cache.sizes() == at::IntArrayRef({num_slots, value_heads, capacity}), "g_cache shape mismatch");
  TORCH_CHECK(checkpoint.stride(3) == 1 && checkpoint.stride(2) == HEAD_DIM &&
                  checkpoint.stride(1) == HEAD_DIM * HEAD_DIM &&
                  checkpoint.stride(0) >= value_heads * HEAD_DIM * HEAD_DIM,
              "checkpoint must have compact [HV,V,K] rows and a valid slot stride");
  TORCH_CHECK(d_cache.stride(3) == 1 && d_cache.stride(2) == HEAD_DIM &&
                  d_cache.stride(1) == capacity * HEAD_DIM &&
                  d_cache.stride(0) >= value_heads * capacity * HEAD_DIM,
              "d_cache must have compact [HV,L,V] rows and a valid slot stride");
  TORCH_CHECK(k_cache.stride(3) == 1 && k_cache.stride(2) == HEAD_DIM &&
                  k_cache.stride(1) == capacity * HEAD_DIM && k_cache.stride(0) >= heads * capacity * HEAD_DIM,
              "k_cache must have compact [H,L,K] rows and a valid slot stride");
  TORCH_CHECK(g_cache.stride(2) == 1 && g_cache.stride(1) == capacity &&
                  g_cache.stride(0) >= value_heads * capacity,
              "g_cache must have compact [HV,L] rows and a valid slot stride");
  TORCH_CHECK(g.sizes() == at::IntArrayRef({batch, value_heads}) && beta.sizes() == g.sizes(), "g/beta must be [B,HV]");
  for (const auto& t : {slots, positions, flush}) {
    TORCH_CHECK(t.dim() == 1 && t.numel() == batch && t.scalar_type() == at::kInt,
                "slots/positions/flush must be int32[B]");
  }
  for (const auto& t : {q, k, v, checkpoint, out}) {
    TORCH_CHECK(t.scalar_type() == at::kBFloat16, "q/k/v/checkpoint/out must be BF16");
  }
  TORCH_CHECK(d_cache.scalar_type() == at::kHalf && k_cache.scalar_type() == at::kHalf, "d/k caches must be FP16");
  TORCH_CHECK(g.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat && g_cache.scalar_type() == at::kFloat,
              "g/beta/g_cache must be FP32");
  TORCH_CHECK(out.sizes() == v.sizes(), "out must match v");

  const std::vector<at::Tensor> caches = {checkpoint, d_cache, k_cache, g_cache};
  for (size_t i = 0; i < caches.size(); ++i) {
    for (size_t j = i + 1; j < caches.size(); ++j) {
      if (!caches[i].is_alias_of(caches[j])) {
        continue;
      }
      const int64_t slot_stride_i = caches[i].stride(0) * static_cast<int64_t>(caches[i].element_size());
      const int64_t slot_stride_j = caches[j].stride(0) * static_cast<int64_t>(caches[j].element_size());
      TORCH_CHECK(slot_stride_i == slot_stride_j, "Aliased Replay caches must use one shared byte stride");
      const int64_t base_i = caches[i].storage_offset() * static_cast<int64_t>(caches[i].element_size());
      const int64_t base_j = caches[j].storage_offset() * static_cast<int64_t>(caches[j].element_size());
      TORCH_CHECK(base_i / slot_stride_i == base_j / slot_stride_j,
                  "Aliased Replay caches must begin in the same physical page");
      const int64_t start_i = base_i % slot_stride_i;
      const int64_t start_j = base_j % slot_stride_j;
      const int64_t bytes_i =
          (caches[i].numel() / num_slots) * static_cast<int64_t>(caches[i].element_size());
      const int64_t bytes_j =
          (caches[j].numel() / num_slots) * static_cast<int64_t>(caches[j].element_size());
      TORCH_CHECK(start_i + bytes_i <= slot_stride_i && start_j + bytes_j <= slot_stride_j,
                  "Aliased Replay cache region crosses a physical-page boundary");
      TORCH_CHECK(start_i + bytes_i <= start_j || start_j + bytes_j <= start_i,
                  "Aliased Replay cache regions overlap inside a physical page");
    }
  }
  TORCH_CHECK(tile_v == 16 || tile_v == 32 || tile_v == 64, "tile_v must be 16,32,64");
  TORCH_CHECK(cores > 0 && cores <= 64, "cores must be in [1,64]; use --cores 1 for first smoke");
  const int64_t tasks = batch * value_heads * (HEAD_DIM / tile_v);
  TORCH_CHECK(status.dim() == 1 && status.numel() == tasks && status.scalar_type() == at::kInt,
              "status must be int32[B*HV*(128/tile_v)]");
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0 && scale <= std::numeric_limits<float>::max(),
              "scale must be finite, positive and representable as FP32");
  for (const auto& t : tensors) {
    TORCH_CHECK(t.numel() <= std::numeric_limits<int32_t>::max(), "Experiment tensor is too large");
  }
  // The four cache views may intentionally alias one shared page backing, but
  // each has a disjoint per-slot byte range. Reject every other storage alias
  // using metadata only; no device-to-host synchronization occurs here.
  for (size_t i : {5u, 6u, 7u, 8u, 12u, 13u}) {
    for (size_t j = 0; j < tensors.size(); ++j) {
      const bool cache_pair = i >= 5 && i <= 8 && j >= 5 && j <= 8;
      if (i != j && !cache_pair) {
        TORCH_CHECK(!tensors[i].is_alias_of(tensors[j]), "Overlapping tensor storage is unsupported");
      }
    }
  }
  const c10_npu::OptionalNPUGuard guard(q.device());
  const auto stream = c10_npu::getCurrentNPUStream().stream();
  auto result = aclrtlaunch_replayssm_ar(
      static_cast<uint32_t>(std::min(cores, tasks)), stream, q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(),
      beta.data_ptr(), checkpoint.data_ptr(), d_cache.data_ptr(), k_cache.data_ptr(), g_cache.data_ptr(),
      slots.data_ptr(), positions.data_ptr(), flush.data_ptr(), out.data_ptr(), status.data_ptr(),
      static_cast<uint32_t>(batch), static_cast<uint32_t>(num_slots), static_cast<uint32_t>(heads),
      static_cast<uint32_t>(value_heads), static_cast<uint32_t>(capacity),
      static_cast<uint64_t>(checkpoint.stride(0)), static_cast<uint64_t>(d_cache.stride(0)),
      static_cast<uint64_t>(k_cache.stride(0)), static_cast<uint64_t>(g_cache.stride(0)),
      static_cast<uint32_t>(tile_v), static_cast<float>(scale));
  TORCH_CHECK(result == ACL_SUCCESS, "Replay AR launch failed: ", result);
}
}  // namespace vllm_ascend

#endif

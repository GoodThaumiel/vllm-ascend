// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

#ifdef VLLM_ASCEND_ENABLE_REPLAYSSM_SD_CURSOR
#include "replayssm_sd_cursor.h"
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "aclrtlaunch_replayssm_sd_cursor.h"

#include <cstdint>
#include <limits>

namespace vllm_ascend {
void replayssm_sd_cursor(const at::Tensor& write_pos, const at::Tensor& cache_base,
                         const at::Tensor& is_flush, const at::Tensor& accepted,
                         const at::Tensor& slots, const at::Tensor& do_reset,
                         const at::Tensor& status, int64_t logical_limit,
                         int64_t max_spec_len, int64_t ring_len) {
  TORCH_CHECK(write_pos.device().type() == c10::DeviceType::PrivateUse1, "NPU tensors required");
  TORCH_CHECK(write_pos.dim() == 1 && write_pos.numel() > 1 && write_pos.is_contiguous() &&
                  write_pos.scalar_type() == at::kInt,
              "write_pos must be contiguous int32[num_slots]");
  const auto num_slots = write_pos.numel();
  TORCH_CHECK(cache_base.sizes() == write_pos.sizes() && cache_base.is_contiguous() &&
                  cache_base.scalar_type() == at::kInt,
              "cache_base must be contiguous int32[num_slots]");
  TORCH_CHECK(is_flush.sizes() == write_pos.sizes() && is_flush.is_contiguous() &&
                  is_flush.scalar_type() == at::kChar,
              "is_flush must be contiguous int8[num_slots]");
  TORCH_CHECK(slots.dim() == 1 && slots.numel() > 0 && slots.is_contiguous() &&
                  slots.scalar_type() == at::kInt,
              "slots must be contiguous int32[B]");
  const auto batch = slots.numel();
  TORCH_CHECK(accepted.sizes() == slots.sizes() && accepted.is_contiguous() &&
                  accepted.scalar_type() == at::kInt,
              "accepted must be contiguous int32[B]");
  TORCH_CHECK(do_reset.sizes() == slots.sizes() && do_reset.is_contiguous() &&
                  do_reset.scalar_type() == at::kChar,
              "do_reset must be contiguous int8[B]");
  TORCH_CHECK(status.sizes() == slots.sizes() && status.is_contiguous() &&
                  status.scalar_type() == at::kInt,
              "status must be contiguous int32[B]");
  for (const auto& tensor : {cache_base, is_flush, accepted, slots, do_reset, status}) {
    TORCH_CHECK(tensor.device() == write_pos.device(), "All tensors must be on one NPU");
  }
  TORCH_CHECK(logical_limit > 0 && logical_limit <= 64 && max_spec_len > 0 &&
                  max_spec_len <= 8 && max_spec_len <= logical_limit &&
                  ring_len >= logical_limit && ring_len <= 128 &&
                  (ring_len & (ring_len - 1)) == 0,
              "Invalid logical_limit/max_spec_len/ring_len");
  TORCH_CHECK(batch <= std::numeric_limits<uint32_t>::max() &&
                  num_slots <= std::numeric_limits<uint32_t>::max(),
              "Tensor geometry exceeds uint32 range");
  const c10_npu::OptionalNPUGuard guard(write_pos.device());
  const auto stream = c10_npu::getCurrentNPUStream().stream();
  const auto result = aclrtlaunch_replayssm_sd_cursor(
      1, stream, write_pos.data_ptr(), cache_base.data_ptr(), is_flush.data_ptr(),
      accepted.data_ptr(), slots.data_ptr(), do_reset.data_ptr(), status.data_ptr(),
      static_cast<uint32_t>(batch), static_cast<uint32_t>(num_slots),
      static_cast<uint32_t>(logical_limit), static_cast<uint32_t>(max_spec_len),
      static_cast<uint32_t>(ring_len));
  TORCH_CHECK(result == ACL_SUCCESS, "Replay SD cursor launch failed: ", result);
}
}  // namespace vllm_ascend
#endif  // VLLM_ASCEND_ENABLE_REPLAYSSM_SD_CURSOR

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
#ifdef VLLM_ASCEND_ENABLE_REPLAYSSM_SD
#include "replayssm_sd_metadata_probe.h"
#include <algorithm>
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "aclrtlaunch_replayssm_sd_metadata_probe.h"

namespace vllm_ascend {
void replayssm_sd_metadata_probe(const at::Tensor& slots, const at::Tensor& ring_slots,
                                 const at::Tensor& positions, const at::Tensor& bases,
                                 const at::Tensor& lengths, const at::Tensor& packed_metadata,
                                 const at::Tensor& out, int64_t heads, int64_t cores, int64_t mode) {
  TORCH_CHECK(slots.device().type() == c10::DeviceType::PrivateUse1 && slots.dim() == 1 &&
                  slots.numel() > 0 && slots.numel() <= 1024 && slots.is_contiguous() &&
                  slots.scalar_type() == at::kInt, "slots must be contiguous NPU int32[B]");
  const int64_t batch = slots.numel();
  for (const auto& t : {ring_slots, positions, bases, lengths}) {
    TORCH_CHECK(t.device() == slots.device() && t.sizes() == slots.sizes() && t.is_contiguous() &&
                    t.scalar_type() == at::kInt, "metadata must be contiguous int32[B] on one NPU");
  }
  TORCH_CHECK(heads >= 1 && heads <= 32 && cores >= 1 && cores <= 40 && (mode == 0 || mode == 1),
              "invalid heads, cores, or mode");
  TORCH_CHECK(packed_metadata.device() == slots.device() && packed_metadata.scalar_type() == at::kInt &&
                  packed_metadata.is_contiguous() && packed_metadata.sizes() == at::IntArrayRef({batch, 8}),
              "packed_metadata must be contiguous int32[B,8] on the same NPU");
  TORCH_CHECK(out.device() == slots.device() && out.scalar_type() == at::kInt && out.is_contiguous() &&
                  out.sizes() == at::IntArrayRef({batch * heads, 8}), "out must be int32[B*heads,8]");
  const c10_npu::OptionalNPUGuard guard(slots.device());
  const auto stream = c10_npu::getCurrentNPUStream().stream();
  auto result = aclrtlaunch_replayssm_sd_metadata_probe(
      static_cast<uint32_t>(std::min(cores, batch * heads)), stream, slots.data_ptr(), ring_slots.data_ptr(),
      positions.data_ptr(), bases.data_ptr(), lengths.data_ptr(), packed_metadata.data_ptr(), out.data_ptr(),
      static_cast<uint32_t>(batch), static_cast<uint32_t>(heads), static_cast<uint32_t>(mode));
  TORCH_CHECK(result == ACL_SUCCESS, "Replay SD metadata probe launch failed: ", result);
}
}
#endif

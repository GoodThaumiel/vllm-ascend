// SPDX-License-Identifier: Apache-2.0
#include "ops.h"

#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "aclrtlaunch_replayssm_sd_310p_io_probe.h"

namespace replayssm_sd_310p {
void io_probe(const at::Tensor& input, const at::Tensor& output, int64_t mode) {
  TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 &&
                  output.device() == input.device(), "io_probe requires one NPU");
  TORCH_CHECK(input.dim() == 1 && input.numel() == 8 && output.sizes() == input.sizes() &&
                  input.scalar_type() == at::kInt && output.scalar_type() == at::kInt &&
                  input.is_contiguous() && output.is_contiguous(),
              "io_probe requires contiguous int32[8] tensors");
  TORCH_CHECK(mode >= 0 && mode <= 4, "io_probe mode must be in [0,4]");
  const c10_npu::OptionalNPUGuard guard(input.device());
  const auto stream = c10_npu::getCurrentNPUStream().stream();
  const aclError result = aclrtlaunch_replayssm_sd_310p_io_probe(
      1, stream, input.data_ptr(), output.data_ptr(), static_cast<uint32_t>(mode));
  TORCH_CHECK(result == ACL_SUCCESS, "io_probe launch failed: ", result);
}
}  // namespace replayssm_sd_310p

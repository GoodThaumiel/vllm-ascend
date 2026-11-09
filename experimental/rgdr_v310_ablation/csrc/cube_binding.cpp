// SPDX-License-Identifier: Apache-2.0
#include <acl/acl.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>

#include "aclrtlaunch_rgdr_v310_cube_matvec.h"

namespace {
void cube_matvec(const at::Tensor& a, const at::Tensor& b,
                 const at::Tensor& out, int64_t cores) {
  TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1,
              "NPU tensors required");
  TORCH_CHECK(a.device() == b.device() && a.device() == out.device() &&
              a.is_contiguous() && b.is_contiguous() && out.is_contiguous(),
              "tensors must be contiguous on one NPU");
  TORCH_CHECK(a.dim() == 3 && a.size(1) == 16 && a.size(2) == 128 &&
              b.sizes() == at::IntArrayRef({a.size(0), 128, 16}) &&
              out.sizes() == at::IntArrayRef({a.size(0), 16, 16}),
              "expected A=[tiles,16,128], B=[tiles,128,16], out=[tiles,16,16]");
  TORCH_CHECK(a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf &&
              out.scalar_type() == at::kFloat, "A/B must be FP16, out FP32");
  TORCH_CHECK(cores >= 1 && cores <= 8 && a.size(0) > 0,
              "cores must be 1..8 and tiles positive");
  const c10_npu::OptionalNPUGuard guard(a.device());
  auto stream = c10_npu::getCurrentNPUStream().stream();
  aclError status = aclrtlaunch_rgdr_v310_cube_matvec(
      static_cast<uint32_t>(std::min<int64_t>(cores, a.size(0))), stream,
      a.data_ptr(), b.data_ptr(), out.data_ptr(),
      static_cast<uint32_t>(a.size(0)));
  TORCH_CHECK(status == ACL_SUCCESS, "cube matvec launch failed: ", status);
}
}  // namespace

TORCH_LIBRARY(rgdr_v310_cube_probe, m) {
  m.def("matvec_(Tensor a, Tensor b, Tensor(c!) out, int cores=8) -> ()");
}
TORCH_LIBRARY_IMPL(rgdr_v310_cube_probe, PrivateUse1, m) {
  m.impl("matvec_", &cube_matvec);
}
TORCH_LIBRARY_IMPL(rgdr_v310_cube_probe, Meta, m) {
  m.impl("matvec_", [](const at::Tensor&, const at::Tensor&,
                       const at::Tensor&, int64_t) {});
}

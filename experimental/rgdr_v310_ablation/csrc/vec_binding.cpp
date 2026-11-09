// SPDX-License-Identifier: Apache-2.0
#include <acl/acl.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <cmath>

#include "aclrtlaunch_rgdr_v310_vec0.h"
#include "aclrtlaunch_rgdr_v310_vec1.h"
#include "aclrtlaunch_rgdr_v310_vec2.h"
#include "aclrtlaunch_rgdr_v310_vec3.h"
#include "aclrtlaunch_rgdr_v310_vector_matvec.h"

namespace {
void run_vec(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
             const at::Tensor& beta, const at::Tensor& state,
             const at::Tensor& lengths, const at::Tensor& state_indices,
             const at::Tensor& g, const at::Tensor& accepted,
             const at::Tensor& out, double scale, int64_t variant,
             int64_t cores) {
  TORCH_CHECK(variant >= 0 && variant <= 3, "variant must be 0..3");
  TORCH_CHECK(cores >= 1 && cores <= 7, "310P vector cores must be 1..7");
  TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1,
              "NPU tensors required");
  for (const auto& tensor : {q, k, v, beta, state, lengths, state_indices,
                             g, accepted, out}) {
    TORCH_CHECK(tensor.device() == q.device() && tensor.is_contiguous(),
                "all inputs must be contiguous on the same NPU");
  }
  TORCH_CHECK(q.dim() == 3 && q.size(1) == 8 && q.size(2) == 128 &&
              k.sizes() == q.sizes(), "q/k must be [T,8,128]");
  const int64_t tokens = q.size(0);
  TORCH_CHECK(v.sizes() == at::IntArrayRef({tokens, 16, 128}) &&
              out.sizes() == v.sizes(), "v/out must be [T,16,128]");
  TORCH_CHECK(beta.sizes() == at::IntArrayRef({tokens, 16}) &&
              g.sizes() == beta.sizes(), "beta/g must be [T,16]");
  const int64_t batch = lengths.numel();
  TORCH_CHECK(batch > 0 && batch <= 1024 && tokens <= batch * 8 &&
              state.dim() == 4 && state.size(0) >= tokens + 1 &&
              state.size(1) == 16 && state.size(2) == 128 &&
              state.size(3) == 128, "invalid batch or state geometry");
  TORCH_CHECK(lengths.sizes() == accepted.sizes() &&
              state_indices.numel() == tokens &&
              lengths.scalar_type() == at::kInt &&
              state_indices.scalar_type() == at::kInt &&
              accepted.scalar_type() == at::kInt, "metadata must be int32");
  for (const auto& tensor : {q, k, v, beta, state, out}) {
    TORCH_CHECK(tensor.scalar_type() == at::kHalf, "data must be FP16");
  }
  TORCH_CHECK(g.scalar_type() == at::kFloat, "g must be FP32");
  TORCH_CHECK(std::isfinite(scale) && scale > 0, "scale must be positive and finite");
  const c10_npu::OptionalNPUGuard guard(q.device());
  auto stream = c10_npu::getCurrentNPUStream().stream();
#define RGDR_LAUNCH_ARGS \
  static_cast<uint32_t>(cores), stream, q.data_ptr(), k.data_ptr(), \
      v.data_ptr(), beta.data_ptr(), state.data_ptr(), lengths.data_ptr(), \
      state_indices.data_ptr(), g.data_ptr(), accepted.data_ptr(), \
      out.data_ptr(), static_cast<uint32_t>(batch), \
      static_cast<uint32_t>(tokens), static_cast<uint32_t>(state.size(0)), \
      static_cast<float>(scale)
  aclError result = ACL_SUCCESS;
  switch (variant) {
    case 0: result = aclrtlaunch_rgdr_v310_vec0(RGDR_LAUNCH_ARGS); break;
    case 1: result = aclrtlaunch_rgdr_v310_vec1(RGDR_LAUNCH_ARGS); break;
    case 2: result = aclrtlaunch_rgdr_v310_vec2(RGDR_LAUNCH_ARGS); break;
    case 3: result = aclrtlaunch_rgdr_v310_vec3(RGDR_LAUNCH_ARGS); break;
  }
#undef RGDR_LAUNCH_ARGS
  TORCH_CHECK(result == ACL_SUCCESS, "rgdr_v310_vec launch failed: ", result);
}

void vector_matvec(const at::Tensor& a, const at::Tensor& vec,
                   const at::Tensor& out, int64_t cores) {
  TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1 &&
              a.device() == vec.device() && a.device() == out.device() &&
              a.is_contiguous() && vec.is_contiguous() && out.is_contiguous(),
              "tensors must be contiguous on one NPU");
  TORCH_CHECK(a.dim() == 3 && a.size(1) == 16 && a.size(2) == 128 &&
              vec.sizes() == at::IntArrayRef({a.size(0), 128}) &&
              out.sizes() == at::IntArrayRef({a.size(0), 16}),
              "expected A=[tiles,16,128], vec=[tiles,128], out=[tiles,16]");
  TORCH_CHECK(a.scalar_type() == at::kHalf && vec.scalar_type() == at::kHalf &&
              out.scalar_type() == at::kFloat, "A/vec must be FP16, out FP32");
  TORCH_CHECK(cores >= 1 && cores <= 7 && a.size(0) > 0,
              "cores must be 1..7 and tiles positive");
  const c10_npu::OptionalNPUGuard guard(a.device());
  auto stream = c10_npu::getCurrentNPUStream().stream();
  aclError status = aclrtlaunch_rgdr_v310_vector_matvec(
      static_cast<uint32_t>(std::min<int64_t>(cores, a.size(0))), stream,
      a.data_ptr(), vec.data_ptr(), out.data_ptr(),
      static_cast<uint32_t>(a.size(0)));
  TORCH_CHECK(status == ACL_SUCCESS, "vector matvec launch failed: ", status);
}
}  // namespace

TORCH_LIBRARY(rgdr_v310_ablation, m) {
  m.def("run_(Tensor q, Tensor k, Tensor v, Tensor beta, Tensor(a!) state, "
        "Tensor lengths, Tensor state_indices, Tensor g, Tensor accepted, "
        "Tensor(b!) out, float scale, int variant, int cores=7) -> ()");
  m.def("matvec_(Tensor a, Tensor vec, Tensor(c!) out, int cores=7) -> ()");
}
TORCH_LIBRARY_IMPL(rgdr_v310_ablation, PrivateUse1, m) {
  m.impl("run_", &run_vec);
  m.impl("matvec_", &vector_matvec);
}
TORCH_LIBRARY_IMPL(rgdr_v310_ablation, Meta, m) {
  m.impl("run_", [](const at::Tensor&, const at::Tensor&, const at::Tensor&,
                     const at::Tensor&, const at::Tensor&, const at::Tensor&,
                     const at::Tensor&, const at::Tensor&, const at::Tensor&,
                     const at::Tensor&, double, int64_t, int64_t) {});
  m.impl("matvec_", [](const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, int64_t) {});
}

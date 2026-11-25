// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <acl/acl.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <algorithm>
#include <cstdint>
#include "aclrtlaunch_replayssm_sd_verify_mix_910b.h"

namespace {
void Verify(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
            const at::Tensor& g, const at::Tensor& beta, const at::Tensor& checkpoint,
            const at::Tensor& d, const at::Tensor& keys, const at::Tensor& gates,
            const at::Tensor& slots, const at::Tensor& ringSlots, const at::Tensor& pos,
            const at::Tensor& base, const at::Tensor& length, const at::Tensor& out,
            const at::Tensor& status, const at::Tensor& zeros, const at::Tensor& scratch,
            int64_t cores) {
  TORCH_CHECK(q.device().is_privateuseone() && q.dim() == 4, "q must be a rank-4 NPU tensor");
  const auto b = q.size(0), hk = q.size(2);
  TORCH_CHECK(v.dim() == 4, "v must be rank-4");
  const auto hv = v.size(2);
  TORCH_CHECK(b >= 1 && b <= 128 && ((hk == 4 && hv == 12) || (hk == 16 && hv == 32)),
              "expected B1..128 and 27B TP4 or 9B TP1");
  TORCH_CHECK(cores >= 1 && cores <= 20, "cores must be 1..20 on 910B4");
  TORCH_CHECK(checkpoint.dim() == 4 && d.dim() == 4, "checkpoint/D must be rank-4");
  const auto numSlots = checkpoint.size(0), numRingSlots = d.size(0);
  TORCH_CHECK(numSlots > 1 && numSlots <= 1024 && numRingSlots > 1 && numRingSlots <= 1024,
              "slot counts must be 2..1024");
  const auto blocks = std::min(cores, b * hk);
  auto expect = [&](const at::Tensor& t, at::IntArrayRef shape, at::ScalarType dtype, const char* name) {
    TORCH_CHECK(t.device() == q.device() && t.is_contiguous() && t.sizes() == shape && t.scalar_type() == dtype,
                name, " has wrong device, contiguity, shape, or dtype");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 32 == 0, name, " must start at a 32 B boundary");
  };
  expect(q, {b, 4, hk, 128}, at::kBFloat16, "q");
  expect(k, q.sizes(), at::kBFloat16, "k");
  expect(v, {b, 4, hv, 128}, at::kBFloat16, "v");
  expect(out, v.sizes(), at::kBFloat16, "out");
  expect(g, {b, 4, hv}, at::kFloat, "g");
  expect(beta, g.sizes(), at::kFloat, "beta");
  expect(checkpoint, {numSlots, hv, 128, 128}, at::kBFloat16, "checkpoint");
  expect(d, {numRingSlots, hv, 32, 128}, at::kHalf, "d");
  expect(keys, {numRingSlots, hk, 32, 128}, at::kHalf, "keys");
  expect(gates, {numRingSlots, hv, 32}, at::kFloat, "gates");
  for (auto& t : {slots, ringSlots, pos, base, length}) {
    expect(t, {b}, at::kInt, "metadata");
  }
  expect(status, {b * hv}, at::kInt, "status");
  expect(zeros, {16, 128}, at::kBFloat16, "zero RHS");
  TORCH_CHECK(scratch.dim() == 3, "scratch must be rank-3");
  const bool perHeadWorkspace = scratch.size(0) == b * hv;
  expect(scratch, {perHeadWorkspace ? b * hv : blocks, 128, 16}, at::kFloat, "scratch");
  const c10_npu::OptionalNPUGuard guard(q.device());
  const auto stream = c10_npu::getCurrentNPUStream().stream();
  const auto rc = aclrtlaunch_replayssm_sd_verify_mix_910b(
      static_cast<uint32_t>(blocks), stream, q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(),
      beta.data_ptr(), checkpoint.data_ptr(), d.data_ptr(), keys.data_ptr(), gates.data_ptr(), slots.data_ptr(),
      ringSlots.data_ptr(), pos.data_ptr(), base.data_ptr(), length.data_ptr(), out.data_ptr(), status.data_ptr(),
      zeros.data_ptr(), scratch.data_ptr(), static_cast<uint32_t>(b), static_cast<uint32_t>(hk),
      static_cast<uint32_t>(hv), static_cast<uint32_t>(numSlots), static_cast<uint32_t>(numRingSlots),
      static_cast<uint32_t>(blocks), static_cast<uint32_t>(perHeadWorkspace));
  TORCH_CHECK(rc == ACL_SUCCESS, "910B mixed verify launch failed: ", rc);
}
}  // namespace

TORCH_LIBRARY(replayssm_mix_910b, m) {
  m.def("verify_(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor checkpoint, "
        "Tensor(a!) d, Tensor(b!) keys, Tensor(c!) gates, Tensor slots, Tensor ring_slots, "
        "Tensor positions, Tensor bases, Tensor lengths, Tensor(d!) out, Tensor(e!) status, "
        "Tensor zeros, Tensor(f!) scratch, int cores) -> ()");
}
TORCH_LIBRARY_IMPL(replayssm_mix_910b, PrivateUse1, m) {
  m.impl("verify_", &Verify);
}

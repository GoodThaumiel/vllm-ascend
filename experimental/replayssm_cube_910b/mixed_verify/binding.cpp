// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <acl/acl.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <algorithm>
#include <cstdint>
#include <optional>
#include "aclrtlaunch_replayssm_sd_verify_mix_910b.h"
#ifdef REPLAYSSM_ROOFLINE_CALIBRATION
#include "aclrtlaunch_replayssm_roofline_vector.h"
#endif

namespace {
constexpr int64_t kMaxTokens = REPLAYSSM_VERIFY_MAX_TOKENS;

#ifdef REPLAYSSM_ROOFLINE_CALIBRATION
void MeasureVector(const at::Tensor& out, int64_t repeats) {
  TORCH_CHECK(out.device().is_privateuseone() && out.is_contiguous() &&
                  out.scalar_type() == at::kFloat && out.sizes() == at::IntArrayRef({40, 4, 4096}),
              "Vector calibration requires contiguous FP32 output [40, 4, 4096]");
  TORCH_CHECK(repeats >= 1 && repeats <= 512, "Vector calibration repeats must be 1..512");
  const c10_npu::OptionalNPUGuard guard(out.device());
  auto stream = c10_npu::getCurrentNPUStream().stream();
  const auto rc = aclrtlaunch_replayssm_roofline_vector(40, stream, out.data_ptr(), static_cast<uint32_t>(repeats));
  TORCH_CHECK(rc == ACL_SUCCESS, "Vector calibration launch failed: ", rc);
}
#endif

void Verify(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
            const at::Tensor& g, const at::Tensor& beta, const at::Tensor& checkpoint,
            const at::Tensor& d, const at::Tensor& keys, const at::Tensor& gates,
            const at::Tensor& slots, const at::Tensor& ringSlots, const at::Tensor& pos,
            const at::Tensor& base, const at::Tensor& length, const at::Tensor& out,
            const at::Tensor& status, const at::Tensor& zeros, const at::Tensor& scratch,
            int64_t cores, bool packed_panel, bool suffix_history, bool bulk_qkv, bool balance_heads, int64_t aiv_ablation,
            std::optional<at::Tensor> flush, std::optional<at::Tensor> tokenOffsets, bool expand_head_grid) {
  const bool packedTokens = tokenOffsets.has_value();
  const auto rank = packedTokens ? 3 : 4;
  TORCH_CHECK(q.device().is_privateuseone() && q.dim() == rank && v.dim() == rank,
              "q/v must have rank ", rank, " for this token layout");
  TORCH_CHECK(slots.dim() == 1, "slots must be rank-1");
  const auto b = slots.size(0), hk = q.size(rank - 2), hv = v.size(rank - 2);
  const auto tokenCount = packedTokens ? q.size(0) : 0;
  TORCH_CHECK(!packedTokens || (tokenCount > 0 && tokenCount <= b * kMaxTokens),
              "packed token count must be in 1..B*", kMaxTokens);
  TORCH_CHECK(kMaxTokens == 4 || (packedTokens && (aiv_ablation == 4 || aiv_ablation == 7)),
              "The eight-token build requires packed tokens and fused verify/compact");
  TORCH_CHECK(b >= 1 && b <= 256 && ((hk == 4 && hv == 12) || (hk == 16 && hv == 32)),
              "expected B1..256 and 27B TP4 or 9B TP1");
  TORCH_CHECK(cores >= 1 && cores <= 20, "cores must be 1..20 on 910B4");
  TORCH_CHECK(!expand_head_grid || balance_heads, "An expanded grid requires value-head balancing");
  TORCH_CHECK(packed_panel || (!suffix_history && !bulk_qkv), "AIV ablations require the packed panel");
  TORCH_CHECK(!suffix_history || !bulk_qkv, "Run suffix history and bulk QKV as separate ablations");
  TORCH_CHECK(aiv_ablation >= 0 && aiv_ablation <= 7, "Invalid AIV ablation");
  TORCH_CHECK(aiv_ablation == 0 || (packed_panel && bulk_qkv && !suffix_history),
              "AIV ablations require the bulk-QKV packed baseline");
  TORCH_CHECK(checkpoint.dim() == 4 && d.dim() == 4, "checkpoint/D must be rank-4");
  const auto numSlots = checkpoint.size(0), numRingSlots = d.size(0);
  TORCH_CHECK(numSlots > 1 && numSlots <= UINT32_MAX && numRingSlots > 1 && numRingSlots <= 1024,
              "checkpoint slots must fit uint32; ring slots must be 2..1024");
  const auto blocks = std::min(cores, b * (expand_head_grid ? hv : hk));
  auto expect = [&](const at::Tensor& t, at::IntArrayRef shape, at::ScalarType dtype, const char* name) {
    TORCH_CHECK(t.device() == q.device() && t.is_contiguous() && t.sizes() == shape && t.scalar_type() == dtype,
                name, " has wrong device, contiguity, shape, or dtype");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 32 == 0, name, " must start at a 32 B boundary");
  };
  if (packedTokens) {
    expect(q, {tokenCount, hk, 128}, at::kBFloat16, "q");
    expect(v, {tokenCount, hv, 128}, at::kBFloat16, "v");
    expect(g, {tokenCount, hv}, at::kFloat, "g");
    expect(*tokenOffsets, {b + 1}, at::kInt, "token offsets");
  } else {
    expect(q, {b, kMaxTokens, hk, 128}, at::kBFloat16, "q");
    expect(v, {b, kMaxTokens, hv, 128}, at::kBFloat16, "v");
    expect(g, {b, kMaxTokens, hv}, at::kFloat, "g");
  }
  expect(k, q.sizes(), at::kBFloat16, "k");
  expect(out, v.sizes(), at::kBFloat16, "out");
  expect(beta, g.sizes(), at::kFloat, "beta");
  const auto checkpointStride = checkpoint.stride(0);
  TORCH_CHECK(checkpoint.device() == q.device() && checkpoint.scalar_type() == at::kBFloat16 &&
              checkpoint.sizes() == at::IntArrayRef({numSlots, hv, 128, 128}) &&
              checkpoint.stride(1) == 128 * 128 && checkpoint.stride(2) == 128 &&
              checkpoint.stride(3) == 1 && checkpointStride >= hv * 128 * 128 &&
              checkpointStride % 16 == 0 && reinterpret_cast<uintptr_t>(checkpoint.data_ptr()) % 32 == 0,
              "checkpoint must have aligned BF16 pages with contiguous [H, 128, 128] entries");
  expect(d, {numRingSlots, hv, 32, 128}, at::kHalf, "d");
  expect(keys, {numRingSlots, hk, 32, 128}, at::kHalf, "keys");
  expect(gates, {numRingSlots, hv, 32}, at::kFloat, "gates");
  for (auto& t : {slots, ringSlots, pos, base, length}) {
    expect(t, {b}, at::kInt, "metadata");
  }
  TORCH_CHECK((aiv_ablation >= 4) == flush.has_value(), "Fused Compact requires a flush tensor");
  if (flush.has_value()) {
    expect(*flush, {b}, at::kInt, "flush");
  }
  expect(status, {b * hv}, at::kInt, "status");
  expect(zeros, {16, 128}, at::kBFloat16, "zero RHS");
  TORCH_CHECK(scratch.dim() == 3, "scratch must be rank-3");
  const bool perHeadWorkspace = scratch.size(0) == b * hv;
  expect(scratch, {perHeadWorkspace ? b * hv : blocks, 128, 16}, at::kFloat, "scratch");
  const c10_npu::OptionalNPUGuard guard(q.device());
  const auto stream = c10_npu::getCurrentNPUStream().stream();
  const uint32_t panel_mode =
      aiv_ablation > 0
      ? static_cast<uint32_t>(3 + aiv_ablation)
      : static_cast<uint32_t>(packed_panel) + static_cast<uint32_t>(suffix_history) + 2 * static_cast<uint32_t>(bulk_qkv);
  const auto rc = aclrtlaunch_replayssm_sd_verify_mix_910b(
      static_cast<uint32_t>(blocks), stream, q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(),
      beta.data_ptr(), checkpoint.data_ptr(), d.data_ptr(), keys.data_ptr(), gates.data_ptr(), slots.data_ptr(),
      ringSlots.data_ptr(), pos.data_ptr(), base.data_ptr(), length.data_ptr(), out.data_ptr(), status.data_ptr(),
      zeros.data_ptr(), scratch.data_ptr(), flush.has_value() ? flush->data_ptr() : slots.data_ptr(),
      packedTokens ? tokenOffsets->data_ptr() : length.data_ptr(), static_cast<uint32_t>(tokenCount),
      static_cast<uint64_t>(checkpointStride), static_cast<uint32_t>(b), static_cast<uint32_t>(hk),
      static_cast<uint32_t>(hv), static_cast<uint32_t>(numSlots), static_cast<uint32_t>(numRingSlots),
      static_cast<uint32_t>(blocks), static_cast<uint32_t>(perHeadWorkspace), panel_mode, static_cast<uint32_t>(balance_heads));
  TORCH_CHECK(rc == ACL_SUCCESS, "910B mixed verify launch failed: ", rc);
}
}  // namespace

TORCH_LIBRARY(replayssm_mix_910b, m) {
  m.def("max_tokens() -> int", []() -> int64_t { return kMaxTokens; });
  m.def("verify_(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor(g!) checkpoint, "
        "Tensor(a!) d, Tensor(b!) keys, Tensor(c!) gates, Tensor slots, Tensor ring_slots, "
        "Tensor positions, Tensor bases, Tensor lengths, Tensor(d!) out, Tensor(e!) status, "
        "Tensor zeros, Tensor(f!) scratch, int cores, bool packed_panel=False, "
        "bool suffix_history=False, bool bulk_qkv=False, bool balance_heads=False, int aiv_ablation=0, "
        "Tensor? flush=None, Tensor? token_offsets=None, bool expand_head_grid=False) -> ()");
#ifdef REPLAYSSM_ROOFLINE_CALIBRATION
  m.def("roofline_vector_(Tensor(a!) out, int repeats) -> ()");
#endif
}
TORCH_LIBRARY_IMPL(replayssm_mix_910b, PrivateUse1, m) {
  m.impl("verify_", &Verify);
#ifdef REPLAYSSM_ROOFLINE_CALIBRATION
  m.impl("roofline_vector_", &MeasureVector);
#endif
}

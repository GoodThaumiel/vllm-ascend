// SPDX-License-Identifier: Apache-2.0
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <acl/acl.h>
#include <tiling/platform/platform_ascendc.h>
#include <tiling/tiling_api.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <array>
#include <cstring>
#include <limits>
#include <tuple>
#include <vector>

#include "aclrtlaunch_replayssm_310p_compute_wy.h"
#include "kernels/wy_launch_tiling.h"

namespace {
using Outputs = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

void fill_cube_tiling(platform_ascendc::PlatformAscendC& platform, optiling::TCubeTiling& out,
                      bool transpose_b, bool from_ub) {
  using namespace matmul_tiling;
  MatmulApiTiling tiler(platform);
  const auto pos = from_ub ? TPosition::VECCALC : TPosition::GM;
  tiler.SetAType(pos, CubeFormat::ND, DataType::DT_FLOAT16, false);
  tiler.SetBType(pos, CubeFormat::ND, DataType::DT_FLOAT16, transpose_b);
  tiler.SetCType(TPosition::VECCALC, CubeFormat::ND, DataType::DT_FLOAT);
  tiler.SetBias(false);
  tiler.SetOrgShape(WY_CHUNK_SIZE, WY_CHUNK_SIZE, WY_CHUNK_SIZE);
  tiler.SetShape(WY_CHUNK_SIZE, WY_CHUNK_SIZE, WY_CHUNK_SIZE);
  tiler.SetBufferSpace(-1, -1, -1);
  tiler.SetFixSplit(WY_CHUNK_SIZE, WY_CHUNK_SIZE, WY_CHUNK_SIZE);
  TORCH_CHECK(tiler.GetTiling(out) != -1, "compute WY Cube tiling failed");
}

struct PlatformTiling {
  uint32_t cores;
  uint32_t system_workspace_bytes;
  std::vector<uint8_t> cube_tiling;
};

PlatformTiling get_platform_tiling() {
  auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  TORCH_CHECK(platform != nullptr, "compute WY could not query the Ascend platform");
  PlatformTiling result{platform->GetCoreNumAic(), platform->GetLibApiWorkSpaceSize(), {}};
  TORCH_CHECK(result.cores > 0 && result.system_workspace_bytes != std::numeric_limits<uint32_t>::max(),
              "compute WY could not query Cube cores or system workspace");

  // Order matches the four TCubeTiling fields in the device ABI.
  std::array<optiling::TCubeTiling, 4> cube;
  fill_cube_tiling(*platform, cube[0], true, false);
  fill_cube_tiling(*platform, cube[1], false, false);
  fill_cube_tiling(*platform, cube[2], false, true);
  fill_cube_tiling(*platform, cube[3], false, true);
  for (auto& field : cube) {
    const auto begin = result.cube_tiling.size();
    const auto size = field.GetDataSize();
    result.cube_tiling.resize(begin + size);
    field.SaveToBuffer(result.cube_tiling.data() + begin, size);
  }
  return result;
}

Outputs make_outputs(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                     const at::Tensor& g, const at::Tensor& beta, int64_t chunk_size) {
  TORCH_CHECK(q.dim() == 4 && q.sizes() == k.sizes(), "q and k must share [B,T,Hk,Dk]");
  TORCH_CHECK(v.dim() == 4 && v.size(0) == q.size(0) && v.size(1) == q.size(1),
              "v must be [B,T,Hv,Dv] with the same B and T as q");
  const auto b = q.size(0);
  const auto t = q.size(1);
  const auto hk = q.size(2);
  const auto hv = v.size(2);
  const auto dk = q.size(3);
  const auto dv = v.size(3);
  TORCH_CHECK(g.sizes() == at::IntArrayRef({b, t, hv}) && beta.sizes() == g.sizes(),
              "g and beta must be [B,T,Hv]");
  TORCH_CHECK(b > 0 && b <= 32 && hk > 0 && hv > 0 && hv <= 64 && hv % hk == 0,
              "compute WY requires 1<=B<=32, 1<=Hv<=64 and Hv divisible by Hk");
  TORCH_CHECK(chunk_size == WY_CHUNK_SIZE && t > 0 && t % chunk_size == 0,
              "compute WY requires T padded to a multiple of 64");
  TORCH_CHECK((dk == 64 || dk == 128) && (dv == 64 || dv == 128),
              "compute WY head dimensions must be 64 or 128");
  for (const auto& tensor : {q, k, v, g, beta}) {
    TORCH_CHECK(tensor.device() == q.device() && tensor.is_contiguous(),
                "compute WY expects contiguous inputs on the same device");
  }
  TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf &&
                  v.scalar_type() == at::kHalf && beta.scalar_type() == at::kHalf &&
                  g.scalar_type() == at::kFloat,
              "compute WY requires float16 q/k/v/beta and float32 g");
  return {at::empty({b, hk, t, dk}, q.options()), at::empty({b, hk, t, dk}, k.options()),
          at::empty({b, hv, t, dk}, k.options()), at::empty({b, hv, t, dv}, v.options()),
          at::empty({b, hv, t}, g.options())};
}

Outputs prepare(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                const at::Tensor& g, const at::Tensor& beta, int64_t chunk_size) {
  const c10_npu::OptionalNPUGuard guard(q.device());
  auto outputs = make_outputs(q, k, v, g, beta, chunk_size);
  auto& [q_out, k_out, w_out, u_out, g_out] = outputs;

  static const auto platform = get_platform_tiling();
  const auto num_chunks = q.size(1) / chunk_size;
  const WyLaunchTiling launch{
      q.size(0), q.size(1), q.size(2), v.size(2), q.size(3), v.size(3),
      chunk_size, num_chunks, v.size(2) / q.size(2), q.size(0) * v.size(2) * num_chunks,
      WY_LOCAL_WORKSPACE_BYTES, WY_PER_CORE_WORKSPACE_BYTES, platform.cores,
      0,  // HAVE_WORKSPACE passes the already-offset user workspace.
  };
  // The device struct is rounded up to an eight-byte boundary.
  const int64_t tiling_bytes = (sizeof(launch) + platform.cube_tiling.size() + 7) & ~size_t{7};
  auto host_tiling = at::zeros({tiling_bytes}, at::TensorOptions().device(at::kCPU).dtype(at::kByte).pinned_memory(true));
  auto* bytes = host_tiling.data_ptr<uint8_t>();
  std::memcpy(bytes, &launch, sizeof(launch));
  std::memcpy(bytes + sizeof(launch), platform.cube_tiling.data(), platform.cube_tiling.size());
  auto device_tiling = host_tiling.to(q.device(), /*non_blocking=*/true);
  const int64_t workspace_bytes = static_cast<int64_t>(platform.system_workspace_bytes) +
                                  static_cast<int64_t>(platform.cores) * WY_PER_CORE_WORKSPACE_BYTES;
  const auto workspace = at::empty({workspace_bytes}, q.options().dtype(at::kByte));
  const auto stream = c10_npu::getCurrentNPUStream().stream();
  const auto status = aclrtlaunch_replayssm_310p_compute_wy(
      platform.cores, stream, q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
      q_out.data_ptr(), k_out.data_ptr(), w_out.data_ptr(), u_out.data_ptr(), g_out.data_ptr(),
      workspace.data_ptr(), device_tiling.data_ptr());
  TORCH_CHECK(status == ACL_SUCCESS, "310P compute WY launch failed: ", status);
  return outputs;
}
}  // namespace

TORCH_LIBRARY(replayssm_310p_compute_wy, m) {
  m.def("prepare(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, int chunk_size) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(replayssm_310p_compute_wy, PrivateUse1, m) {
  m.impl("prepare", &prepare);
}
TORCH_LIBRARY_IMPL(replayssm_310p_compute_wy, Meta, m) {
  m.impl("prepare", &make_outputs);
}

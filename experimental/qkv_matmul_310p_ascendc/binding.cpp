// SPDX-License-Identifier: Apache-2.0
#include <acl/acl.h>
#include <tiling/platform/platform_ascendc.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/NPUStorageImpl.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <array>

#include "aclrtlaunch_qkv_matmul_310p_fp16.h"
#include "geometry.h"

namespace {
using namespace qkv310p;
constexpr int64_t MAX_ROWS = 4096;
constexpr uintptr_t DMA_ALIGNMENT = 32;

const torch_npu::NPUStorageDesc& descriptor(const at::Tensor& tensor) {
  const auto* storage = tensor.storage().unsafeGetStorageImpl();
  return static_cast<const torch_npu::NPUStorageImpl*>(storage)->npu_desc_;
}

void check_nz(const at::Tensor& tensor, int64_t rows, const char* name) {
  const int64_t row_blocks = (rows + C0 - 1) / C0;
  const int64_t physical_bytes = K * row_blocks * C0 * FP16_BYTES;
  const auto& desc = descriptor(tensor);
  const std::array<int64_t, 4> storage_shape{K / C0, row_blocks, C0, C0};
  const at::IntArrayRef expected_storage(storage_shape);
  const auto storage_sizes = at::IntArrayRef(desc.storage_sizes_);
  const bool native_nz = desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ &&
                         tensor.sizes() == at::IntArrayRef({rows, K}) &&
                         at::IntArrayRef(desc.base_sizes_) == tensor.sizes() &&
                         storage_sizes == expected_storage;
  // TorchNPU can tag a contiguous 4D upload as NCHW. Its bytes are in the
  // same row-major order as ND, and already packed as NZ by the caller.
  const bool packed_nz = (desc.npu_format_ == ACL_FORMAT_ND || desc.npu_format_ == ACL_FORMAT_NCHW) &&
                         tensor.sizes() == expected_storage;
  TORCH_CHECK(native_nz || packed_nz,
              name, " must be FRACTAL_NZ or contiguous packed storage ", expected_storage,
              ", got format=", static_cast<int>(desc.npu_format_), ", shape=", tensor.sizes(),
              ", storage=", storage_sizes);
  TORCH_CHECK(tensor.storage_offset() == 0 &&
                  tensor.storage().nbytes() >= static_cast<size_t>(physical_bytes),
              name, " must own the complete NZ storage at offset zero");
}

uint32_t get_cube_cores() {
  using platform_ascendc::CoreMemType;
  auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  TORCH_CHECK(platform != nullptr, "could not query the Ascend platform");
  const auto require = [platform](CoreMemType type, uint64_t needed, const char* name) {
    uint64_t available = 0;
    platform->GetCoreMemSize(type, available);
    TORCH_CHECK(available >= needed, name, " has ", available, " bytes, tile needs ", needed);
  };
  require(CoreMemType::L1, A_SLICE_BYTES + W_PANEL_BYTES, "L1");
  require(CoreMemType::L0_A, A_SLICE_BYTES, "L0A");
  require(CoreMemType::L0_B, W_SLICE_BYTES, "L0B");
  require(CoreMemType::L0_C, C_TILE_BYTES, "L0C");
  require(CoreMemType::UB, C_TILE_BYTES, "UB");
  const auto cores = platform->GetCoreNumAic();
  TORCH_CHECK(cores > 0, "could not query the Cube core count");
  return cores;
}

void run(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& out,
         int64_t cores) {
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
              "QKV matmul requires Ascend tensors");
  for (const auto& tensor : {x, weight, out}) {
    TORCH_CHECK(tensor.device() == x.device() && tensor.scalar_type() == at::kHalf &&
                    tensor.is_contiguous(),
                "QKV matmul requires contiguous FP16 tensors on a single NPU");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % DMA_ALIGNMENT == 0,
                "QKV matmul requires 32-byte aligned tensors");
  }
  TORCH_CHECK(out.dim() == 2 && out.size(1) == N && out.size(0) >= 1 &&
                  out.size(0) <= MAX_ROWS && descriptor(out).npu_format_ == ACL_FORMAT_ND,
              "out must be an ND FP16 [M,6144] tensor with 1 <= M <= ", MAX_ROWS);
  const auto rows = out.size(0);
  check_nz(x, rows, "activation");
  check_nz(weight, N, "weight");
  TORCH_CHECK(!out.is_alias_of(x) && !out.is_alias_of(weight),
              "out must not overlap either input");

  const c10_npu::OptionalNPUGuard guard(x.device());
  // This experiment is built for one SoC. Cache its per-core capacities.
  static const uint32_t cube_cores = get_cube_cores();
  TORCH_CHECK(cores >= 0 && cores <= cube_cores,
              "cores must be 0 (auto) or in [1,", cube_cores, "]");
  const auto blocks = std::min(cores == 0 ? cube_cores : static_cast<uint32_t>(cores),
                              N_TILES);
  const auto stream = c10_npu::getCurrentNPUStream().stream();
  const auto status = aclrtlaunch_qkv_matmul_310p_fp16(
      blocks, stream, x.data_ptr(), weight.data_ptr(), out.data_ptr(), static_cast<uint32_t>(rows));
  TORCH_CHECK(status == ACL_SUCCESS, "QKV matmul launch failed: ", status);
}
}  // namespace

TORCH_LIBRARY(qkv_matmul_310p, m) {
  m.def("run_(Tensor x, Tensor weight, Tensor(a!) out, int cores=0) -> ()");
}
TORCH_LIBRARY_IMPL(qkv_matmul_310p, PrivateUse1, m) {
  m.impl("run_", &run);
}

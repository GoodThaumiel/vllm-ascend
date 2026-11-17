// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <acl/acl.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <limits>

#include "../layout.h"
#include "aclrtlaunch_dflash_310p_prepare.h"

namespace dflash310 {
namespace {

void check_layout(const at::Tensor& requests, const at::Tensor& positions, const at::Tensor& tables,
                  const at::Tensor& group_info, const at::Tensor& query, const at::Tensor& context,
                  int64_t spec, int64_t max_model_len, int64_t mask_id, bool) {
  constexpr auto max_index = std::numeric_limits<int32_t>::max();
  for (const auto& tensor : {requests, positions, tables, group_info, query, context}) {
    TORCH_CHECK(tensor.device() == requests.device() && tensor.scalar_type() == at::kInt &&
                    tensor.is_contiguous() && tensor.numel() > 0 && tensor.numel() <= max_index,
                "DFlash buffers must be nonempty contiguous int32 tensors on one device");
  }
  TORCH_CHECK(spec >= 1 && spec <= kMaxDraftTokens && max_model_len >= 1 &&
                  max_model_len <= max_index - kDmaLanes && mask_id >= 0 && mask_id <= max_index,
              "Invalid K, maximum model length or DFlash mask token");
  TORCH_CHECK(requests.dim() == 2 && requests.size(1) == kDmaLanes, "Expected requests [B,8]");
  TORCH_CHECK(positions.dim() == 1 && positions.numel() % kDmaLanes == 0, "Expected 32B-padded positions");
  const auto batch = requests.size(0);
  TORCH_CHECK(tables.dim() == 3 && tables.size(0) <= kMaxGroups && tables.size(1) == batch,
              "Expected block tables [groups,B,table_width] for 1 or 2 groups");
  const auto groups = tables.size(0);
  TORCH_CHECK(group_info.sizes() == at::IntArrayRef({groups, kDmaLanes}), "Expected group info [groups,8]");
  TORCH_CHECK(query.sizes() == at::IntArrayRef({groups, batch, QueryFields, kDmaLanes}), "Invalid query buffer shape");
  TORCH_CHECK(context.sizes() == at::IntArrayRef({groups, ContextFields, positions.numel()}),
              "Invalid context buffer shape");
}

void prepare(const at::Tensor& requests, const at::Tensor& positions, const at::Tensor& tables,
             const at::Tensor& group_info, const at::Tensor& query, const at::Tensor& context,
             int64_t spec, int64_t max_model_len, int64_t mask_id, bool from_anchor) {
  check_layout(requests, positions, tables, group_info, query, context, spec, max_model_len, mask_id, from_anchor);
  TORCH_CHECK(requests.device().type() == c10::DeviceType::PrivateUse1, "NPU tensors required");
  for (const auto& output : {query, context}) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(output.data_ptr()) % 32 == 0, "DFlash outputs must be 32B aligned");
  }
  for (const auto& input : {requests, positions, tables, group_info}) {
    TORCH_CHECK(!query.is_alias_of(input) && !context.is_alias_of(input), "DFlash outputs must own separate storage");
  }
  TORCH_CHECK(!query.is_alias_of(context), "DFlash outputs must own separate storage");
  const c10_npu::OptionalNPUGuard guard(requests.device());
  const auto stream = c10_npu::getCurrentNPUStream().stream();
  const auto groups = static_cast<uint32_t>(tables.size(0));
  const auto result = aclrtlaunch_dflash_310p_prepare(
      groups, stream, requests.data_ptr(), positions.data_ptr(), tables.data_ptr(), group_info.data_ptr(),
      query.data_ptr(), context.data_ptr(), static_cast<uint32_t>(requests.size(0)),
      static_cast<uint32_t>(positions.numel()), static_cast<uint32_t>(tables.size(2)), static_cast<uint32_t>(spec),
      static_cast<uint32_t>(max_model_len), static_cast<int32_t>(mask_id), static_cast<uint32_t>(from_anchor));
  TORCH_CHECK(result == ACL_SUCCESS, "310P DFlash prepare launch failed: ", result);
}

}  // namespace
}  // namespace dflash310

TORCH_LIBRARY(dflash_310p, m) {
  m.def("prepare_(Tensor requests, Tensor positions, Tensor block_tables, Tensor group_info, "
        "Tensor(a!) query, Tensor(b!) context, int spec, int max_model_len, int mask_id, bool from_anchor) -> ()");
}

TORCH_LIBRARY_IMPL(dflash_310p, PrivateUse1, m) {
  m.impl("prepare_", &dflash310::prepare);
}

TORCH_LIBRARY_IMPL(dflash_310p, Meta, m) {
  m.impl("prepare_", &dflash310::check_layout);
}

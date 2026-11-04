// SPDX-License-Identifier: Apache-2.0
#include <torch/library.h>
#include <ATen/ATen.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <acl/acl.h>
#include <algorithm>
#include "aclrtlaunch_replayssm_probe_u2_history_tree.h"
#include "aclrtlaunch_replayssm_probe_baseline.h"
#include "aclrtlaunch_replayssm_probe_a1_control.h"
#include "aclrtlaunch_replayssm_probe_a1_replace.h"
#include "aclrtlaunch_replayssm_probe_a2_control.h"
#include "aclrtlaunch_replayssm_probe_a2_remove.h"
#include "aclrtlaunch_replayssm_probe_local.h"
#include "aclrtlaunch_replayssm_probe_u1_chunk4k.h"
#include "aclrtlaunch_replayssm_probe_u1_chunk8k.h"

void run_probe(const c10::List<at::Tensor>& tensors, const at::Tensor& sink,
               int64_t mode, int64_t cores, int64_t repeats, bool profiling_only) {
  TORCH_CHECK(profiling_only, "experimental operator requires profiling_only=true");
  TORCH_CHECK(tensors.size() == 16 && mode >= 0 && mode <= 8 && cores >= 1 && cores <= 40 &&
              repeats >= 1 && repeats <= 64, "invalid arguments");
  std::vector<at::Tensor> t;
  for (const auto& x : tensors) t.push_back(x);
  auto dev = t[0].device();
  TORCH_CHECK(dev.type() == c10::DeviceType::PrivateUse1, "NPU only");
  const auto b = t[0].size(0);
  TORCH_CHECK(b >= 1 && b <= 64, "batch outside 1..64");
  for (const auto& x : t) TORCH_CHECK(x.device() == dev && x.is_contiguous(), "contiguous same-device input required");
  auto shape = [&](int i, std::initializer_list<int64_t> dims, at::ScalarType dtype) {
    TORCH_CHECK(t[i].sizes() == at::IntArrayRef(dims) && t[i].scalar_type() == dtype, "tensor shape/dtype mismatch at ", i);
  };
  shape(0,{b,4,4,128},at::kBFloat16); shape(1,{b,4,4,128},at::kBFloat16);
  shape(2,{b,4,12,128},at::kBFloat16); shape(3,{b,4,12},at::kFloat); shape(4,{b,4,12},at::kFloat);
  shape(5,{b+5,12,128,128},at::kBFloat16);
  shape(6,{b+1,12,32,128},at::kHalf); shape(7,{b+1,4,32,128},at::kHalf); shape(8,{b+1,12,32},at::kFloat);
  for(int i=9;i<=13;++i) shape(i,{b},at::kInt);
  shape(14,{b,4,12,128},at::kBFloat16); shape(15,{b*12},at::kInt);
  TORCH_CHECK(sink.device()==dev && sink.scalar_type()==at::kFloat && sink.is_contiguous() &&
              sink.sizes()==at::IntArrayRef({b*12,4096}), "sink must be FP32[B*12,4096]");
  // Values are validated once by the experimental Python runner, before graph capture.
  // This entry is not exposed by the production library and accepts only its trusted fixture.
  const c10_npu::OptionalNPUGuard guard(dev);
  const auto stream=c10_npu::getCurrentNPUStream().stream();
  const uint32_t blocks=mode>=5 ? 1 : static_cast<uint32_t>(std::min(cores,b*12));
#define PROBE_ARGS blocks, stream, t[0].data_ptr(), t[1].data_ptr(), t[2].data_ptr(), t[3].data_ptr(), t[4].data_ptr(), t[5].data_ptr(), t[6].data_ptr(), t[7].data_ptr(), t[8].data_ptr(), t[9].data_ptr(), t[10].data_ptr(), t[11].data_ptr(), t[12].data_ptr(), t[13].data_ptr(), t[14].data_ptr(), t[15].data_ptr(), static_cast<uint32_t>(b), static_cast<uint32_t>(b+5), static_cast<uint32_t>(b+1), 4U, 12U, 32U, static_cast<uint64_t>(t[5].stride(0)), static_cast<uint64_t>(t[6].stride(0)), static_cast<uint64_t>(t[7].stride(0)), static_cast<uint64_t>(t[8].stride(0)), 128U, 0.08838834764831845f, sink.data_ptr(), static_cast<uint32_t>(repeats)
  aclError error=ACL_SUCCESS;
  switch(mode) {
    case 0: error=aclrtlaunch_replayssm_probe_baseline(PROBE_ARGS); break;
    case 1: error=aclrtlaunch_replayssm_probe_a1_control(PROBE_ARGS); break;
    case 2: error=aclrtlaunch_replayssm_probe_a1_replace(PROBE_ARGS); break;
    case 3: error=aclrtlaunch_replayssm_probe_a2_control(PROBE_ARGS); break;
    case 4: error=aclrtlaunch_replayssm_probe_a2_remove(PROBE_ARGS); break;
    case 5: error=aclrtlaunch_replayssm_probe_local(PROBE_ARGS); break;
    case 6: error=aclrtlaunch_replayssm_probe_u1_chunk4k(PROBE_ARGS); break;
    case 8: error=aclrtlaunch_replayssm_probe_u2_history_tree(PROBE_ARGS); break;
    case 7: error=aclrtlaunch_replayssm_probe_u1_chunk8k(PROBE_ARGS); break;
  }
#undef PROBE_ARGS
  TORCH_CHECK(error==ACL_SUCCESS,"probe launch failed: ",error);
}
TORCH_LIBRARY(replayssm_sd_probing,m) {
  m.def("run_(Tensor(a!)[] tensors, Tensor(b!) sink, int mode, int cores, int repeats, bool profiling_only) -> ()");
}
TORCH_LIBRARY_IMPL(replayssm_sd_probing,PrivateUse1,m) {m.impl("run_", &run_probe);}

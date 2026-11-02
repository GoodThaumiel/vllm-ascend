// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
#pragma once
#include <ATen/ATen.h>

#define VLLM_ASCEND_REPLAYSSM_SD_METADATA_PROBE_SCHEMA \
  "(Tensor slots, Tensor ring_slots, Tensor positions, Tensor bases, Tensor lengths, " \
  "Tensor packed_metadata, Tensor(a!) out, int heads=12, int cores=40, int mode=0) -> ()"

namespace vllm_ascend {
void replayssm_sd_metadata_probe(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                 const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t);
inline void replayssm_sd_metadata_probe_meta(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                              const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                              const at::Tensor&, int64_t, int64_t, int64_t) {}
}

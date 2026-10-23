// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
#include <torch/library.h>
#include "replayssm_sd_compact.h"

TORCH_LIBRARY(replayssm_sd_compact_lab, m) {
  m.def("compact_" VLLM_ASCEND_REPLAYSSM_SD_COMPACT_SCHEMA);
}
TORCH_LIBRARY_IMPL(replayssm_sd_compact_lab, PrivateUse1, m) {
  m.impl("compact_", &vllm_ascend::replayssm_sd_compact);
}
TORCH_LIBRARY_IMPL(replayssm_sd_compact_lab, Meta, m) {
  m.impl("compact_", &vllm_ascend::replayssm_sd_compact_meta);
}

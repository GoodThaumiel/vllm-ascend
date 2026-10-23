// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
#include <torch/library.h>
#include "replayssm_sd_cursor.h"

TORCH_LIBRARY(replayssm_sd_cursor_lab, m) { m.def("step_" VLLM_ASCEND_REPLAYSSM_SD_CURSOR_SCHEMA); }
TORCH_LIBRARY_IMPL(replayssm_sd_cursor_lab, PrivateUse1, m) {
  m.impl("step_", &vllm_ascend::replayssm_sd_cursor);
}
TORCH_LIBRARY_IMPL(replayssm_sd_cursor_lab, Meta, m) {
  m.impl("step_", &vllm_ascend::replayssm_sd_cursor_meta);
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
#include <torch/library.h>
#include "replayssm_ar.h"
#include "replayssm_sd.h"
#include "replayssm_sd_compact.h"
#include "replayssm_sd_cursor.h"

// Test the real shared schema and Meta function without compiling any NPU
// implementation. Run in an isolated CPU process, without the real extension.
TORCH_LIBRARY(_C_ascend, m) {
  m.def("replayssm_ar_step_" VLLM_ASCEND_REPLAYSSM_AR_SCHEMA);
  m.def("replayssm_sd_verify_" VLLM_ASCEND_REPLAYSSM_SD_SCHEMA);
  m.def("replayssm_sd_compact_" VLLM_ASCEND_REPLAYSSM_SD_COMPACT_SCHEMA);
  m.def("replayssm_sd_cursor_" VLLM_ASCEND_REPLAYSSM_SD_CURSOR_SCHEMA);
}
TORCH_LIBRARY_IMPL(_C_ascend, Meta, m) {
  m.impl("replayssm_ar_step_", &vllm_ascend::replayssm_ar_step_meta);
  m.impl("replayssm_sd_verify_", &vllm_ascend::replayssm_sd_verify_meta);
  m.impl("replayssm_sd_compact_", &vllm_ascend::replayssm_sd_compact_meta);
  m.impl("replayssm_sd_cursor_", &vllm_ascend::replayssm_sd_cursor_meta);
}

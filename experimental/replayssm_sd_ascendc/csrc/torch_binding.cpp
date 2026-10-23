// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
#include <torch/library.h>
#include "replayssm_sd.h"

TORCH_LIBRARY(replayssm_sd_lab, m) { m.def("verify_" VLLM_ASCEND_REPLAYSSM_SD_SCHEMA); }
TORCH_LIBRARY_IMPL(replayssm_sd_lab, PrivateUse1, m) { m.impl("verify_", &vllm_ascend::replayssm_sd_verify); }
TORCH_LIBRARY_IMPL(replayssm_sd_lab, Meta, m) { m.impl("verify_", &vllm_ascend::replayssm_sd_verify_meta); }

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
#include <torch/library.h>
#include "replayssm_ar.h"

TORCH_LIBRARY(replayssm_lab, m) { m.def("ar_step_" VLLM_ASCEND_REPLAYSSM_AR_SCHEMA); }
TORCH_LIBRARY_IMPL(replayssm_lab, PrivateUse1, m) { m.impl("ar_step_", &vllm_ascend::replayssm_ar_step); }
TORCH_LIBRARY_IMPL(replayssm_lab, Meta, m) { m.impl("ar_step_", &vllm_ascend::replayssm_ar_step_meta); }

/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
#include "arch20/compat_310p.h"
#endif
#include "arch20/compute_wy_kernel.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void replayssm_310p_compute_wy(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR q_kernel, GM_ADDR k_kernel, GM_ADDR w_kernel, GM_ADDR u_kernel, GM_ADDR g_kernel,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
    ChunkGatedDeltaRuleComputeWyTilingData tilingData;
    auto *dst = reinterpret_cast<uint32_t *>(&tilingData);
    auto *src = reinterpret_cast<__gm__ const uint32_t *>(tiling);
    for (uint32_t word = 0; word < sizeof(tilingData) / sizeof(uint32_t); ++word) {
        dst[word] = src[word];
    }

    // HAVE_WORKSPACE sets the system workspace and passes the user workspace here.
    TPipe pipe;
    ChunkGatedDeltaRuleComputeWy::KernelComputeWy op;
    op.Init(q, k, v, g, beta, q_kernel, k_kernel, w_kernel, u_kernel, g_kernel, workspace, &tilingData, &pipe);
    op.Process();
}

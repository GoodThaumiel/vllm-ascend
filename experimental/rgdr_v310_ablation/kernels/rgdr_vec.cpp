// SPDX-License-Identifier: Apache-2.0
// Isolated 310P native recurrent GDR ablations. The production ACLNN op is untouched.
#include "rgdr_vec_impl.h"

using namespace AscendC;
using namespace RecurrentGatedDeltaRuleV310;

template <bool DIRECT_TASKS, bool HOIST_METADATA>
__aicore__ inline void RunRGDR(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR state,
    GM_ADDR cuSeqlens, GM_ADDR ssmStateIndices, GM_ADDR gama,
    GM_ADDR numAcceptedTokens, GM_ADDR out, uint32_t batch,
    uint32_t totalTokens, uint32_t stateSlots, float scale)
{
    RecurrentGatedDeltaRuleV310TilingData tilingData{};
    tilingData.vectorCoreNum = GetBlockNum();
    tilingData.ubCalSize = 256 * 1024;
    // For NK=8, NV=16, DK=DV=128, FP16 and MAX_MTP=8, this is the
    // production host-tiling formula with vStep=128 and one queue buffer.
    tilingData.ubRestBytes = 189056;
    tilingData.t = totalTokens;
    tilingData.nk = 8;
    tilingData.dk = 128;
    tilingData.nv = 16;
    tilingData.dv = 128;
    tilingData.sBlockNum = stateSlots;
    tilingData.b = batch;
    tilingData.vStep = 128;
    tilingData.stateOutBufferNum = 1;
    tilingData.attnOutBufferNum = 1;
    tilingData.scale = scale;
    tilingData.hasGama = 1;
    tilingData.hasGamaK = 0;
    tilingData.hasAcceptedTokens = 1;
    RGDRInitParams params{query, key, value, gama, nullptr, beta, state,
                          cuSeqlens, ssmStateIndices, numAcceptedTokens,
                          out, state};
    TPipe pipe;
    RGDR<half, half, DIRECT_TASKS, HOIST_METADATA> op(&tilingData);
    op.Init(params, &pipe);
    op.Process();
}

#define RGDR_VECTOR_ENTRY(NAME, DIRECT_TASKS, HOIST_METADATA)                 \
    extern "C" __global__ __aicore__ void NAME(                                \
        GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR state, \
        GM_ADDR cuSeqlens, GM_ADDR ssmStateIndices, GM_ADDR gama,              \
        GM_ADDR numAcceptedTokens, GM_ADDR out, uint32_t batch,               \
        uint32_t totalTokens, uint32_t stateSlots, float scale)               \
    {                                                                         \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                       \
        RunRGDR<DIRECT_TASKS, HOIST_METADATA>(                                 \
            query, key, value, beta, state, cuSeqlens, ssmStateIndices, gama, \
            numAcceptedTokens, out, batch, totalTokens, stateSlots, scale);   \
    }

RGDR_VECTOR_ENTRY(rgdr_v310_vec0, false, false)
RGDR_VECTOR_ENTRY(rgdr_v310_vec1, true, false)
RGDR_VECTOR_ENTRY(rgdr_v310_vec2, false, true)
RGDR_VECTOR_ENTRY(rgdr_v310_vec3, true, true)

#undef RGDR_VECTOR_ENTRY

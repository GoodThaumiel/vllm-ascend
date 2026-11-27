// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
// V23 verify and suffix-weighted compact with a shared UB checkpoint.
#define REPLAY_SD_NO_ENTRY
#include "replayssm_sd_verify.cpp"

extern "C" __global__ __aicore__ void replayssm_sd_verify_compact_fused_v24(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR checkpoint,
    GM_ADDR dCache, GM_ADDR kCache, GM_ADDR gCache, GM_ADDR slots, GM_ADDR ringSlots,
    GM_ADDR positions, GM_ADDR bases, GM_ADDR lengths, GM_ADDR flush, GM_ADDR out,
    GM_ADDR verifyStatus, GM_ADDR compactStatus, uint32_t batch, uint32_t numSlots,
    uint32_t numRingSlots, uint32_t heads, uint32_t valueHeads, uint32_t ringLen,
    uint64_t checkpointSlotStride, uint64_t dSlotStride, uint64_t kSlotStride,
    uint64_t gSlotStride, uint32_t tileV, float scale) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  ReplaySDVerify<24> op;
  op.Init(q, k, v, g, beta, checkpoint, dCache, kCache, gCache, slots, ringSlots,
          positions, bases, lengths, out, verifyStatus, batch, numSlots,
          numRingSlots, heads, valueHeads, ringLen, checkpointSlotStride,
          dSlotStride, kSlotStride, gSlotStride, tileV, scale);
  op.SetFusionOutputs(flush, compactStatus);
  op.Process();
}

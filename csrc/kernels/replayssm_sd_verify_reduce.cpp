// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
// Compile-time ablation of the validated SD verify dataflow.
#define REPLAY_SD_NO_ENTRY
#include "replayssm_sd_verify.cpp"

extern "C" __global__ __aicore__ void replayssm_sd_verify_reduce(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR checkpoint, GM_ADDR dCache,
    GM_ADDR kCache, GM_ADDR gCache, GM_ADDR slots, GM_ADDR positions, GM_ADDR bases, GM_ADDR lengths,
    GM_ADDR out, GM_ADDR status, uint32_t batch, uint32_t numSlots, uint32_t heads, uint32_t valueHeads,
    uint32_t ringLen, uint64_t checkpointSlotStride, uint64_t dSlotStride, uint64_t kSlotStride,
    uint64_t gSlotStride, uint32_t tileV, float scale) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  ReplaySDVerify<1> op;
  op.Init(q, k, v, g, beta, checkpoint, dCache, kCache, gCache, slots, positions, bases, lengths, out,
          status, batch, numSlots, heads, valueHeads, ringLen, checkpointSlotStride, dSlotStride,
          kSlotStride, gSlotStride, tileV, scale);
  op.Process();
}

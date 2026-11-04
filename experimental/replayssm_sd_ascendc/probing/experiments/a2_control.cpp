// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
// Independent v13-derived ablation 20: chunk8k_gate_batch.
#define REPLAY_SD_NO_ENTRY
#include "verify_snapshot.h"

extern "C" __global__ __aicore__ void replayssm_probe_a2_control(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR checkpoint, GM_ADDR dCache,
    GM_ADDR kCache, GM_ADDR gCache, GM_ADDR slots, GM_ADDR ringSlots, GM_ADDR positions, GM_ADDR bases, GM_ADDR lengths,
    GM_ADDR out, GM_ADDR status, uint32_t batch, uint32_t numSlots, uint32_t numRingSlots, uint32_t heads, uint32_t valueHeads,
    uint32_t ringLen, uint64_t checkpointSlotStride, uint64_t dSlotStride, uint64_t kSlotStride,
    uint64_t gSlotStride, uint32_t tileV, float scale, GM_ADDR sink, uint32_t repeats) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  ReplaySDVerify<20, 3> op;
  op.Init(q, k, v, g, beta, checkpoint, dCache, kCache, gCache, slots, ringSlots, positions, bases, lengths, out,
          status, batch, numSlots, numRingSlots, heads, valueHeads, ringLen, checkpointSlotStride, dSlotStride,
          kSlotStride, gSlotStride, tileV, scale);
  op.SetSink(sink);
  for (uint32_t rep = 0; rep < repeats; ++rep) op.Process();
}

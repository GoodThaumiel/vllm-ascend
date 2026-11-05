// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

using namespace AscendC;

// Independent stages for the Ascend310P3 FP16/FP32 conversion investigation.
// Each input and output is 32-byte aligned; 16/32/64 elements need at most
// 256 bytes per tensor. Modes 0/1 test raw aligned I/O without Cast, 2 tests
// half->float, 3/4 test float->half, and 5/6 test both conversions together.
// Modes 4/6 deliberately retain the original CAST_RINT for diagnosis.
extern "C" __global__ __aicore__ void replayssm_sd_310p_cast_probe(
    GM_ADDR input, GM_ADDR output, uint32_t count, uint32_t mode) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  TBuf<QuePosition::VECCALC> inputBuffer, workBuffer, outputBuffer;
  pipe.InitBuffer(inputBuffer, 256);
  pipe.InitBuffer(workBuffer, 256);
  pipe.InitBuffer(outputBuffer, 256);

  GlobalTensor<half> sourceHalf, destinationHalf;
  GlobalTensor<float> sourceFloat, destinationFloat;
  sourceHalf.SetGlobalBuffer((__gm__ half*)input);
  destinationHalf.SetGlobalBuffer((__gm__ half*)output);
  sourceFloat.SetGlobalBuffer((__gm__ float*)input);
  destinationFloat.SetGlobalBuffer((__gm__ float*)output);
  auto inputHalf = inputBuffer.Get<half>();
  auto inputFloat = inputBuffer.Get<float>();
  auto workFloat = workBuffer.Get<float>();
  auto outputHalf = outputBuffer.Get<half>();

  const DataCopyParams halfBlocks{1, static_cast<uint16_t>(count / 16), 0, 0};
  const DataCopyParams floatBlocks{1, static_cast<uint16_t>(count / 8), 0, 0};
  if (mode == 0 || mode == 2 || mode == 5 || mode == 6) {
    DataCopy(inputHalf, sourceHalf, halfBlocks);
  } else {
    DataCopy(inputFloat, sourceFloat, floatBlocks);
  }

  if (mode == 0 || mode == 1) {
    // This isolates GM->UB and UB->GM from both Cast instructions.
    SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    if (mode == 0) {
      DataCopy(destinationHalf, inputHalf, halfBlocks);
    } else {
      DataCopy(destinationFloat, inputFloat, floatBlocks);
    }
    return;
  }

  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  if (mode == 2 || mode == 5 || mode == 6) {
    Cast(workFloat, inputHalf, RoundMode::CAST_NONE, count);
    if (mode == 2) {
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      DataCopy(destinationFloat, workFloat, floatBlocks);
      return;
    }
    PipeBarrier<PIPE_V>();
    if (mode == 5) {
      Cast(outputHalf, workFloat, RoundMode::CAST_NONE, count);
    } else {
      Cast(outputHalf, workFloat, RoundMode::CAST_RINT, count);
    }
  } else if (mode == 3) {
    Cast(outputHalf, inputFloat, RoundMode::CAST_NONE, count);
  } else {
    Cast(outputHalf, inputFloat, RoundMode::CAST_RINT, count);
  }
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
  DataCopy(destinationHalf, outputHalf, halfBlocks);
}

// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

using namespace AscendC;

// Each launch tests one primitive. The output is a dedicated, aligned 32-byte
// buffer so an unsupported short copy cannot corrupt another tensor.
extern "C" __global__ __aicore__ void replayssm_sd_310p_io_probe(
    GM_ADDR input, GM_ADDR output, uint32_t mode) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  GlobalTensor<int32_t> source, destination;
  source.SetGlobalBuffer((__gm__ int32_t*)input);
  destination.SetGlobalBuffer((__gm__ int32_t*)output);
  TPipe pipe;
  TQue<QuePosition::VECIN, 1> inputQueue;
  TQue<QuePosition::VECOUT, 1> outputQueue;
  pipe.InitBuffer(inputQueue, 1, 32);
  pipe.InitBuffer(outputQueue, 1, 32);
  DataCopyParams aligned{1, 1, 0, 0};  // one 32-byte block, as in native V310

  if (mode == 0 || mode == 2) {
    auto local = inputQueue.AllocTensor<int32_t>();
    if (mode == 0) {
      DataCopy(local, source, aligned);
    } else {
      DataCopyExtParams cp{1, 4, 0, 0, 0};
      DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
      DataCopyPad(local, source, cp, pad);
    }
    inputQueue.EnQue(local);
    local = inputQueue.DeQue<int32_t>();
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    auto out = outputQueue.AllocTensor<int32_t>();
    for (uint32_t i = 0; i < 8; ++i) {
      out.SetValue(i, local.GetValue(mode == 0 ? i : 0));
    }
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    outputQueue.EnQue(out);
    out = outputQueue.DeQue<int32_t>();
    DataCopy(destination, out, aligned);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    outputQueue.FreeTensor(out);
    inputQueue.FreeTensor(local);
    return;
  }

  if (mode == 5) {
    auto before = inputQueue.AllocTensor<int32_t>();
    DataCopy(before, destination, aligned);
    inputQueue.EnQue(before);
    before = inputQueue.DeQue<int32_t>();
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    auto updated = outputQueue.AllocTensor<int32_t>();
    for (uint32_t i = 0; i < 8; ++i) {
      updated.SetValue(i, i == 0 ? 0x1234 : before.GetValue(i));
    }
    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    inputQueue.FreeTensor(before);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    outputQueue.EnQue(updated);
    updated = outputQueue.DeQue<int32_t>();
    DataCopy(destination, updated, aligned);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    outputQueue.FreeTensor(updated);
    return;
  }

  auto out = outputQueue.AllocTensor<int32_t>();
  const int32_t value = mode == 4 ? static_cast<int32_t>(GetBlockNum()) : 0x1234;
  for (uint32_t i = 0; i < 8; ++i) out.SetValue(i, value);
  SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
  outputQueue.EnQue(out);
  out = outputQueue.DeQue<int32_t>();
  if (mode == 3) {
    DataCopyExtParams cp{1, 4, 0, 0, 0};
    DataCopyPad(destination, out, cp);
  } else {
    DataCopy(destination, out, aligned);
  }
  SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
  WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
  outputQueue.FreeTensor(out);
}

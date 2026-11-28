// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void replayssm_roofline_vector(GM_ADDR out, uint32_t repeats) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  constexpr uint32_t width = 4096;
  constexpr uint32_t streams = 4;
  TPipe pipe;
  TBuf<TPosition::VECCALC> inputs;
  TQue<TPosition::VECOUT, 1> outputs;
  pipe.InitBuffer(inputs, 2 * width * sizeof(float));
  pipe.InitBuffer(outputs, 1, streams * width * sizeof(float));
  auto a = inputs.Get<float>();
  auto b = a[width];
  auto accum = outputs.AllocTensor<float>();
  Duplicate(a, 0.015625f, width);
  Duplicate(b, 0.125f, width);
  Duplicate(accum, 0.0f, width);
  Duplicate(accum[width], 1.0f, width);
  Duplicate(accum[2 * width], 2.0f, width);
  Duplicate(accum[3 * width], 3.0f, width);
  PipeBarrier<PIPE_V>();

  for (uint32_t i = 0; i < repeats; ++i) {
#pragma unroll
    for (uint32_t index = 0; index < streams; ++index) {
      MulAddDst(accum[index * width], a, b, width);
    }
    PipeBarrier<PIPE_V>();
  }

  GlobalTensor<float> target;
  target.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out) + GetBlockIdx() * streams * width);
  outputs.EnQue(accum);
  accum = outputs.DeQue<float>();
  DataCopy(target, accum, streams * width);
  outputs.FreeTensor(accum);
}

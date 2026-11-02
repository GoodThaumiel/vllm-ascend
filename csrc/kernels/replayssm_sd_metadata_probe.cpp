// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
#include "kernel_operator.h"

using namespace AscendC;

class ReplaySDMetadataProbe {
 public:
  __aicore__ inline void Init(GM_ADDR slots, GM_ADDR ringSlots, GM_ADDR positions,
                              GM_ADDR bases, GM_ADDR lengths, GM_ADDR packed, GM_ADDR out,
                              uint32_t batch, uint32_t heads, uint32_t mode) {
    slots_.SetGlobalBuffer((__gm__ int32_t*)slots);
    ringSlots_.SetGlobalBuffer((__gm__ int32_t*)ringSlots);
    positions_.SetGlobalBuffer((__gm__ int32_t*)positions);
    bases_.SetGlobalBuffer((__gm__ int32_t*)bases);
    lengths_.SetGlobalBuffer((__gm__ int32_t*)lengths);
    packed_.SetGlobalBuffer((__gm__ int32_t*)packed);
    out_.SetGlobalBuffer((__gm__ int32_t*)out);
    batch_ = batch;
    heads_ = heads;
    mode_ = mode;
    pipe_.InitBuffer(input_, 1, 32);
    pipe_.InitBuffer(output_, 1, 32);
  }

  __aicore__ inline void Process() {
    for (uint32_t task = GetBlockIdx(); task < batch_ * heads_; task += GetBlockNum()) {
      const uint32_t row = task / heads_;
      int32_t values[5];
      if (mode_ == 0) {
        values[0] = ReadOne(slots_, row);
        values[1] = ReadOne(ringSlots_, row);
        values[2] = ReadOne(positions_, row);
        values[3] = ReadOne(bases_, row);
        values[4] = ReadOne(lengths_, row);
      } else {
        auto local = input_.AllocTensor<int32_t>();
        DataCopyExtParams cp{1, 32, 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(local, packed_[row * 8], cp, pad);
        input_.EnQue(local);
        local = input_.DeQue<int32_t>();
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        for (uint32_t i = 0; i < 5; ++i) values[i] = local.GetValue(i);
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        input_.FreeTensor(local);
      }
      // One aligned 32-byte output block per task avoids cross-core writes to
      // the same block and makes the metadata contents independently visible.
      auto result = output_.AllocTensor<int32_t>();
      for (uint32_t i = 0; i < 5; ++i) result.SetValue(i, values[i]);
      for (uint32_t i = 5; i < 8; ++i) result.SetValue(i, 0);
      SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
      output_.EnQue(result);
      result = output_.DeQue<int32_t>();
      DataCopyExtParams cp{1, 32, 0, 0, 0};
      DataCopyPad(out_[task * 8], result, cp);
      SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
      output_.FreeTensor(result);
    }
  }

 private:
  __aicore__ inline int32_t ReadOne(GlobalTensor<int32_t>& src, uint32_t row) {
    auto local = input_.AllocTensor<int32_t>();
    DataCopyExtParams cp{1, sizeof(int32_t), 0, 0, 0};
    DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    DataCopyPad(local, src[row], cp, pad);
    input_.EnQue(local);
    local = input_.DeQue<int32_t>();
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    int32_t value = local.GetValue(0);
    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    input_.FreeTensor(local);
    return value;
  }
  TPipe pipe_;
  TQue<QuePosition::VECIN, 1> input_;
  TQue<QuePosition::VECOUT, 1> output_;
  GlobalTensor<int32_t> slots_, ringSlots_, positions_, bases_, lengths_, packed_, out_;
  uint32_t batch_, heads_, mode_;
};

extern "C" __global__ __aicore__ void replayssm_sd_metadata_probe(
    GM_ADDR slots, GM_ADDR ringSlots, GM_ADDR positions, GM_ADDR bases,
    GM_ADDR lengths, GM_ADDR packed, GM_ADDR out, uint32_t batch, uint32_t heads, uint32_t mode) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  ReplaySDMetadataProbe op;
  op.Init(slots, ringSlots, positions, bases, lengths, packed, out, batch, heads, mode);
  op.Process();
}

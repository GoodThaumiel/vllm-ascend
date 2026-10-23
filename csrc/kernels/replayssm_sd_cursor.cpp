// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

#include "kernel_operator.h"

using namespace AscendC;

namespace {
// Cursor work is O(B) scalar metadata. One AIV core owns every update so
// different cores never write different words of one GM cache line. The caller
// guarantees each positive physical slot appears once in the row list.
class ReplaySDCursor {
 public:
  __aicore__ inline void Init(GM_ADDR writePos, GM_ADDR cacheBase, GM_ADDR isFlush,
                              GM_ADDR accepted, GM_ADDR slots, GM_ADDR doReset,
                              GM_ADDR status, uint32_t batch, uint32_t numSlots,
                              uint32_t logicalLimit, uint32_t maxSpecLen,
                              uint32_t ringLen) {
    wp_.SetGlobalBuffer((__gm__ int32_t*)writePos);
    base_.SetGlobalBuffer((__gm__ int32_t*)cacheBase);
    flush_.SetGlobalBuffer((__gm__ int8_t*)isFlush);
    accepted_.SetGlobalBuffer((__gm__ int32_t*)accepted);
    slots_.SetGlobalBuffer((__gm__ int32_t*)slots);
    reset_.SetGlobalBuffer((__gm__ int8_t*)doReset);
    status_.SetGlobalBuffer((__gm__ int32_t*)status);
    batch_ = batch;
    numSlots_ = numSlots;
    logicalLimit_ = logicalLimit;
    maxSpecLen_ = maxSpecLen;
    ringLen_ = ringLen;
    pipe_.InitBuffer(in_, 1, 32);
    pipe_.InitBuffer(out_, 1, 32);
  }

  __aicore__ inline void Process() {
    const int32_t initialFlush = 2 * maxSpecLen_ > logicalLimit_ ? 1 : 0;
    for (uint32_t row = 0; row < batch_; ++row) {
      const int32_t slot = Read(slots_, row);
      if (slot == 0) {
        Write(status_, row, 0);
        continue;
      }
      if (slot < 0 || slot >= static_cast<int32_t>(numSlots_)) {
        Write(status_, row, 1);
        continue;
      }
      // Reset wins over stale acceptance and stale cursor metadata at a
      // prefill-to-decode handoff or resumed request. Slot 0 remains padding.
      if (Read(reset_, row) != 0) {
        Write(wp_, slot, 0);
        Write(base_, slot, 0);
        Write(flush_, slot, static_cast<int8_t>(initialFlush));
        Write(status_, row, 0);
        continue;
      }

      const int32_t oldWp = Read(wp_, slot);
      const int32_t oldBase = Read(base_, slot);
      const int32_t oldFlush = Read(flush_, slot);
      const int32_t n = Read(accepted_, row);
      if (n < 0 || n > static_cast<int32_t>(maxSpecLen_)) {
        Write(status_, row, 2);
        continue;
      }
      if (oldWp < 0 || oldWp > static_cast<int32_t>(logicalLimit_) ||
          oldBase < 0 || oldBase >= static_cast<int32_t>(ringLen_) ||
          (oldFlush != 0 && oldFlush != 1)) {
        Write(status_, row, 3);
        continue;
      }
      if (n == 0) {
        Write(status_, row, 0);
        continue;
      }

      const int32_t newWp = (oldFlush != 0 ? 0 : oldWp) + n;
      if (newWp > static_cast<int32_t>(logicalLimit_)) {
        Write(status_, row, 4);
        continue;
      }
      const int32_t newBase = oldFlush != 0 ? (oldBase + oldWp) & (ringLen_ - 1) : oldBase;
      Write(base_, slot, newBase);
      Write(wp_, slot, newWp);
      Write(flush_, slot, static_cast<int8_t>(newWp + 2 * maxSpecLen_ > logicalLimit_));
      Write(status_, row, 0);
    }
  }

 private:
  template <typename T>
  __aicore__ inline T Read(GlobalTensor<T>& source, uint32_t offset) {
    auto local = in_.AllocTensor<T>();
    DataCopyExtParams cp{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    DataCopyPad(local, source[offset], cp, pad);
    in_.EnQue(local);
    local = in_.DeQue<T>();
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    T value = local.GetValue(0);
    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    in_.FreeTensor(local);
    return value;
  }

  template <typename T>
  __aicore__ inline void Write(GlobalTensor<T>& dest, uint32_t offset, T value) {
    auto local = out_.AllocTensor<T>();
    local.SetValue(0, value);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    out_.EnQue(local);
    local = out_.DeQue<T>();
    DataCopyExtParams cp{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPad(dest[offset], local, cp);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    out_.FreeTensor(local);
  }

  TPipe pipe_;
  TQue<QuePosition::VECIN, 1> in_;
  TQue<QuePosition::VECOUT, 1> out_;
  GlobalTensor<int32_t> wp_, base_, accepted_, slots_, status_;
  GlobalTensor<int8_t> flush_, reset_;
  uint32_t batch_, numSlots_, logicalLimit_, maxSpecLen_, ringLen_;
};
}  // namespace

extern "C" __global__ __aicore__ void replayssm_sd_cursor(
    GM_ADDR writePos, GM_ADDR cacheBase, GM_ADDR isFlush, GM_ADDR accepted,
    GM_ADDR slots, GM_ADDR doReset, GM_ADDR status, uint32_t batch,
    uint32_t numSlots, uint32_t logicalLimit, uint32_t maxSpecLen,
    uint32_t ringLen) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  ReplaySDCursor op;
  op.Init(writePos, cacheBase, isFlush, accepted, slots, doReset, status,
          batch, numSlots, logicalLimit, maxSpecLen, ringLen);
  op.Process();
}

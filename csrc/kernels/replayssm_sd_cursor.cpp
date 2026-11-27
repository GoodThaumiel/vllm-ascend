// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr int32_t FLUSH_BIT = 1;
constexpr int32_t FIRST_FLUSH_PENDING = 2;

// Cursor work is O(B) scalar metadata. One AIV core owns every update so
// different cores never write different words of one GM cache line. The caller
// guarantees each positive physical slot appears once in the row list.
class ReplaySDCursor {
 public:
  __aicore__ inline void Init(GM_ADDR writePos, GM_ADDR cacheBase, GM_ADDR isFlush,
                              GM_ADDR accepted, GM_ADDR slots, GM_ADDR doReset,
                              GM_ADDR status, uint32_t batch, uint32_t numSlots,
                              uint32_t logicalLimit, uint32_t maxSpecLen,
                              uint32_t ringLen, uint32_t staggerFirst) {
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
    staggerFirst_ = staggerFirst != 0;
    phaseCount_ = staggerFirst_ ? 1 + (logicalLimit_ - 2 * maxSpecLen_) / maxSpecLen_ : 1;
    if (UseBatchTransfers()) {
      const uint32_t rowBytes = Align32(batch_ * sizeof(int32_t));
      const uint32_t slotBytes = Align32(numSlots_ * sizeof(int32_t));
      pipe_.InitBuffer(batchInput_, 2 * rowBytes + Align32(batch_) + 2 * slotBytes + Align32(numSlots_));
      pipe_.InitBuffer(batchOutput_, rowBytes + 3 * 32 * batch_ + Align32(batch_));
    } else {
      pipe_.InitBuffer(in_, 1, 32);
      pipe_.InitBuffer(out_, 1, 32);
    }
  }

  __aicore__ inline void Process() {
    if (UseBatchTransfers()) {
      ProcessBatched();
      return;
    }
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
        Write(flush_, slot, NextFlags(slot, 0, staggerFirst_));
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
          !ValidFlags(oldFlush)) {
        Write(status_, row, 3);
        continue;
      }
      if (n == 0) {
        Write(status_, row, 0);
        continue;
      }

      const bool folded = (oldFlush & FLUSH_BIT) != 0;
      const int32_t newWp = (folded ? 0 : oldWp) + n;
      if (newWp > static_cast<int32_t>(logicalLimit_)) {
        Write(status_, row, 4);
        continue;
      }
      const int32_t newBase = folded ? (oldBase + oldWp) & (ringLen_ - 1) : oldBase;
      Write(base_, slot, newBase);
      Write(wp_, slot, newWp);
      Write(flush_, slot, NextFlags(slot, newWp, !folded && (oldFlush & FIRST_FLUSH_PENDING)));
      Write(status_, row, 0);
    }
  }

 private:
  __aicore__ inline bool ValidFlags(int32_t flags) const {
    const int32_t validBits = staggerFirst_ ? FLUSH_BIT | FIRST_FLUSH_PENDING : FLUSH_BIT;
    return flags >= 0 && (flags & ~validBits) == 0;
  }

  __aicore__ inline int8_t NextFlags(uint32_t slot, uint32_t pos, bool firstPending) const {
    uint32_t limit = logicalLimit_;
    if (firstPending) {
      // Shorten only the first window. The pending bit is independent of
      // the physical ring base, which can wrap back to zero later.
      const uint32_t index = slot - 1;
      // Rotate assignments across consecutive groups so a phase cohort
      // does not repeatedly align with the same Cube block interval.
      limit -= ((index + index / phaseCount_) % phaseCount_) * maxSpecLen_;
    }
    return static_cast<int8_t>((firstPending ? FIRST_FLUSH_PENDING : 0) |
                               (pos + 2 * maxSpecLen_ > limit ? FLUSH_BIT : 0));
  }

  __aicore__ inline bool UseBatchTransfers() const {
    return batch_ >= 32 && batch_ <= 1024 && numSlots_ <= 1024;
  }

  __aicore__ inline uint32_t Align32(uint32_t bytes) const {
    return (bytes + 31) / 32 * 32;
  }

  template <typename T>
  __aicore__ inline void CopyIn(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t count) {
    const DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    const DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    DataCopyPad(dst, src, cp, pad);
  }

  template <typename T>
  __aicore__ inline void CopyOut(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t count) {
    const DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPad(dst, src, cp);
  }

  __aicore__ inline void ProcessBatched() {
    const uint32_t rowBytes = Align32(batch_ * sizeof(int32_t));
    const uint32_t slotBytes = Align32(numSlots_ * sizeof(int32_t));
    auto input = batchInput_.Get<int8_t>();
    auto slots = input.ReinterpretCast<int32_t>();
    auto accepted = input[rowBytes].ReinterpretCast<int32_t>();
    auto resets = input[2 * rowBytes];
    const uint32_t cursorOffset = 2 * rowBytes + Align32(batch_);
    auto positions = input[cursorOffset].ReinterpretCast<int32_t>();
    auto bases = input[cursorOffset + slotBytes].ReinterpretCast<int32_t>();
    auto flushes = input[cursorOffset + 2 * slotBytes];
    CopyIn(slots, slots_, batch_);
    CopyIn(accepted, accepted_, batch_);
    CopyIn(resets, reset_, batch_);
    CopyIn(positions, wp_, numSlots_);
    CopyIn(bases, base_, numSlots_);
    CopyIn(flushes, flush_, numSlots_);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

    auto output = batchOutput_.Get<int8_t>();
    auto statuses = output.ReinterpretCast<int32_t>();
    auto newPositions = output[rowBytes].ReinterpretCast<int32_t>();
    auto newBases = output[rowBytes + 32 * batch_].ReinterpretCast<int32_t>();
    auto newFlushes = output[rowBytes + 2 * 32 * batch_];
    auto writeRows = output[rowBytes + 3 * 32 * batch_];
    for (uint32_t row = 0; row < batch_; ++row) {
      const int32_t slot = slots.GetValue(row);
      const int32_t n = accepted.GetValue(row);
      int32_t status = 0;
      int32_t newPos = 0;
      int32_t newBase = 0;
      bool firstPending = false;
      bool write = false;
      if (slot < 0 || slot >= static_cast<int32_t>(numSlots_)) {
        status = 1;
      } else if (slot != 0) {
        if (resets.GetValue(row) != 0) {
          firstPending = staggerFirst_;
          write = true;
        } else if (n < 0 || n > static_cast<int32_t>(maxSpecLen_)) {
          status = 2;
        } else {
          const int32_t oldPos = positions.GetValue(slot);
          const int32_t oldBase = bases.GetValue(slot);
          const int32_t oldFlush = flushes.GetValue(slot);
          if (oldPos < 0 || oldPos > static_cast<int32_t>(logicalLimit_) ||
              oldBase < 0 || oldBase >= static_cast<int32_t>(ringLen_) ||
              !ValidFlags(oldFlush)) {
            status = 3;
          } else if (n != 0) {
            const bool folded = (oldFlush & FLUSH_BIT) != 0;
            newPos = (folded ? 0 : oldPos) + n;
            newBase = folded ? (oldBase + oldPos) & (ringLen_ - 1) : oldBase;
            firstPending = !folded && (oldFlush & FIRST_FLUSH_PENDING);
            if (newPos > static_cast<int32_t>(logicalLimit_)) {
              status = 4;
            } else {
              write = true;
            }
          }
        }
      }
      statuses.SetValue(row, status);
      writeRows.SetValue(row, static_cast<int8_t>(write));
      if (write) {
        newPositions.SetValue(row * 8, newPos);
        newBases.SetValue(row * 8, newBase);
        newFlushes.SetValue(row * 32, NextFlags(slot, newPos, firstPending));
      }
    }

    // A row owns one aligned UB block per mutable field. Individual GM
    // stores retain the existing slot ownership and skip unlisted slots.
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    for (uint32_t row = 0; row < batch_; ++row) {
      if (writeRows.GetValue(row) != 0) {
        const int32_t slot = slots.GetValue(row);
        CopyOut(wp_[slot], newPositions[row * 8], 1);
        CopyOut(base_[slot], newBases[row * 8], 1);
        CopyOut(flush_[slot], newFlushes[row * 32], 1);
      }
    }
    CopyOut(status_, statuses, batch_);
  }

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
  TBuf<QuePosition::VECCALC> batchInput_, batchOutput_;
  GlobalTensor<int32_t> wp_, base_, accepted_, slots_, status_;
  GlobalTensor<int8_t> flush_, reset_;
  uint32_t batch_, numSlots_, logicalLimit_, maxSpecLen_, ringLen_, phaseCount_;
  bool staggerFirst_;
};
}  // namespace

extern "C" __global__ __aicore__ void replayssm_sd_cursor(
    GM_ADDR writePos, GM_ADDR cacheBase, GM_ADDR isFlush, GM_ADDR accepted,
    GM_ADDR slots, GM_ADDR doReset, GM_ADDR status, uint32_t batch,
    uint32_t numSlots, uint32_t logicalLimit, uint32_t maxSpecLen,
    uint32_t ringLen, uint32_t staggerFirst) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  ReplaySDCursor op;
  op.Init(writePos, cacheBase, isFlush, accepted, slots, doReset, status,
          batch, numSlots, logicalLimit, maxSpecLen, ringLen, staggerFirst);
  op.Process();
}

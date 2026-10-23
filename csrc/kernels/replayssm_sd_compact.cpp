// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t K_DIM = 128;
constexpr uint32_t V_DIM = 128;
constexpr uint32_t MAX_HISTORY = 16;
constexpr uint32_t FP32_BLOCK = 8;

// Run after SD verify and before accepted-token cursor commit. A flush row
// folds only the *previously committed* [0, position) ring records into the
// checkpoint. Draft records written by verify start at position and are not
// read here. The subsequent cursor kernel advances base by the old position.
class ReplaySDCompact {
 public:
  __aicore__ inline void Init(GM_ADDR checkpoint, GM_ADDR dCache, GM_ADDR kCache, GM_ADDR gCache,
                              GM_ADDR slots, GM_ADDR positions, GM_ADDR bases, GM_ADDR flush,
                              GM_ADDR status, uint32_t batch, uint32_t numSlots, uint32_t heads,
                              uint32_t valueHeads, uint32_t ringLen, uint64_t checkpointStride,
                              uint64_t dStride, uint64_t kStride, uint64_t gStride, uint32_t tileV) {
    checkpoint_.SetGlobalBuffer((__gm__ bfloat16_t*)checkpoint);
    dCache_.SetGlobalBuffer((__gm__ half*)dCache);
    kCache_.SetGlobalBuffer((__gm__ half*)kCache);
    gCache_.SetGlobalBuffer((__gm__ float*)gCache);
    slots_.SetGlobalBuffer((__gm__ int32_t*)slots);
    positions_.SetGlobalBuffer((__gm__ int32_t*)positions);
    bases_.SetGlobalBuffer((__gm__ int32_t*)bases);
    flush_.SetGlobalBuffer((__gm__ int32_t*)flush);
    status_.SetGlobalBuffer((__gm__ int32_t*)status);
    batch_ = batch;
    numSlots_ = numSlots;
    heads_ = heads;
    valueHeads_ = valueHeads;
    ringLen_ = ringLen;
    checkpointStride_ = checkpointStride;
    dStride_ = dStride;
    kStride_ = kStride;
    gStride_ = gStride;
    tileV_ = tileV;
    tiles_ = V_DIM / tileV;

    // At tile_v=32: two 16-KiB state/product tiles, 8-KiB input,
    // 8-KiB output plus <1 KiB vectors. No per-history materialization.
    pipe_.InitBuffer(in_, 1, tileV * K_DIM * sizeof(bfloat16_t));
    pipe_.InitBuffer(out_, 1, tileV * K_DIM * sizeof(bfloat16_t));
    pipe_.InitBuffer(stateBuffer_, tileV * K_DIM * sizeof(float));
    pipe_.InitBuffer(productBuffer_, tileV * K_DIM * sizeof(float));
    pipe_.InitBuffer(vectorBuffer_, (K_DIM + tileV_ + FP32_BLOCK) * sizeof(float));
    state_ = stateBuffer_.Get<float>();
    product_ = productBuffer_.Get<float>();
    auto vec = vectorBuffer_.Get<float>();
    key_ = vec;
    delta_ = vec[K_DIM];
    scalar_ = vec[K_DIM + tileV_];
  }

  __aicore__ inline void Process() {
    for (uint32_t task = GetBlockIdx(); task < batch_ * valueHeads_ * tiles_; task += GetBlockNum()) {
      const uint32_t row = task / (valueHeads_ * tiles_);
      const uint32_t head = (task / tiles_) % valueHeads_;
      const uint32_t vStart = (task % tiles_) * tileV_;
      const uint32_t keyHead = head / (valueHeads_ / heads_);
      const int32_t slot = ReadInt(slots_, row);
      const int32_t pos = ReadInt(positions_, row);
      const int32_t base = ReadInt(bases_, row);
      const int32_t flush = ReadInt(flush_, row);
      int32_t error = 0;
      if (slot < 0 || slot >= static_cast<int32_t>(numSlots_)) {
        error = 1;
      } else if (slot == 0) {
        if (pos != 0 || base != 0 || flush != 0) {
          error = 4;
        }
      } else if (pos < 0 || pos > static_cast<int32_t>(MAX_HISTORY)) {
        error = 2;
      } else if (base < 0 || base >= static_cast<int32_t>(ringLen_)) {
        error = 3;
      } else if (flush != 0 && flush != 1) {
        error = 5;
      }
      if (error != 0 || slot == 0 || flush == 0 || pos == 0) {
        WriteStatus(task, error);
        continue;
      }

      const uint64_t stateOffset = static_cast<uint64_t>(slot) * checkpointStride_ +
                                   static_cast<uint64_t>(head) * V_DIM * K_DIM + vStart * K_DIM;
      const uint64_t dOffset = static_cast<uint64_t>(slot) * dStride_ +
                               static_cast<uint64_t>(head) * ringLen_ * V_DIM;
      const uint64_t kOffset = static_cast<uint64_t>(slot) * kStride_ +
                               static_cast<uint64_t>(keyHead) * ringLen_ * K_DIM;
      const uint64_t gOffset = static_cast<uint64_t>(slot) * gStride_ +
                               static_cast<uint64_t>(head) * ringLen_;
      Load(checkpoint_, stateOffset, state_, tileV_ * K_DIM);
      for (int32_t i = 0; i < pos; ++i) {
        const uint32_t phys = (static_cast<uint32_t>(base) + static_cast<uint32_t>(i)) & (ringLen_ - 1);
        Load(kCache_, kOffset + static_cast<uint64_t>(phys) * K_DIM, key_, K_DIM);
        LoadD(dOffset + static_cast<uint64_t>(phys) * V_DIM + vStart);
        Load(gCache_, gOffset + phys, scalar_, 1);
        Exp(scalar_, scalar_, 1);
        PipeBarrier<PIPE_V>();
        const float alpha = ReadLocal(scalar_, 0);
        Muls(state_, state_, alpha, tileV_ * K_DIM);
        PipeBarrier<PIPE_V>();
        OuterAdd();
      }
      Store(checkpoint_, stateOffset, state_, tileV_ * K_DIM);
      WriteStatus(task, 0);
    }
  }

 private:
  template <typename T>
  __aicore__ inline void Load(GlobalTensor<T>& source, uint64_t offset, LocalTensor<float> dest,
                              uint32_t count) {
    auto input = in_.AllocTensor<T>();
    DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    DataCopyPad(input, source[offset], cp, pad);
    in_.EnQue(input);
    input = in_.DeQue<T>();
    if constexpr (sizeof(T) == sizeof(float)) {
      DataCopy(dest, input, (count + FP32_BLOCK - 1) / FP32_BLOCK * FP32_BLOCK);
    } else {
      Cast(dest, input, RoundMode::CAST_NONE, count);
    }
    PipeBarrier<PIPE_V>();
    in_.FreeTensor(input);
  }

  __aicore__ inline void LoadD(uint64_t offset) {
    Load(dCache_, offset, delta_, tileV_);
  }

  __aicore__ inline int32_t ReadInt(GlobalTensor<int32_t>& source, uint32_t offset) {
    auto input = in_.AllocTensor<int32_t>();
    DataCopyExtParams cp{1, sizeof(int32_t), 0, 0, 0};
    DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    DataCopyPad(input, source[offset], cp, pad);
    in_.EnQue(input);
    input = in_.DeQue<int32_t>();
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    const int32_t value = input.GetValue(0);
    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    in_.FreeTensor(input);
    return value;
  }

  __aicore__ inline float ReadLocal(LocalTensor<float> source, uint32_t offset) {
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    return source.GetValue(offset);
  }

  __aicore__ inline void OuterAdd() {
    uint32_t shape[2] = {tileV_, K_DIM};
    uint32_t deltaShape[2] = {tileV_, 1};
    Broadcast<float, 2, 1>(product_, delta_, shape, deltaShape);
    PipeBarrier<PIPE_V>();
    Mul(product_, product_, key_, 64, tileV_, {1, 1, 1, 16, 16, 0});
    Mul(product_[64], product_[64], key_[64], 64, tileV_, {1, 1, 1, 16, 16, 0});
    PipeBarrier<PIPE_V>();
    Add(state_, state_, product_, tileV_ * K_DIM);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void Store(GlobalTensor<bfloat16_t>& dest, uint64_t offset,
                               LocalTensor<float> source, uint32_t count) {
    auto output = out_.AllocTensor<bfloat16_t>();
    Cast(output, source, RoundMode::CAST_RINT, count);
    out_.EnQue(output);
    output = out_.DeQue<bfloat16_t>();
    DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(bfloat16_t)), 0, 0, 0};
    DataCopyPad(dest[offset], output, cp);
    out_.FreeTensor(output);
  }

  __aicore__ inline void WriteStatus(uint32_t task, int32_t error) {
    auto output = out_.AllocTensor<int32_t>();
    output.SetValue(0, error);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    out_.EnQue(output);
    output = out_.DeQue<int32_t>();
    DataCopyExtParams cp{1, sizeof(int32_t), 0, 0, 0};
    DataCopyPad(status_[task], output, cp);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    out_.FreeTensor(output);
  }

  TPipe pipe_;
  TQue<QuePosition::VECIN, 1> in_;
  TQue<QuePosition::VECOUT, 1> out_;
  TBuf<QuePosition::VECCALC> stateBuffer_, productBuffer_, vectorBuffer_;
  LocalTensor<float> state_, product_, key_, delta_, scalar_;
  GlobalTensor<bfloat16_t> checkpoint_;
  GlobalTensor<half> dCache_, kCache_;
  GlobalTensor<float> gCache_;
  GlobalTensor<int32_t> slots_, positions_, bases_, flush_, status_;
  uint32_t batch_, numSlots_, heads_, valueHeads_, ringLen_, tileV_, tiles_;
  uint64_t checkpointStride_, dStride_, kStride_, gStride_;
};
}  // namespace

extern "C" __global__ __aicore__ void replayssm_sd_compact(
    GM_ADDR checkpoint, GM_ADDR dCache, GM_ADDR kCache, GM_ADDR gCache, GM_ADDR slots,
    GM_ADDR positions, GM_ADDR bases, GM_ADDR flush, GM_ADDR status, uint32_t batch,
    uint32_t numSlots, uint32_t heads, uint32_t valueHeads, uint32_t ringLen,
    uint64_t checkpointStride, uint64_t dStride, uint64_t kStride, uint64_t gStride,
    uint32_t tileV) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  ReplaySDCompact op;
  op.Init(checkpoint, dCache, kCache, gCache, slots, positions, bases, flush, status, batch,
          numSlots, heads, valueHeads, ringLen, checkpointStride, dStride, kStride, gStride, tileV);
  op.Process();
}

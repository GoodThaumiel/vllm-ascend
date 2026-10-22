// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t K_DIM = 128;
constexpr uint32_t V_DIM = 128;
constexpr uint32_t MAX_HISTORY = 64;
constexpr uint32_t FP32_BLOCK = 8;

// A fresh vector-only implementation. The layout and queue/reduction structure
// follow the native RGDR design; there is no Cube/Vector cross-core handshake.
class ReplayAR {
 public:
  __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR checkpoint,
                              GM_ADDR dCache, GM_ADDR kCache, GM_ADDR gCache, GM_ADDR slots, GM_ADDR positions,
                              GM_ADDR flush, GM_ADDR out, GM_ADDR status, uint32_t batch, uint32_t numSlots,
                              uint32_t heads, uint32_t valueHeads, uint32_t capacity,
                              uint64_t checkpointSlotStride, uint64_t dCacheSlotStride, uint64_t kCacheSlotStride,
                              uint64_t gCacheSlotStride, uint32_t tileV, float scale) {
    q_.SetGlobalBuffer((__gm__ bfloat16_t*)q);
    k_.SetGlobalBuffer((__gm__ bfloat16_t*)k);
    v_.SetGlobalBuffer((__gm__ bfloat16_t*)v);
    g_.SetGlobalBuffer((__gm__ float*)g);
    beta_.SetGlobalBuffer((__gm__ float*)beta);
    checkpoint_.SetGlobalBuffer((__gm__ bfloat16_t*)checkpoint);
    dCache_.SetGlobalBuffer((__gm__ half*)dCache);
    kCache_.SetGlobalBuffer((__gm__ half*)kCache);
    gCache_.SetGlobalBuffer((__gm__ float*)gCache);
    slots_.SetGlobalBuffer((__gm__ int32_t*)slots);
    positions_.SetGlobalBuffer((__gm__ int32_t*)positions);
    flush_.SetGlobalBuffer((__gm__ int32_t*)flush);
    out_.SetGlobalBuffer((__gm__ bfloat16_t*)out);
    status_.SetGlobalBuffer((__gm__ int32_t*)status);
    batch_ = batch;
    numSlots_ = numSlots;
    heads_ = heads;
    valueHeads_ = valueHeads;
    capacity_ = capacity;
    checkpointSlotStride_ = checkpointSlotStride;
    dCacheSlotStride_ = dCacheSlotStride;
    kCacheSlotStride_ = kCacheSlotStride;
    gCacheSlotStride_ = gCacheSlotStride;
    tileV_ = tileV;
    scale_ = scale;
    tiles_ = V_DIM / tileV;
    // Reuse the same input/output staging queues across nonoverlapping
    // lifetimes. All sizes/offsets are multiples of the 32-byte DMA block.
    uint32_t inElements = capacity * K_DIM;
    if (inElements < tileV * K_DIM) {
      inElements = tileV * K_DIM;
    }
    pipe_.InitBuffer(in_, 1, inElements * sizeof(half));
    pipe_.InitBuffer(outQueue_, 1, tileV * K_DIM * sizeof(bfloat16_t));
    pipe_.InitBuffer(stateBuffer_, tileV * K_DIM * sizeof(float));
    pipe_.InitBuffer(productBuffer_, tileV * K_DIM * sizeof(float));
    pipe_.InitBuffer(historyKBuffer_, capacity * K_DIM * sizeof(float));
    pipe_.InitBuffer(historyDBuffer_, capacity * tileV * sizeof(float));
    pipe_.InitBuffer(historyGBuffer_, MAX_HISTORY * sizeof(float));
    pipe_.InitBuffer(vectorBuffer_, (2 * K_DIM + 3 * tileV + FP32_BLOCK) * sizeof(float));
    pipe_.InitBuffer(reduceBuffer_, 4096);
    state_ = stateBuffer_.Get<float>();
    product_ = productBuffer_.Get<float>();
    historyK_ = historyKBuffer_.Get<float>();
    historyD_ = historyDBuffer_.Get<float>();
    historyG_ = historyGBuffer_.Get<float>();
    auto vectors = vectorBuffer_.Get<float>();
    qLocal_ = vectors;
    kLocal_ = vectors[K_DIM];
    value_ = vectors[2 * K_DIM];
    delta_ = vectors[2 * K_DIM + tileV];
    result_ = vectors[2 * K_DIM + 2 * tileV];
    scalar_ = vectors[2 * K_DIM + 3 * tileV];
  }

  __aicore__ inline void Process() {
    for (uint32_t task = GetBlockIdx(); task < batch_ * valueHeads_ * tiles_; task += GetBlockNum()) {
      uint32_t row = task / (valueHeads_ * tiles_);
      uint32_t head = (task / tiles_) % valueHeads_;
      uint32_t vStart = (task % tiles_) * tileV_;
      uint32_t keyHead = head / (valueHeads_ / heads_);
      int32_t slot = ReadInt(slots_, row);
      int32_t pos = ReadInt(positions_, row);
      int32_t flush = ReadInt(flush_, row);
      int32_t error = 0;
      if (slot < 0 || slot >= static_cast<int32_t>(numSlots_)) {
        error = 1;
      } else if (pos < 0 || pos >= static_cast<int32_t>(capacity_)) {
        error = 2;
      } else if ((flush != 0 && flush != 1) || (slot > 0 && pos == static_cast<int32_t>(capacity_ - 1) && !flush)) {
        error = 3;
      } else if (slot == 0 && (pos != 0 || flush != 0)) {
        error = 4;
      }
      uint64_t outputOffset = (static_cast<uint64_t>(row) * valueHeads_ + head) * V_DIM + vStart;
      if (error != 0 || slot == 0) {
        Duplicate(result_, 0.0f, tileV_);
        PipeBarrier<PIPE_V>();
        Store(out_, outputOffset, result_, tileV_);
        WriteStatus(task, error);
        continue;
      }
      uint64_t stateOffset = static_cast<uint64_t>(slot) * checkpointSlotStride_ +
                             static_cast<uint64_t>(head) * V_DIM * K_DIM + vStart * K_DIM;
      uint64_t dOffset = static_cast<uint64_t>(slot) * dCacheSlotStride_ +
                         static_cast<uint64_t>(head) * capacity_ * V_DIM;
      uint64_t kOffset = static_cast<uint64_t>(slot) * kCacheSlotStride_ +
                         static_cast<uint64_t>(keyHead) * capacity_ * K_DIM;
      uint64_t gOffset = static_cast<uint64_t>(slot) * gCacheSlotStride_ +
                         static_cast<uint64_t>(head) * capacity_;
      Load(checkpoint_, stateOffset, state_, tileV_ * K_DIM);
      Load(q_, (static_cast<uint64_t>(row) * heads_ + keyHead) * K_DIM, qLocal_, K_DIM);
      Load(k_, (static_cast<uint64_t>(row) * heads_ + keyHead) * K_DIM, kLocal_, K_DIM);
      Load(v_, outputOffset, value_, tileV_);
      Muls(qLocal_, qLocal_, scale_, K_DIM);
      PipeBarrier<PIPE_V>();
      if (pos > 0) {
        // Read only committed rows. Poisoned/uninitialized cache tails
        // are never loaded; appending at pos cannot alias this prefix.
        Load(kCache_, kOffset, historyK_, pos * K_DIM);
        LoadHistoryD(dOffset + vStart, pos);
        Load(gCache_, gOffset, historyG_, pos);
        Exp(historyG_, historyG_, pos);
        PipeBarrier<PIPE_V>();
        for (int32_t i = 0; i < pos; ++i) {
          float alpha = ReadLocal(historyG_, i);
          Muls(state_, state_, alpha, tileV_ * K_DIM);
          PipeBarrier<PIPE_V>();
          OuterAdd(historyD_[i * tileV_], historyK_[i * K_DIM]);
        }
      }
      Load(g_, static_cast<uint64_t>(row) * valueHeads_ + head, scalar_, 1);
      float logGate = ReadLocal(scalar_, 0);
      Exp(scalar_, scalar_, 1);
      PipeBarrier<PIPE_V>();
      float alpha = ReadLocal(scalar_, 0);
      Load(beta_, static_cast<uint64_t>(row) * valueHeads_ + head, scalar_, 1);
      float beta = ReadLocal(scalar_, 0);
      Muls(state_, state_, alpha, tileV_ * K_DIM);
      PipeBarrier<PIPE_V>();
      MatVec(kLocal_, delta_);
      Sub(delta_, value_, delta_, tileV_);
      PipeBarrier<PIPE_V>();
      Muls(delta_, delta_, beta, tileV_);
      PipeBarrier<PIPE_V>();
      OuterAdd(delta_, kLocal_);
      MatVec(qLocal_, result_);
      Store(out_, outputOffset, result_, tileV_);
      if (flush) {
        Store(checkpoint_, stateOffset, state_, tileV_ * K_DIM);
      } else {
        Store(dCache_, dOffset + static_cast<uint64_t>(pos) * V_DIM + vStart, delta_, tileV_);
        if (vStart == 0) {
          WriteGate(gOffset + pos, logGate);
          if (head % (valueHeads_ / heads_) == 0) {
            Store(kCache_, kOffset + static_cast<uint64_t>(pos) * K_DIM, kLocal_, K_DIM);
          }
        }
      }
      WriteStatus(task, 0);
    }
  }

 private:
  template <typename T>
  __aicore__ inline void Load(GlobalTensor<T>& source, uint64_t offset, LocalTensor<float> dest, uint32_t count) {
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

  __aicore__ inline void LoadHistoryD(uint64_t offset, uint32_t rows) {
    auto input = in_.AllocTensor<half>();
    DataCopyExtParams cp{static_cast<uint16_t>(rows), static_cast<uint32_t>(tileV_ * sizeof(half)),
                         static_cast<uint32_t>((V_DIM - tileV_) * sizeof(half)), 0, 0};
    DataCopyPadExtParams<half> pad{false, 0, 0, 0};
    DataCopyPad(input, dCache_[offset], cp, pad);
    in_.EnQue(input);
    input = in_.DeQue<half>();
    Cast(historyD_, input, RoundMode::CAST_NONE, rows * tileV_);
    PipeBarrier<PIPE_V>();
    in_.FreeTensor(input);
  }

  // All scalar GM traffic goes through 32-byte UB staging; no unbounded GM
  // GetValue/SetValue or rounded-up GM reads past the allocation are used.
  __aicore__ inline int32_t ReadInt(GlobalTensor<int32_t>& source, uint32_t offset) {
    auto input = in_.AllocTensor<int32_t>();
    DataCopyExtParams cp{1, sizeof(int32_t), 0, 0, 0};
    DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    DataCopyPad(input, source[offset], cp, pad);
    in_.EnQue(input);
    input = in_.DeQue<int32_t>();
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    int32_t value = input.GetValue(0);
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

  __aicore__ inline void OuterAdd(LocalTensor<float> d, LocalTensor<float> k) {
    uint32_t shape[2] = {tileV_, K_DIM};
    uint32_t deltaShape[2] = {tileV_, 1};
    Broadcast<float, 2, 1>(product_, d, shape, deltaShape);
    PipeBarrier<PIPE_V>();
    Mul(product_, product_, k, 64, tileV_, {1, 1, 1, 16, 16, 0});
    Mul(product_[64], product_[64], k[64], 64, tileV_, {1, 1, 1, 16, 16, 0});
    PipeBarrier<PIPE_V>();
    Add(state_, state_, product_, tileV_ * K_DIM);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void MatVec(LocalTensor<float> vector, LocalTensor<float> dest) {
    // Native RGDR's row-broadcast Mul followed by a K reduction. K=128
    // uses two 64-lane repeats per V row, below the repeatTimes limit.
    Mul(product_, state_, vector, 64, tileV_, {1, 1, 1, 16, 16, 0});
    Mul(product_[64], state_[64], vector[64], 64, tileV_, {1, 1, 1, 16, 16, 0});
    PipeBarrier<PIPE_V>();
    uint32_t shape[2] = {tileV_, K_DIM};
    ReduceSum<float, Pattern::Reduce::AR, true>(dest, product_, reduceBuffer_.Get<uint8_t>(), shape, true);
    PipeBarrier<PIPE_V>();
  }

  template <typename T>
  __aicore__ inline void Store(GlobalTensor<T>& dest, uint64_t offset, LocalTensor<float> source, uint32_t count) {
    auto output = outQueue_.AllocTensor<T>();
    Cast(output, source, RoundMode::CAST_RINT, count);
    outQueue_.EnQue(output);
    output = outQueue_.DeQue<T>();
    DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPad(dest[offset], output, cp);
    outQueue_.FreeTensor(output);
  }

  __aicore__ inline void WriteGate(uint64_t offset, float gate) {
    auto output = outQueue_.AllocTensor<float>();
    output.SetValue(0, gate);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    outQueue_.EnQue(output);
    output = outQueue_.DeQue<float>();
    DataCopyExtParams cp{1, sizeof(float), 0, 0, 0};
    DataCopyPad(gCache_[offset], output, cp);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    outQueue_.FreeTensor(output);
  }

  __aicore__ inline void WriteStatus(uint32_t task, int32_t error) {
    auto output = outQueue_.AllocTensor<int32_t>();
    output.SetValue(0, error);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    outQueue_.EnQue(output);
    output = outQueue_.DeQue<int32_t>();
    DataCopyExtParams cp{1, sizeof(int32_t), 0, 0, 0};
    DataCopyPad(status_[task], output, cp);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    outQueue_.FreeTensor(output);
  }

  TPipe pipe_;
  TQue<QuePosition::VECIN, 1> in_;
  TQue<QuePosition::VECOUT, 1> outQueue_;
  TBuf<QuePosition::VECCALC> stateBuffer_, productBuffer_, historyKBuffer_, historyDBuffer_;
  TBuf<QuePosition::VECCALC> historyGBuffer_, vectorBuffer_, reduceBuffer_;
  LocalTensor<float> state_, product_, historyK_, historyD_, historyG_;
  LocalTensor<float> qLocal_, kLocal_, value_, delta_, result_, scalar_;
  GlobalTensor<bfloat16_t> q_, k_, v_, checkpoint_, out_;
  GlobalTensor<half> dCache_, kCache_;
  GlobalTensor<float> g_, beta_, gCache_;
  GlobalTensor<int32_t> slots_, positions_, flush_, status_;
  uint32_t batch_, numSlots_, heads_, valueHeads_, capacity_, tileV_, tiles_;
  uint64_t checkpointSlotStride_, dCacheSlotStride_, kCacheSlotStride_, gCacheSlotStride_;
  float scale_;
};
}  // namespace

extern "C" __global__ __aicore__ void replayssm_ar(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
                                                   GM_ADDR checkpoint, GM_ADDR dCache, GM_ADDR kCache, GM_ADDR gCache,
                                                   GM_ADDR slots, GM_ADDR positions, GM_ADDR flush, GM_ADDR out,
                                                   GM_ADDR status, uint32_t batch, uint32_t numSlots, uint32_t heads,
                                                   uint32_t valueHeads, uint32_t capacity,
                                                   uint64_t checkpointSlotStride, uint64_t dCacheSlotStride,
                                                   uint64_t kCacheSlotStride, uint64_t gCacheSlotStride,
                                                   uint32_t tileV, float scale) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  ReplayAR op;
  op.Init(q, k, v, g, beta, checkpoint, dCache, kCache, gCache, slots, positions, flush, out, status, batch, numSlots,
          heads, valueHeads, capacity, checkpointSlotStride, dCacheSlotStride, kCacheSlotStride, gCacheSlotStride,
          tileV, scale);
  op.Process();
}

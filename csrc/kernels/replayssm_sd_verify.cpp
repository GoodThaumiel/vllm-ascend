// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t K_DIM = 128;
constexpr uint32_t V_DIM = 128;
constexpr uint32_t MAX_HISTORY = 16;
constexpr uint32_t MAX_TOKENS = 4;
constexpr uint32_t MAX_WINDOW = MAX_HISTORY + MAX_TOKENS;
constexpr uint32_t FP32_BLOCK = 8;

// L3.1 non-flush pilot: one AIV task owns (request, value head, V tile).
// No intermediate recurrent state is formed or written. The triangular GDN
// correction solve is evaluated in token order with the earlier corrections
// held in UB; this is algebraically the same unit-triangular solve as Eq. 10
// of the paper, but is not yet the final performance tiling.
class ReplaySDVerify {
 public:
  __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR checkpoint,
                              GM_ADDR dCache, GM_ADDR kCache, GM_ADDR gCache, GM_ADDR slots, GM_ADDR positions,
                              GM_ADDR bases, GM_ADDR lengths, GM_ADDR out, GM_ADDR status, uint32_t batch,
                              uint32_t numSlots, uint32_t heads, uint32_t valueHeads, uint32_t ringLen,
                              uint64_t checkpointSlotStride, uint64_t dSlotStride, uint64_t kSlotStride,
                              uint64_t gSlotStride, uint32_t tileV, float scale) {
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
    bases_.SetGlobalBuffer((__gm__ int32_t*)bases);
    lengths_.SetGlobalBuffer((__gm__ int32_t*)lengths);
    out_.SetGlobalBuffer((__gm__ bfloat16_t*)out);
    status_.SetGlobalBuffer((__gm__ int32_t*)status);
    batch_ = batch;
    numSlots_ = numSlots;
    heads_ = heads;
    valueHeads_ = valueHeads;
    ringLen_ = ringLen;
    checkpointStride_ = checkpointSlotStride;
    dStride_ = dSlotStride;
    kStride_ = kSlotStride;
    gStride_ = gSlotStride;
    tileV_ = tileV;
    tiles_ = V_DIM / tileV;
    scale_ = scale;

    pipe_.InitBuffer(in_, 1, tileV * K_DIM * sizeof(bfloat16_t));
    // A key-cache store casts all 128 elements to FP16 in this queue.
    pipe_.InitBuffer(outQueue_, 1, K_DIM * sizeof(float));
    pipe_.InitBuffer(checkpointBuffer_, tileV * K_DIM * sizeof(float));
    pipe_.InitBuffer(productBuffer_, tileV * K_DIM * sizeof(float));
    pipe_.InitBuffer(historyKBuffer_, MAX_HISTORY * K_DIM * sizeof(float));
    pipe_.InitBuffer(historyDBuffer_, MAX_HISTORY * tileV * sizeof(float));
    pipe_.InitBuffer(specKBuffer_, MAX_TOKENS * K_DIM * sizeof(float));
    pipe_.InitBuffer(specDBuffer_, MAX_TOKENS * tileV * sizeof(float));
    pipe_.InitBuffer(vectorBuffer_, (3 * K_DIM + 7 * tileV + FP32_BLOCK) * sizeof(float));
    pipe_.InitBuffer(reduceBuffer_, 4096);

    state_ = checkpointBuffer_.Get<float>();
    product_ = productBuffer_.Get<float>();
    historyK_ = historyKBuffer_.Get<float>();
    historyD_ = historyDBuffer_.Get<float>();
    specK_ = specKBuffer_.Get<float>();
    specD_ = specDBuffer_.Get<float>();
    auto vec = vectorBuffer_.Get<float>();
    qLocal_ = vec;
    kLocal_ = vec[K_DIM];
    value_ = vec[3 * K_DIM];
    prevQ_ = vec[3 * K_DIM + tileV];
    prevK_ = vec[3 * K_DIM + 2 * tileV];
    delta_ = vec[3 * K_DIM + 3 * tileV];
    result_ = vec[3 * K_DIM + 4 * tileV];
    term_ = vec[3 * K_DIM + 5 * tileV];
    scalar_ = vec[3 * K_DIM + 7 * tileV];
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
      const int32_t length = ReadInt(lengths_, row);
      int32_t error = 0;
      if (slot < 0 || slot >= static_cast<int32_t>(numSlots_)) {
        error = 1;
      } else if (slot == 0) {
        if (pos != 0 || base != 0 || length != 0) {
          error = 4;
        }
      } else if (pos < 0 || pos > static_cast<int32_t>(MAX_HISTORY) || length < 1 ||
                 length > static_cast<int32_t>(MAX_TOKENS) || pos + length > static_cast<int32_t>(MAX_WINDOW) ||
                 pos + length > static_cast<int32_t>(ringLen_)) {
        error = 2;  // history plus speculative tokens must fit the physical ring
      } else if (base < 0 || base >= static_cast<int32_t>(ringLen_)) {
        error = 3;
      }
      if (error != 0 || slot == 0) {
        Duplicate(result_, 0.0f, tileV_);
        PipeBarrier<PIPE_V>();
        for (uint32_t s = 0; s < MAX_TOKENS; ++s) {
          Store(out_, OutputOffset(row, s, head, vStart), result_, tileV_);
        }
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
      float historyAlpha[MAX_HISTORY];
      float specAlpha[MAX_TOKENS];
      float historyProduct = 1.0f;
      for (int32_t i = 0; i < pos; ++i) {
        const uint32_t phys = Physical(base, i);
        Load(kCache_, kOffset + static_cast<uint64_t>(phys) * K_DIM, historyK_[i * K_DIM], K_DIM);
        Load(dCache_, dOffset + static_cast<uint64_t>(phys) * V_DIM + vStart, historyD_[i * tileV_], tileV_);
        Load(gCache_, gOffset + phys, scalar_, 1);
        Exp(scalar_, scalar_, 1);
        PipeBarrier<PIPE_V>();
        historyAlpha[i] = ReadLocal(scalar_, 0);
        historyProduct *= historyAlpha[i];
      }

      float specProduct = 1.0f;
      for (int32_t s = 0; s < length; ++s) {
        const uint64_t qkOffset = (static_cast<uint64_t>(row) * MAX_TOKENS + s) * heads_ * K_DIM +
                                  static_cast<uint64_t>(keyHead) * K_DIM;
        const uint64_t vOffset = OutputOffset(row, s, head, vStart);
        Load(q_, qkOffset, qLocal_, K_DIM);
        Load(k_, qkOffset, kLocal_, K_DIM);
        Load(v_, vOffset, value_, tileV_);
        Muls(qLocal_, qLocal_, scale_, K_DIM);
        PipeBarrier<PIPE_V>();
        MatVec(qLocal_, prevQ_);
        MatVec(kLocal_, prevK_);
        Muls(prevQ_, prevQ_, historyProduct * specProduct, tileV_);
        Muls(prevK_, prevK_, historyProduct * specProduct, tileV_);
        PipeBarrier<PIPE_V>();

        float suffix = specProduct;
        for (int32_t remaining = pos; remaining > 0; --remaining) {
          const int32_t i = remaining - 1;
          auto histK = historyK_[i * K_DIM];
          auto histD = historyD_[i * tileV_];
          Accumulate(prevQ_, histD, suffix * Dot(histK, qLocal_));
          Accumulate(prevK_, histD, suffix * Dot(histK, kLocal_));
          suffix *= historyAlpha[i];
        }
        suffix = 1.0f;
        for (int32_t remaining = s; remaining > 0; --remaining) {
          const int32_t j = remaining - 1;
          auto oldK = specK_[j * K_DIM];
          auto oldD = specD_[j * tileV_];
          Accumulate(prevQ_, oldD, suffix * Dot(oldK, qLocal_));
          Accumulate(prevK_, oldD, suffix * Dot(oldK, kLocal_));
          suffix *= specAlpha[j];
        }

        Load(g_, (static_cast<uint64_t>(row) * MAX_TOKENS + s) * valueHeads_ + head, scalar_, 1);
        const float logGate = ReadLocal(scalar_, 0);
        Exp(scalar_, scalar_, 1);
        PipeBarrier<PIPE_V>();
        const float alpha = ReadLocal(scalar_, 0);
        Load(beta_, (static_cast<uint64_t>(row) * MAX_TOKENS + s) * valueHeads_ + head, scalar_, 1);
        const float beta = ReadLocal(scalar_, 0);
        Muls(prevK_, prevK_, alpha, tileV_);
        PipeBarrier<PIPE_V>();
        Sub(delta_, value_, prevK_, tileV_);
        PipeBarrier<PIPE_V>();
        Muls(delta_, delta_, beta, tileV_);
        PipeBarrier<PIPE_V>();
        const float kq = Dot(kLocal_, qLocal_);
        Muls(prevQ_, prevQ_, alpha, tileV_);
        Muls(term_, delta_, kq, tileV_);
        PipeBarrier<PIPE_V>();
        Add(result_, prevQ_, term_, tileV_);
        PipeBarrier<PIPE_V>();
        Store(out_, vOffset, result_, tileV_);
        DataCopy(specD_[s * tileV_], delta_, tileV_);
        DataCopy(specK_[s * K_DIM], kLocal_, K_DIM);
        PipeBarrier<PIPE_V>();
        const uint32_t phys = Physical(base, pos + s);
        Store(dCache_, dOffset + static_cast<uint64_t>(phys) * V_DIM + vStart, delta_, tileV_);
        if (vStart == 0) {
          WriteGate(gOffset + phys, logGate);
          if (head % (valueHeads_ / heads_) == 0) {
            Store(kCache_, kOffset + static_cast<uint64_t>(phys) * K_DIM, kLocal_, K_DIM);
          }
        }
        specAlpha[s] = alpha;
        specProduct *= alpha;
      }
      Duplicate(result_, 0.0f, tileV_);
      PipeBarrier<PIPE_V>();
      for (int32_t s = length; s < static_cast<int32_t>(MAX_TOKENS); ++s) {
        Store(out_, OutputOffset(row, s, head, vStart), result_, tileV_);
      }
      WriteStatus(task, 0);
    }
  }

 private:
  __aicore__ inline uint32_t Physical(int32_t base, int32_t logical) const {
    return (base + logical) & (ringLen_ - 1);
  }

  __aicore__ inline uint64_t OutputOffset(uint32_t row, uint32_t token, uint32_t head, uint32_t vStart) const {
    return ((static_cast<uint64_t>(row) * MAX_TOKENS + token) * valueHeads_ + head) * V_DIM + vStart;
  }

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

  __aicore__ inline float Dot(LocalTensor<float> a, LocalTensor<float> b) {
    Mul(product_, a, b, K_DIM);
    PipeBarrier<PIPE_V>();
    uint32_t shape[2] = {1, K_DIM};
    ReduceSum<float, Pattern::Reduce::AR, true>(scalar_, product_, reduceBuffer_.Get<uint8_t>(), shape, true);
    PipeBarrier<PIPE_V>();
    return ReadLocal(scalar_, 0);
  }

  __aicore__ inline void MatVec(LocalTensor<float> vector, LocalTensor<float> dest) {
    Mul(product_, state_, vector, 64, tileV_, {1, 1, 1, 16, 16, 0});
    Mul(product_[64], state_[64], vector[64], 64, tileV_, {1, 1, 1, 16, 16, 0});
    PipeBarrier<PIPE_V>();
    uint32_t shape[2] = {tileV_, K_DIM};
    ReduceSum<float, Pattern::Reduce::AR, true>(dest, product_, reduceBuffer_.Get<uint8_t>(), shape, true);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void Accumulate(LocalTensor<float> dest, LocalTensor<float> d, float coefficient) {
    Muls(term_, d, coefficient, tileV_);
    PipeBarrier<PIPE_V>();
    Add(dest, dest, term_, tileV_);
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
  TBuf<QuePosition::VECCALC> checkpointBuffer_, productBuffer_, historyKBuffer_, historyDBuffer_;
  TBuf<QuePosition::VECCALC> specKBuffer_, specDBuffer_, vectorBuffer_, reduceBuffer_;
  LocalTensor<float> state_, product_, historyK_, historyD_, specK_, specD_;
  LocalTensor<float> qLocal_, kLocal_, value_, prevQ_, prevK_, delta_, result_, term_, scalar_;
  GlobalTensor<bfloat16_t> q_, k_, v_, checkpoint_, out_;
  GlobalTensor<half> dCache_, kCache_;
  GlobalTensor<float> g_, beta_, gCache_;
  GlobalTensor<int32_t> slots_, positions_, bases_, lengths_, status_;
  uint32_t batch_, numSlots_, heads_, valueHeads_, ringLen_, tileV_, tiles_;
  uint64_t checkpointStride_, dStride_, kStride_, gStride_;
  float scale_;
};
}  // namespace

extern "C" __global__ __aicore__ void replayssm_sd_verify(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR checkpoint, GM_ADDR dCache,
    GM_ADDR kCache, GM_ADDR gCache, GM_ADDR slots, GM_ADDR positions, GM_ADDR bases, GM_ADDR lengths,
    GM_ADDR out, GM_ADDR status, uint32_t batch, uint32_t numSlots, uint32_t heads, uint32_t valueHeads,
    uint32_t ringLen, uint64_t checkpointSlotStride, uint64_t dSlotStride, uint64_t kSlotStride,
    uint64_t gSlotStride, uint32_t tileV, float scale) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  ReplaySDVerify op;
  op.Init(q, k, v, g, beta, checkpoint, dCache, kCache, gCache, slots, positions, bases, lengths, out,
          status, batch, numSlots, heads, valueHeads, ringLen, checkpointSlotStride, dSlotStride,
          kSlotStride, gSlotStride, tileV, scale);
  op.Process();
}

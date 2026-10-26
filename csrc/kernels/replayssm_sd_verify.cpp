// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t K_DIM = 128;
constexpr uint32_t V_DIM = 128;
constexpr uint32_t MAX_HISTORY = 16;
constexpr uint32_t MAX_TOKENS = 4;
constexpr uint32_t SPEC_ROWS = 8;
constexpr uint32_t MAX_WINDOW = MAX_HISTORY + MAX_TOKENS;
constexpr uint32_t FP32_BLOCK = 8;
constexpr uint32_t MAX_RING = 64;
constexpr uint32_t MAX_VALUE_HEADS = 12;

// L3.1 non-flush pilot: the baseline AIV task owns one V tile; shared variants
// process all V tiles for a (request, value head) in one task.
// No intermediate recurrent state is formed or written. The triangular GDN
// correction solve is evaluated in token order with the earlier corrections
// held in UB; this is algebraically the same unit-triangular solve as Eq. 10
// of the paper, but is not yet the final performance tiling.
template <int VARIANT>
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
    pipe_.InitBuffer(specDBuffer_, (VARIANT == 8 ? SPEC_ROWS : MAX_TOKENS) * tileV * sizeof(float));
    pipe_.InitBuffer(vectorBuffer_, (3 * K_DIM + 7 * tileV + FP32_BLOCK) * sizeof(float));
    if constexpr (VARIANT == 0) {
      pipe_.InitBuffer(reduceBuffer_, 4096);
    }
    if constexpr (VARIANT >= 2) {
      pipe_.InitBuffer(historyGateBuffer_, MAX_RING * sizeof(float));
      pipe_.InitBuffer(candidateGateBuffer_, MAX_TOKENS * MAX_VALUE_HEADS * sizeof(float));
      pipe_.InitBuffer(candidateLogGateBuffer_, MAX_TOKENS * MAX_VALUE_HEADS * sizeof(float));
      pipe_.InitBuffer(candidateBetaBuffer_, MAX_TOKENS * MAX_VALUE_HEADS * sizeof(float));
      historyGate_ = historyGateBuffer_.Get<float>();
      candidateGate_ = candidateGateBuffer_.Get<float>();
      candidateLogGate_ = candidateLogGateBuffer_.Get<float>();
      candidateBeta_ = candidateBetaBuffer_.Get<float>();
    }
    if constexpr (VARIANT >= 3) {
      pipe_.InitBuffer(allQBuffer_, MAX_TOKENS * K_DIM * sizeof(float));
      allQ_ = allQBuffer_.Get<float>();
    }
    if constexpr (VARIANT == 11) {
      pipe_.InitBuffer(allVBuffer_, MAX_TOKENS * tileV_ * sizeof(float));
      allV_ = allVBuffer_.Get<float>();
    }
    if constexpr (VARIANT >= 5) {
      pipe_.InitBuffer(dotRowBuffer_, MAX_HISTORY * sizeof(float));
      dotRows_ = dotRowBuffer_.Get<float>();
    }
    if constexpr (VARIANT >= 7) {
      pipe_.InitBuffer(historyQDotBuffer_, MAX_TOKENS * MAX_HISTORY * sizeof(float));
      pipe_.InitBuffer(historyKDotBuffer_, MAX_TOKENS * MAX_HISTORY * sizeof(float));
      pipe_.InitBuffer(historySuffixBuffer_, MAX_HISTORY * sizeof(float));
      pipe_.InitBuffer(scaledSuffixBuffer_, MAX_HISTORY * sizeof(float));
      pipe_.InitBuffer(historyCoeffBuffer_, MAX_HISTORY * sizeof(float));
      pipe_.InitBuffer(expandedCoeffBuffer_, MAX_HISTORY * tileV_ * sizeof(float));
      pipe_.InitBuffer(weightedHistoryBuffer_, MAX_HISTORY * tileV_ * sizeof(float));
      historyQDot_ = historyQDotBuffer_.Get<float>();
      historyKDot_ = historyKDotBuffer_.Get<float>();
      historySuffix_ = historySuffixBuffer_.Get<float>();
      scaledSuffix_ = scaledSuffixBuffer_.Get<float>();
      historyCoeff_ = historyCoeffBuffer_.Get<float>();
      expandedCoeff_ = expandedCoeffBuffer_.Get<float>();
      weightedHistory_ = weightedHistoryBuffer_.Get<float>();
    }
    if constexpr (VARIANT == 8) {
      pipe_.InitBuffer(specQDotBuffer_, MAX_TOKENS * SPEC_ROWS * sizeof(float));
      pipe_.InitBuffer(specKDotBuffer_, MAX_TOKENS * SPEC_ROWS * sizeof(float));
      pipe_.InitBuffer(specSuffixBuffer_, MAX_TOKENS * SPEC_ROWS * sizeof(float));
      specQDot_ = specQDotBuffer_.Get<float>();
      specKDot_ = specKDotBuffer_.Get<float>();
      specSuffix_ = specSuffixBuffer_.Get<float>();
    }
    if constexpr (VARIANT == 4 || VARIANT == 6) {
      pipe_.InitBuffer(statePrefetchQueue_, 2, tileV_ * K_DIM * sizeof(bfloat16_t));
    }

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
    if constexpr (VARIANT >= 3) {
      ProcessSharedHeads();
      return;
    }
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
      if constexpr (VARIANT >= 2) {
        Load(gCache_, gOffset, historyGate_, ringLen_);
        Exp(historyGate_, historyGate_, ringLen_);
        PipeBarrier<PIPE_V>();
        SyncVectorToScalar();
        Load(g_, static_cast<uint64_t>(row) * MAX_TOKENS * valueHeads_, candidateLogGate_,
             MAX_TOKENS * valueHeads_);
        DataCopy(candidateGate_, candidateLogGate_, MAX_TOKENS * valueHeads_);
        PipeBarrier<PIPE_V>();
        Exp(candidateGate_, candidateGate_, MAX_TOKENS * valueHeads_);
        PipeBarrier<PIPE_V>();
        Load(beta_, static_cast<uint64_t>(row) * MAX_TOKENS * valueHeads_, candidateBeta_,
             MAX_TOKENS * valueHeads_);
        SyncVectorToScalar();
      }
      for (int32_t i = 0; i < pos; ++i) {
        const uint32_t phys = Physical(base, i);
        Load(kCache_, kOffset + static_cast<uint64_t>(phys) * K_DIM, historyK_[i * K_DIM], K_DIM);
        Load(dCache_, dOffset + static_cast<uint64_t>(phys) * V_DIM + vStart, historyD_[i * tileV_], tileV_);
        if constexpr (VARIANT >= 2) {
          historyAlpha[i] = historyGate_.GetValue(phys);
        } else {
          Load(gCache_, gOffset + phys, scalar_, 1);
          Exp(scalar_, scalar_, 1);
          PipeBarrier<PIPE_V>();
          historyAlpha[i] = ReadLocal(scalar_, 0);
        }
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

        float logGate;
        float alpha;
        float beta;
        if constexpr (VARIANT >= 2) {
          const uint32_t gateIndex = s * valueHeads_ + head;
          alpha = candidateGate_.GetValue(gateIndex);
          beta = candidateBeta_.GetValue(gateIndex);
          logGate = candidateLogGate_.GetValue(gateIndex);
        } else {
          Load(g_, (static_cast<uint64_t>(row) * MAX_TOKENS + s) * valueHeads_ + head, scalar_, 1);
          logGate = ReadLocal(scalar_, 0);
          Exp(scalar_, scalar_, 1);
          PipeBarrier<PIPE_V>();
          alpha = ReadLocal(scalar_, 0);
          Load(beta_, (static_cast<uint64_t>(row) * MAX_TOKENS + s) * valueHeads_ + head, scalar_, 1);
          beta = ReadLocal(scalar_, 0);
        }
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
  __aicore__ inline void ProcessSharedHeads() {
    // One task owns all V tiles of a (request, value head). The K-side
    // dot table is invariant across V tiles, so build it once per task.
    for (uint32_t task = GetBlockIdx(); task < batch_ * valueHeads_; task += GetBlockNum()) {
      const uint32_t row = task / valueHeads_;
      const uint32_t head = task % valueHeads_;
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
        error = 2;
      } else if (base < 0 || base >= static_cast<int32_t>(ringLen_)) {
        error = 3;
      }
      if (error != 0 || slot == 0) {
        Duplicate(result_, 0.0f, tileV_);
        PipeBarrier<PIPE_V>();
        for (uint32_t tile = 0; tile < tiles_; ++tile) {
          for (uint32_t s = 0; s < MAX_TOKENS; ++s) {
            Store(out_, OutputOffset(row, s, head, tile * tileV_), result_, tileV_);
          }
          WriteStatus(task * tiles_ + tile, error);
        }
        continue;
      }

      const uint64_t dOffset = static_cast<uint64_t>(slot) * dStride_ +
                               static_cast<uint64_t>(head) * ringLen_ * V_DIM;
      const uint64_t kOffset = static_cast<uint64_t>(slot) * kStride_ +
                               static_cast<uint64_t>(keyHead) * ringLen_ * K_DIM;
      const uint64_t gOffset = static_cast<uint64_t>(slot) * gStride_ +
                               static_cast<uint64_t>(head) * ringLen_;
      const uint64_t stateOffset = static_cast<uint64_t>(slot) * checkpointStride_ +
                                   static_cast<uint64_t>(head) * V_DIM * K_DIM;

      Load(gCache_, gOffset, historyGate_, ringLen_);
      Exp(historyGate_, historyGate_, ringLen_);
      PipeBarrier<PIPE_V>();
      Load(g_, static_cast<uint64_t>(row) * MAX_TOKENS * valueHeads_, candidateLogGate_,
           MAX_TOKENS * valueHeads_);
      DataCopy(candidateGate_, candidateLogGate_, MAX_TOKENS * valueHeads_);
      PipeBarrier<PIPE_V>();
      Exp(candidateGate_, candidateGate_, MAX_TOKENS * valueHeads_);
      PipeBarrier<PIPE_V>();
      Load(beta_, static_cast<uint64_t>(row) * MAX_TOKENS * valueHeads_, candidateBeta_,
           MAX_TOKENS * valueHeads_);
      SyncVectorToScalar();
      float historyAlpha[MAX_HISTORY];
      float alpha[MAX_TOKENS];
      float beta[MAX_TOKENS];
      float logGate[MAX_TOKENS];
      float historyProduct = 1.0f;
      for (int32_t i = 0; i < pos; ++i) {
        const uint32_t phys = Physical(base, i);
        historyAlpha[i] = historyGate_.GetValue(phys);
        historyProduct *= historyAlpha[i];
        if constexpr (VARIANT != 9 && VARIANT != 11) {
          Load(kCache_, kOffset + static_cast<uint64_t>(phys) * K_DIM, historyK_[i * K_DIM], K_DIM);
        }
      }
      if constexpr (VARIANT == 9 || VARIANT == 11) {
        // Logical history crosses the physical ring at most once. Two large
        // copies replace up to 16 per-record K copies and casts.
        const uint32_t available = ringLen_ - static_cast<uint32_t>(base);
        const uint32_t first = static_cast<uint32_t>(pos) < available ? static_cast<uint32_t>(pos) : available;
        if (first > 0) {
          Load(kCache_, kOffset + static_cast<uint64_t>(base) * K_DIM,
               historyK_, first * K_DIM);
        }
        if (static_cast<uint32_t>(pos) > first) {
          Load(kCache_, kOffset, historyK_[first * K_DIM],
               (static_cast<uint32_t>(pos) - first) * K_DIM);
        }
      }
      if constexpr (VARIANT >= 7) {
        // Suffix(i) = product of gates after history row i. It is shared by
        // every candidate token and V tile; only specProduct changes per token.
        float suffix = 1.0f;
        for (int32_t i = pos - 1; i >= 0; --i) {
          historySuffix_.SetValue(i, suffix);
          suffix *= historyAlpha[i];
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
      }
      for (int32_t s = 0; s < length; ++s) {
        const uint32_t gateIndex = s * valueHeads_ + head;
        alpha[s] = candidateGate_.GetValue(gateIndex);
        beta[s] = candidateBeta_.GetValue(gateIndex);
        logGate[s] = candidateLogGate_.GetValue(gateIndex);
        if constexpr (VARIANT != 11) {
          const uint64_t qkOffset = (static_cast<uint64_t>(row) * MAX_TOKENS + s) * heads_ * K_DIM +
                                    static_cast<uint64_t>(keyHead) * K_DIM;
          Load(q_, qkOffset, allQ_[s * K_DIM], K_DIM);
          Muls(allQ_[s * K_DIM], allQ_[s * K_DIM], scale_, K_DIM);
          PipeBarrier<PIPE_V>();
          Load(k_, qkOffset, specK_[s * K_DIM], K_DIM);
        }
      }
      if constexpr (VARIANT == 11) {
        // Q/K are [B, 4, H, 128]. Copy the active tokens as packed UB rows.
        const uint64_t qkOffset = static_cast<uint64_t>(row) * MAX_TOKENS * heads_ * K_DIM +
                                  static_cast<uint64_t>(keyHead) * K_DIM;
        LoadCandidateRows(q_, qkOffset, allQ_, static_cast<uint32_t>(length), K_DIM,
                          heads_ * K_DIM);
        LoadCandidateRows(k_, qkOffset, specK_, static_cast<uint32_t>(length), K_DIM,
                          heads_ * K_DIM);
        for (int32_t s = 0; s < length; ++s) {
          Muls(allQ_[s * K_DIM], allQ_[s * K_DIM], scale_, K_DIM);
          PipeBarrier<PIPE_V>();
        }
      }
      if constexpr (VARIANT == 8) {
        // Candidate gate suffixes are invariant across V tiles. Keep the
        // triangular correction coefficients in UB, including T=2/3 cases.
        for (int32_t s = 1; s < length; ++s) {
          float suffix = 1.0f;
          for (int32_t j = s - 1; j >= 0; --j) {
            specSuffix_.SetValue(s * SPEC_ROWS + j, suffix);
            suffix *= alpha[j];
          }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
      }

      if constexpr (VARIANT == 6) {
        // Keep the first checkpoint DMA in flight while computing the K-side
        // dot table, instead of enqueuing it after all dot work is finished.
        PrefetchCheckpoint(stateOffset);
        if (tiles_ > 1) {
          PrefetchCheckpoint(stateOffset + tileV_ * K_DIM);
        }
      }
      float historyQDot[MAX_TOKENS][MAX_HISTORY];
      float historyKDot[MAX_TOKENS][MAX_HISTORY];
      float specQDot[MAX_TOKENS][MAX_TOKENS];
      float specKDot[MAX_TOKENS][MAX_TOKENS];
      float currentKQ[MAX_TOKENS];
      for (int32_t s = 0; s < length; ++s) {
        auto qVec = allQ_[s * K_DIM];
        auto kVec = specK_[s * K_DIM];
        if constexpr (VARIANT >= 5) {
          // Reduce all history rows together. V7 keeps history results in UB;
          // earlier variants convert each batched result to Scalar once.
          if (pos > 0) {
            if constexpr (VARIANT >= 7) {
              DotRowsToUB(historyK_, qVec, historyQDot_[s * MAX_HISTORY], pos);
              DotRowsToUB(historyK_, kVec, historyKDot_[s * MAX_HISTORY], pos);
            } else {
              DotRows(historyK_, qVec, historyQDot[s], pos);
              DotRows(historyK_, kVec, historyKDot[s], pos);
            }
          }
          if constexpr (VARIANT == 8) {
            DotRowsToUB(specK_, qVec, specQDot_[s * SPEC_ROWS], s + 1);
            SyncVectorToScalar();
            currentKQ[s] = specQDot_.GetValue(s * SPEC_ROWS + s);
            if (s > 0) {
              DotRowsToUB(specK_, kVec, specKDot_[s * SPEC_ROWS], s);
            }
          } else {
            DotRows(specK_, qVec, specQDot[s], s + 1);
            currentKQ[s] = specQDot[s][s];
            if (s > 0) {
              DotRows(specK_, kVec, specKDot[s], s);
            }
          }
        } else {
          for (int32_t i = 0; i < pos; ++i) {
            auto oldK = historyK_[i * K_DIM];
            historyQDot[s][i] = Dot(oldK, qVec);
            historyKDot[s][i] = Dot(oldK, kVec);
          }
          for (int32_t j = 0; j < s; ++j) {
            auto oldK = specK_[j * K_DIM];
            specQDot[s][j] = Dot(oldK, qVec);
            specKDot[s][j] = Dot(oldK, kVec);
          }
          currentKQ[s] = Dot(kVec, qVec);
        }
      }

      if constexpr (VARIANT == 4) {
        PrefetchCheckpoint(stateOffset);
        if (tiles_ > 1) {
          PrefetchCheckpoint(stateOffset + tileV_ * K_DIM);
        }
      }
      for (uint32_t tile = 0; tile < tiles_; ++tile) {
        const uint32_t vStart = tile * tileV_;
        if constexpr (VARIANT == 4 || VARIANT == 6) {
          ConsumeCheckpoint();
          if (tile + 2 < tiles_) {
            PrefetchCheckpoint(stateOffset + (tile + 2) * tileV_ * K_DIM);
          }
        } else {
          Load(checkpoint_, stateOffset + vStart * K_DIM, state_, tileV_ * K_DIM);
        }
        if constexpr (VARIANT == 9 || VARIANT == 11) {
          const uint32_t available = ringLen_ - static_cast<uint32_t>(base);
          const uint32_t first = static_cast<uint32_t>(pos) < available ? static_cast<uint32_t>(pos) : available;
          if (first > 0) {
            LoadHistoryDRows(dOffset + static_cast<uint64_t>(base) * V_DIM + vStart,
                             historyD_, first);
          }
          if (static_cast<uint32_t>(pos) > first) {
            LoadHistoryDRows(dOffset + vStart, historyD_[first * tileV_],
                             static_cast<uint32_t>(pos) - first);
          }
        } else {
          for (int32_t i = 0; i < pos; ++i) {
            const uint32_t phys = Physical(base, i);
            Load(dCache_, dOffset + static_cast<uint64_t>(phys) * V_DIM + vStart,
                 historyD_[i * tileV_], tileV_);
          }
        }
        if constexpr (VARIANT >= 7) {
          // Broadcast requires an aligned leading dimension. Keep the
          // reduction at 16 rows, with absent history explicitly zeroed.
          if (pos < static_cast<int32_t>(MAX_HISTORY)) {
            Duplicate(historyD_[pos * tileV_], 0.0f,
                      (MAX_HISTORY - pos) * tileV_);
            PipeBarrier<PIPE_V>();
          }
        }
        if constexpr (VARIANT == 8) {
          // Eight rows satisfy Broadcast's aligned source dimension. Draft
          // rows not yet produced must be zero before the full-width product.
          Duplicate(specD_, 0.0f, SPEC_ROWS * tileV_);
          PipeBarrier<PIPE_V>();
        }
        if constexpr (VARIANT == 11) {
          // V is [B, 4, HV, 128]; each V tile has its own row offset.
          LoadCandidateRows(v_, OutputOffset(row, 0, head, vStart), allV_,
                            static_cast<uint32_t>(length), tileV_, valueHeads_ * V_DIM);
        }
        float specProduct = 1.0f;
        for (int32_t s = 0; s < length; ++s) {
          const uint64_t vOffset = OutputOffset(row, s, head, vStart);
          if constexpr (VARIANT != 11) {
            Load(v_, vOffset, value_, tileV_);
          }
          MatVec(allQ_[s * K_DIM], prevQ_);
          MatVec(specK_[s * K_DIM], prevK_);
          Muls(prevQ_, prevQ_, historyProduct * specProduct, tileV_);
          Muls(prevK_, prevK_, historyProduct * specProduct, tileV_);
          PipeBarrier<PIPE_V>();
          if constexpr (VARIANT >= 7) {
            if (pos > 0) {
              Muls(scaledSuffix_, historySuffix_, specProduct, pos);
              PipeBarrier<PIPE_V>();
              AccumulateHistory(prevQ_, historyQDot_[s * MAX_HISTORY], pos);
              AccumulateHistory(prevK_, historyKDot_[s * MAX_HISTORY], pos);
            }
          } else {
            float suffix = specProduct;
            for (int32_t remaining = pos; remaining > 0; --remaining) {
              const int32_t i = remaining - 1;
              auto oldD = historyD_[i * tileV_];
              Accumulate(prevQ_, oldD, suffix * historyQDot[s][i]);
              Accumulate(prevK_, oldD, suffix * historyKDot[s][i]);
              suffix *= historyAlpha[i];
            }
          }
          if constexpr (VARIANT == 8) {
            if (s > 0) {
              AccumulateSpec(prevQ_, specQDot_[s * SPEC_ROWS], specSuffix_[s * SPEC_ROWS], s);
              AccumulateSpec(prevK_, specKDot_[s * SPEC_ROWS], specSuffix_[s * SPEC_ROWS], s);
            }
          } else {
            float suffix = 1.0f;
            for (int32_t remaining = s; remaining > 0; --remaining) {
              const int32_t j = remaining - 1;
              auto oldD = specD_[j * tileV_];
              Accumulate(prevQ_, oldD, suffix * specQDot[s][j]);
              Accumulate(prevK_, oldD, suffix * specKDot[s][j]);
              suffix *= alpha[j];
            }
          }
          Muls(prevK_, prevK_, alpha[s], tileV_);
          PipeBarrier<PIPE_V>();
          if constexpr (VARIANT == 11) {
            Sub(delta_, allV_[s * tileV_], prevK_, tileV_);
          } else {
            Sub(delta_, value_, prevK_, tileV_);
          }
          PipeBarrier<PIPE_V>();
          Muls(delta_, delta_, beta[s], tileV_);
          PipeBarrier<PIPE_V>();
          Muls(prevQ_, prevQ_, alpha[s], tileV_);
          Muls(term_, delta_, currentKQ[s], tileV_);
          PipeBarrier<PIPE_V>();
          Add(result_, prevQ_, term_, tileV_);
          PipeBarrier<PIPE_V>();
          Store(out_, vOffset, result_, tileV_);
          DataCopy(specD_[s * tileV_], delta_, tileV_);
          PipeBarrier<PIPE_V>();
          const uint32_t phys = Physical(base, pos + s);
          Store(dCache_, dOffset + static_cast<uint64_t>(phys) * V_DIM + vStart, delta_, tileV_);
          if (vStart == 0) {
            WriteGate(gOffset + phys, logGate[s]);
            if (head % (valueHeads_ / heads_) == 0) {
              Store(kCache_, kOffset + static_cast<uint64_t>(phys) * K_DIM,
                    specK_[s * K_DIM], K_DIM);
            }
          }
          specProduct *= alpha[s];
        }
        Duplicate(result_, 0.0f, tileV_);
        PipeBarrier<PIPE_V>();
        for (int32_t s = length; s < static_cast<int32_t>(MAX_TOKENS); ++s) {
          Store(out_, OutputOffset(row, s, head, vStart), result_, tileV_);
        }
        WriteStatus(task * tiles_ + tile, 0);
      }
    }
  }

  __aicore__ inline void PrefetchCheckpoint(uint64_t offset) {
    auto input = statePrefetchQueue_.AllocTensor<bfloat16_t>();
    DataCopyExtParams cp{1, static_cast<uint32_t>(tileV_ * K_DIM * sizeof(bfloat16_t)), 0, 0, 0};
    DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, 0};
    DataCopyPad(input, checkpoint_[offset], cp, pad);
    statePrefetchQueue_.EnQue(input);
  }

  __aicore__ inline void ConsumeCheckpoint() {
    auto input = statePrefetchQueue_.DeQue<bfloat16_t>();
    Cast(state_, input, RoundMode::CAST_NONE, tileV_ * K_DIM);
    PipeBarrier<PIPE_V>();
    statePrefetchQueue_.FreeTensor(input);
  }

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

  __aicore__ inline void LoadHistoryDRows(uint64_t offset, LocalTensor<float> dest, uint32_t rows) {
    auto input = in_.AllocTensor<half>();
    // GM rows have V_DIM elements, while the UB tensor is packed by tileV_.
    // Both tileV_ and the physical row pitch are 32-byte aligned.
    DataCopyExtParams cp{static_cast<uint16_t>(rows),
                         static_cast<uint32_t>(tileV_ * sizeof(half)),
                         static_cast<uint32_t>((V_DIM - tileV_) * sizeof(half)), 0, 0};
    DataCopyPadExtParams<half> pad{false, 0, 0, 0};
    DataCopyPad(input, dCache_[offset], cp, pad);
    in_.EnQue(input);
    input = in_.DeQue<half>();
    Cast(dest, input, RoundMode::CAST_NONE, rows * tileV_);
    PipeBarrier<PIPE_V>();
    in_.FreeTensor(input);
  }

  __aicore__ inline void LoadCandidateRows(GlobalTensor<bfloat16_t>& source, uint64_t offset,
                                            LocalTensor<float> dest, uint32_t rows,
                                            uint32_t rowWidth, uint32_t rowPitch) {
    auto input = in_.AllocTensor<bfloat16_t>();
    // DataCopyExtParams blockLen/srcStride are bytes; the UB rows are packed.
    DataCopyExtParams cp{static_cast<uint16_t>(rows),
                         static_cast<uint32_t>(rowWidth * sizeof(bfloat16_t)),
                         static_cast<uint32_t>((rowPitch - rowWidth) * sizeof(bfloat16_t)), 0, 0};
    DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, 0};
    DataCopyPad(input, source[offset], cp, pad);
    in_.EnQue(input);
    input = in_.DeQue<bfloat16_t>();
    Cast(dest, input, RoundMode::CAST_NONE, rows * rowWidth);
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
    ReduceK128(scalar_, 1);
    PipeBarrier<PIPE_V>();
    return ReadLocal(scalar_, 0);
  }

  __aicore__ inline void DotRows(LocalTensor<float> keys, LocalTensor<float> vector,
                                 float* output, uint32_t rows) {
    DotRowsToUB(keys, vector, dotRows_, rows);
    SyncVectorToScalar();
    for (uint32_t i = 0; i < rows; ++i) {
      output[i] = dotRows_.GetValue(i);
    }
  }

  __aicore__ inline void DotRowsToUB(LocalTensor<float> keys, LocalTensor<float> vector,
                                     LocalTensor<float> dest, uint32_t rows) {
    Mul(product_, keys, vector, 64, rows, {1, 1, 1, 16, 16, 0});
    Mul(product_[64], keys[64], vector[64], 64, rows, {1, 1, 1, 16, 16, 0});
    PipeBarrier<PIPE_V>();
    ReduceK128(dest, rows);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void AccumulateHistory(LocalTensor<float> dest, LocalTensor<float> dots,
                                            uint32_t rows) {
    Duplicate(historyCoeff_, 0.0f, MAX_HISTORY);
    PipeBarrier<PIPE_V>();
    Mul(historyCoeff_, dots, scaledSuffix_, rows);
    PipeBarrier<PIPE_V>();
    uint32_t dstShape[2] = {MAX_HISTORY, tileV_};
    uint32_t srcShape[2] = {MAX_HISTORY, 1};
    Broadcast<float, 2, 1>(expandedCoeff_, historyCoeff_, dstShape, srcShape);
    PipeBarrier<PIPE_V>();
    Mul(weightedHistory_, historyD_, expandedCoeff_, MAX_HISTORY * tileV_);
    PipeBarrier<PIPE_V>();
    // Pairwise reduction across 16 history rows. Unused rows were zeroed;
    // each step updates disjoint destinations and halves the active rows.
    for (uint32_t step = 1; step < MAX_HISTORY; step *= 2) {
      const uint8_t repeats = MAX_HISTORY / (2 * step);
      const uint8_t rowStride = 2 * step * tileV_ / FP32_BLOCK;
      Add(weightedHistory_, weightedHistory_, weightedHistory_[step * tileV_],
          tileV_, repeats, {1, 1, 1, rowStride, rowStride, rowStride});
      PipeBarrier<PIPE_V>();
    }
    Add(dest, dest, weightedHistory_, tileV_);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void AccumulateSpec(LocalTensor<float> dest, LocalTensor<float> dots,
                                         LocalTensor<float> suffix, uint32_t rows) {
    Duplicate(historyCoeff_, 0.0f, SPEC_ROWS);
    PipeBarrier<PIPE_V>();
    Mul(historyCoeff_, dots, suffix, rows);
    PipeBarrier<PIPE_V>();
    uint32_t dstShape[2] = {SPEC_ROWS, tileV_};
    uint32_t srcShape[2] = {SPEC_ROWS, 1};
    Broadcast<float, 2, 1>(expandedCoeff_, historyCoeff_, dstShape, srcShape);
    PipeBarrier<PIPE_V>();
    Mul(weightedHistory_, specD_, expandedCoeff_, SPEC_ROWS * tileV_);
    PipeBarrier<PIPE_V>();
    for (uint32_t step = 1; step < SPEC_ROWS; step *= 2) {
      const uint8_t repeats = SPEC_ROWS / (2 * step);
      const uint8_t rowStride = 2 * step * tileV_ / FP32_BLOCK;
      Add(weightedHistory_, weightedHistory_, weightedHistory_[step * tileV_],
          tileV_, repeats, {1, 1, 1, rowStride, rowStride, rowStride});
      PipeBarrier<PIPE_V>();
    }
    Add(dest, dest, weightedHistory_, tileV_);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void MatVec(LocalTensor<float> vector, LocalTensor<float> dest) {
    Mul(product_, state_, vector, 64, tileV_, {1, 1, 1, 16, 16, 0});
    Mul(product_[64], state_[64], vector[64], 64, tileV_, {1, 1, 1, 16, 16, 0});
    PipeBarrier<PIPE_V>();
    ReduceK128(dest, tileV_);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void ReduceK128(LocalTensor<float> dest, uint32_t rows) {
    if constexpr (VARIANT == 0) {
      uint32_t shape[2] = {rows, K_DIM};
      ReduceSum<float, Pattern::Reduce::AR, true>(dest, product_, reduceBuffer_.Get<uint8_t>(), shape, true);
    } else {
      // Match the 910B native RGDR K=128 path: fold 128 to 64 in place,
      // then reduce all rows with one WholeReduceSum call.
      Add(product_[64], product_, product_[64], 64, rows, {1, 1, 1, 16, 16, 16});
      PipeBarrier<PIPE_V>();
      WholeReduceSum(dest, product_[64], 64, rows, 1, 1, 16);
    }
  }

  __aicore__ inline void SyncVectorToScalar() {
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
  }

  __aicore__ inline void Accumulate(LocalTensor<float> dest, LocalTensor<float> d, float coefficient) {
    if constexpr (VARIANT == 10) {
      // MulAddDst is tensor*tensor+dst, so materialize the scalar as one UB
      // vector. This isolates the fused arithmetic from the V8b/C changes.
      Duplicate(term_, coefficient, tileV_);
      PipeBarrier<PIPE_V>();
      MulAddDst(dest, d, term_, tileV_);
      PipeBarrier<PIPE_V>();
    } else {
      Muls(term_, d, coefficient, tileV_);
      PipeBarrier<PIPE_V>();
      Add(dest, dest, term_, tileV_);
      PipeBarrier<PIPE_V>();
    }
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
  TQue<QuePosition::VECIN, 2> statePrefetchQueue_;
  TQue<QuePosition::VECOUT, 1> outQueue_;
  TBuf<QuePosition::VECCALC> checkpointBuffer_, productBuffer_, historyKBuffer_, historyDBuffer_;
  TBuf<QuePosition::VECCALC> specKBuffer_, specDBuffer_, vectorBuffer_, reduceBuffer_;
  TBuf<QuePosition::VECCALC> historyGateBuffer_, candidateGateBuffer_, candidateLogGateBuffer_;
  TBuf<QuePosition::VECCALC> candidateBetaBuffer_, allQBuffer_, allVBuffer_, dotRowBuffer_;
  TBuf<QuePosition::VECCALC> historyQDotBuffer_, historyKDotBuffer_, historySuffixBuffer_;
  TBuf<QuePosition::VECCALC> scaledSuffixBuffer_, historyCoeffBuffer_, expandedCoeffBuffer_, weightedHistoryBuffer_;
  TBuf<QuePosition::VECCALC> specQDotBuffer_, specKDotBuffer_, specSuffixBuffer_;
  LocalTensor<float> state_, product_, historyK_, historyD_, specK_, specD_;
  LocalTensor<float> historyGate_, candidateGate_, candidateLogGate_, candidateBeta_, allQ_, allV_;
  LocalTensor<float> dotRows_;
  LocalTensor<float> historyQDot_, historyKDot_, historySuffix_, scaledSuffix_, historyCoeff_;
  LocalTensor<float> expandedCoeff_, weightedHistory_;
  LocalTensor<float> specQDot_, specKDot_, specSuffix_;
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

#ifndef REPLAY_SD_NO_ENTRY
extern "C" __global__ __aicore__ void replayssm_sd_verify(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR checkpoint, GM_ADDR dCache,
    GM_ADDR kCache, GM_ADDR gCache, GM_ADDR slots, GM_ADDR positions, GM_ADDR bases, GM_ADDR lengths,
    GM_ADDR out, GM_ADDR status, uint32_t batch, uint32_t numSlots, uint32_t heads, uint32_t valueHeads,
    uint32_t ringLen, uint64_t checkpointSlotStride, uint64_t dSlotStride, uint64_t kSlotStride,
    uint64_t gSlotStride, uint32_t tileV, float scale) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  ReplaySDVerify<0> op;
  op.Init(q, k, v, g, beta, checkpoint, dCache, kCache, gCache, slots, positions, bases, lengths, out,
          status, batch, numSlots, heads, valueHeads, ringLen, checkpointSlotStride, dSlotStride,
          kSlotStride, gSlotStride, tileV, scale);
  op.Process();
}
#endif

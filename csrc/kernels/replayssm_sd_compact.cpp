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
                              uint64_t dStride, uint64_t kStride, uint64_t gStride, uint32_t tileV,
                              uint32_t variant) {
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
    variant_ = variant;

    if (variant >= 3) {
      // Metadata and status remain live on the no-flush path. The large
      // checkpoint queues and vector tiles are initialized on first Fold.
      pipe_.InitBuffer(metaIn_, 1, 32);
      pipe_.InitBuffer(statusOut_, 1,
                       static_cast<uint32_t>(valueHeads_ * tiles_ * sizeof(int32_t)));
      heavyReady_ = false;
    } else {
      InitHeavyBuffers();
      heavyReady_ = true;
    }
  }

  __aicore__ inline void InitHeavyBuffers() {
    // At tile_v=32: two 16-KiB state/product tiles, 8-KiB input,
    // 8-KiB output plus <1 KiB vectors. No per-history materialization.
    pipe_.InitBuffer(in_, 1, tileV_ * K_DIM * sizeof(bfloat16_t));
    pipe_.InitBuffer(out_, 1, tileV_ * K_DIM * sizeof(bfloat16_t));
    pipe_.InitBuffer(stateBuffer_, tileV_ * K_DIM * sizeof(float));
    pipe_.InitBuffer(productBuffer_, tileV_ * K_DIM * sizeof(float));
    pipe_.InitBuffer(vectorBuffer_, (K_DIM + tileV_ + FP32_BLOCK) * sizeof(float));
    state_ = stateBuffer_.Get<float>();
    product_ = productBuffer_.Get<float>();
    auto vec = vectorBuffer_.Get<float>();
    key_ = vec;
    delta_ = vec[K_DIM];
    scalar_ = vec[K_DIM + tileV_];
  }

  __aicore__ inline void Process(uint32_t variant) {
    if (variant == 1) {
      ProcessGrouped();
    } else if (variant == 2) {
      ProcessHybrid();
    } else if (variant >= 3) {
      ProcessGroupedLight();
    } else {
      ProcessOriginal();
    }
  }

 private:
  __aicore__ inline int32_t Validate(int32_t slot, int32_t pos, int32_t base, int32_t flush) {
    if (slot < 0 || slot >= static_cast<int32_t>(numSlots_)) {
      return 1;
    }
    if (slot == 0) {
      return (pos != 0 || base != 0 || flush != 0) ? 4 : 0;
    }
    if (pos < 0 || pos > static_cast<int32_t>(MAX_HISTORY)) {
      return 2;
    }
    if (base < 0 || base >= static_cast<int32_t>(ringLen_)) {
      return 3;
    }
    return (flush != 0 && flush != 1) ? 5 : 0;
  }

  __aicore__ inline void ProcessOriginal() {
    for (uint32_t task = GetBlockIdx(); task < batch_ * valueHeads_ * tiles_; task += GetBlockNum()) {
      const uint32_t row = task / (valueHeads_ * tiles_);
      const int32_t slot = ReadInt(slots_, row);
      const int32_t pos = ReadInt(positions_, row);
      const int32_t base = ReadInt(bases_, row);
      const int32_t flush = ReadInt(flush_, row);
      const int32_t error = Validate(slot, pos, base, flush);
      if (error != 0 || slot == 0 || flush == 0 || pos == 0) {
        WriteStatus(task, error);
        continue;
      }
      Fold(task, slot, pos, base);
    }
  }

  // At high batch, one task owns all V tiles for a (row, value head).
  // The no-flush/status-only path writes the entire row from head zero.
  // This keeps the graph launch fixed while reducing repeated metadata and
  // four-byte status transfers; no host readback of the device flush flag.
  __aicore__ inline void ProcessGrouped() {
    const uint32_t rowTasks = valueHeads_ * tiles_;
    for (uint32_t group = GetBlockIdx(); group < batch_ * valueHeads_; group += GetBlockNum()) {
      const uint32_t row = group / valueHeads_;
      const uint32_t head = group % valueHeads_;
      const int32_t flush = ReadInt(flush_, row);
      if (flush == 0 && head != 0) {
        continue;
      }
      const int32_t slot = ReadInt(slots_, row);
      const int32_t pos = ReadInt(positions_, row);
      const int32_t base = ReadInt(bases_, row);
      const int32_t error = Validate(slot, pos, base, flush);
      if (error != 0 || slot == 0 || flush == 0 || pos == 0) {
        if (head == 0) {
          WriteStatusSpan(row * rowTasks, rowTasks, error);
        }
        continue;
      }
      for (uint32_t tile = 0; tile < tiles_; ++tile) {
        Fold(group * tiles_ + tile, slot, pos, base);
      }
    }
  }

  struct Metadata {
    int32_t slot;
    int32_t pos;
    int32_t base;
    int32_t flush;
  };

  // v3 isolates lazy UB setup from v1; v4 additionally reads a pre-packed
  // [B,4] metadata allocation. Both keep the v1 grouped task mapping.
  __aicore__ inline void ProcessGroupedLight() {
    const uint32_t rowTasks = valueHeads_ * tiles_;
    for (uint32_t group = GetBlockIdx(); group < batch_ * valueHeads_; group += GetBlockNum()) {
      const uint32_t row = group / valueHeads_;
      const uint32_t head = group % valueHeads_;
      Metadata md;
      if (variant_ == 4) {
        if (head != 0 && ReadIntLight(flush_, row * 4) == 0) {
          continue;
        }
        md = ReadPackedMetadata(row);
      } else {
        md.flush = ReadIntLight(flush_, row);
        if (md.flush == 0 && head != 0) {
          continue;
        }
        md.slot = ReadIntLight(slots_, row);
        md.pos = ReadIntLight(positions_, row);
        md.base = ReadIntLight(bases_, row);
      }
      const int32_t error = Validate(md.slot, md.pos, md.base, md.flush);
      if (error != 0 || md.slot == 0 || md.flush == 0 || md.pos == 0) {
        if (head == 0) {
          WriteStatusSpanLight(row * rowTasks, rowTasks, error);
        }
        continue;
      }
      if (!heavyReady_) {
        InitHeavyBuffers();
        heavyReady_ = true;
      }
      for (uint32_t tile = 0; tile < tiles_; ++tile) {
        Fold(group * tiles_ + tile, md.slot, md.pos, md.base);
      }
    }
  }

  // Preserve the original tile-level flush parallelism while coalescing the
  // no-flush status update into one aligned row write.
  __aicore__ inline void ProcessHybrid() {
    const uint32_t rowTasks = valueHeads_ * tiles_;
    for (uint32_t task = GetBlockIdx(); task < batch_ * rowTasks; task += GetBlockNum()) {
      const uint32_t row = task / rowTasks;
      const int32_t flush = ReadInt(flush_, row);
      if (flush == 0 && task % rowTasks != 0) {
        continue;
      }
      const int32_t slot = ReadInt(slots_, row);
      const int32_t pos = ReadInt(positions_, row);
      const int32_t base = ReadInt(bases_, row);
      const int32_t error = Validate(slot, pos, base, flush);
      if (flush == 0) {
        WriteStatusSpan(row * rowTasks, rowTasks, error);
      } else if (error != 0 || slot == 0 || pos == 0) {
        WriteStatus(task, error);
      } else {
        Fold(task, slot, pos, base);
      }
    }
  }

  __aicore__ inline void Fold(uint32_t task, int32_t slot, int32_t pos, int32_t base) {
    const uint32_t head = (task / tiles_) % valueHeads_;
    const uint32_t vStart = (task % tiles_) * tileV_;
    const uint32_t keyHead = head / (valueHeads_ / heads_);
    const uint64_t stateOffset =
        static_cast<uint64_t>(slot) * checkpointStride_ + static_cast<uint64_t>(head) * V_DIM * K_DIM + vStart * K_DIM;
    const uint64_t dOffset = static_cast<uint64_t>(slot) * dStride_ + static_cast<uint64_t>(head) * ringLen_ * V_DIM;
    const uint64_t kOffset = static_cast<uint64_t>(slot) * kStride_ + static_cast<uint64_t>(keyHead) * ringLen_ * K_DIM;
    const uint64_t gOffset = static_cast<uint64_t>(slot) * gStride_ + static_cast<uint64_t>(head) * ringLen_;
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

  __aicore__ inline void WriteStatusSpan(uint32_t offset, uint32_t count, int32_t error) {
    auto output = out_.AllocTensor<int32_t>();
    for (uint32_t i = 0; i < count; ++i) {
      output.SetValue(i, error);
    }
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    out_.EnQue(output);
    output = out_.DeQue<int32_t>();
    DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(status_[offset], output, cp);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    out_.FreeTensor(output);
  }

  __aicore__ inline void WriteStatusSpanLight(uint32_t offset, uint32_t count, int32_t error) {
    auto output = statusOut_.AllocTensor<int32_t>();
    for (uint32_t i = 0; i < count; ++i) {
      output.SetValue(i, error);
    }
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    statusOut_.EnQue(output);
    output = statusOut_.DeQue<int32_t>();
    DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(status_[offset], output, cp);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    statusOut_.FreeTensor(output);
  }

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

  __aicore__ inline int32_t ReadIntLight(GlobalTensor<int32_t>& source, uint32_t offset) {
    auto input = metaIn_.AllocTensor<int32_t>();
    DataCopyExtParams cp{1, sizeof(int32_t), 0, 0, 0};
    DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    DataCopyPad(input, source[offset], cp, pad);
    metaIn_.EnQue(input);
    input = metaIn_.DeQue<int32_t>();
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    const int32_t value = input.GetValue(0);
    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    metaIn_.FreeTensor(input);
    return value;
  }

  __aicore__ inline Metadata ReadPackedMetadata(uint32_t row) {
    auto input = metaIn_.AllocTensor<int32_t>();
    DataCopyExtParams cp{1, static_cast<uint32_t>(4 * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    DataCopyPad(input, slots_[row * 4], cp, pad);
    metaIn_.EnQue(input);
    input = metaIn_.DeQue<int32_t>();
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    Metadata md{input.GetValue(0), input.GetValue(1), input.GetValue(2), input.GetValue(3)};
    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    metaIn_.FreeTensor(input);
    return md;
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
  TQue<QuePosition::VECIN, 1> metaIn_;
  TQue<QuePosition::VECOUT, 1> statusOut_;
  TBuf<QuePosition::VECCALC> stateBuffer_, productBuffer_, vectorBuffer_;
  LocalTensor<float> state_, product_, key_, delta_, scalar_;
  GlobalTensor<bfloat16_t> checkpoint_;
  GlobalTensor<half> dCache_, kCache_;
  GlobalTensor<float> gCache_;
  GlobalTensor<int32_t> slots_, positions_, bases_, flush_, status_;
  uint32_t batch_, numSlots_, heads_, valueHeads_, ringLen_, tileV_, tiles_, variant_;
  bool heavyReady_;
  uint64_t checkpointStride_, dStride_, kStride_, gStride_;
};
}  // namespace

extern "C" __global__ __aicore__ void replayssm_sd_compact(
    GM_ADDR checkpoint, GM_ADDR dCache, GM_ADDR kCache, GM_ADDR gCache, GM_ADDR slots,
    GM_ADDR positions, GM_ADDR bases, GM_ADDR flush, GM_ADDR status, uint32_t batch,
    uint32_t numSlots, uint32_t heads, uint32_t valueHeads, uint32_t ringLen,
    uint64_t checkpointStride, uint64_t dStride, uint64_t kStride, uint64_t gStride,
    uint32_t tileV, uint32_t variant) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  ReplaySDCompact op;
  op.Init(checkpoint, dCache, kCache, gCache, slots, positions, bases, flush, status, batch,
          numSlots, heads, valueHeads, ringLen, checkpointStride, dStride, kStride, gStride, tileV,
          variant);
  op.Process(variant);
}

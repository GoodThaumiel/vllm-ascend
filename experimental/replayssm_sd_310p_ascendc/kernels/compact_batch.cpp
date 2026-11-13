// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t DIM = 128;
constexpr uint32_t RING = 32;
constexpr uint32_t HISTORY = 16;
constexpr uint32_t MATRIX = DIM * DIM;
constexpr uint32_t PAIR = 2;
constexpr uint32_t STATUS_WORDS = 8;  // Two heads, four ABI tiles per head.

template <HardEvent Event>
__aicore__ inline void Sync()
{
    const auto id = static_cast<event_t>(GetTPipePtr()->FetchEventID(Event));
    SetFlag<Event>(id);
    WaitFlag<Event>(id);
}

class ReplayCompactBatch {
public:
    __aicore__ inline void Init(
        GM_ADDR checkpoint, GM_ADDR dCache, GM_ADDR kCache, GM_ADDR gCache,
        GM_ADDR slots, GM_ADDR ringSlots, GM_ADDR positions, GM_ADDR bases, GM_ADDR flush,
        GM_ADDR status, uint32_t batch, uint32_t numSlots, uint32_t numRingSlots,
        uint32_t heads, uint64_t checkpointStride, uint64_t dStride, uint64_t kStride, uint64_t gStride)
    {
        checkpoint_.SetGlobalBuffer((__gm__ half*)checkpoint);
        dCache_.SetGlobalBuffer((__gm__ half*)dCache);
        kCache_.SetGlobalBuffer((__gm__ half*)kCache);
        gCache_.SetGlobalBuffer((__gm__ float*)gCache);
        slots_.SetGlobalBuffer((__gm__ int32_t*)slots);
        ringSlots_.SetGlobalBuffer((__gm__ int32_t*)ringSlots);
        positions_.SetGlobalBuffer((__gm__ int32_t*)positions);
        bases_.SetGlobalBuffer((__gm__ int32_t*)bases);
        flush_.SetGlobalBuffer((__gm__ int32_t*)flush);
        status_.SetGlobalBuffer((__gm__ int32_t*)status);
        batch_ = batch;
        numSlots_ = numSlots;
        numRingSlots_ = numRingSlots;
        heads_ = heads;
        checkpointStride_ = checkpointStride;
        dStride_ = dStride;
        kStride_ = kStride;
        gStride_ = gStride;
        pipe_.InitBuffer(statusBuffer_, heads_ * STATUS_WORDS * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t pair = GetBlockIdx(); pair < batch_ * heads_; pair += GetBlockNum()) {
            const uint32_t row = pair / heads_;
            const uint32_t keyHead = pair % heads_;
            const int32_t flush = flush_.GetValue(row);
            if (flush == 0 && keyHead != 0) {
                continue;
            }
            const int32_t slot = slots_.GetValue(row);
            const int32_t pos = positions_.GetValue(row);
            const int32_t base = bases_.GetValue(row);
            const int32_t ringSlot = flush == 0 ? 0 : ringSlots_.GetValue(row);
            const int32_t error = Validate(slot, ringSlot, pos, base, flush);
            if (error != 0 || slot == 0 || flush == 0 || pos == 0) {
                if (keyHead == 0) {
                    WriteStatus(row * heads_ * STATUS_WORDS, heads_ * STATUS_WORDS, error);
                }
                continue;
            }
            if (!heavyReady_) {
                InitFoldBuffers();
                heavyReady_ = true;
            }
            FoldPair(keyHead, slot, ringSlot, static_cast<uint32_t>(pos), static_cast<uint32_t>(base));
            WriteStatus(pair * STATUS_WORDS, STATUS_WORDS, 0);
        }
    }

private:
    __aicore__ inline void InitFoldBuffers()
    {
        // 216 KiB plus two aligned gate vectors and status. Both value heads
        // reuse all work buffers except that the converted K panel stays live.
        pipe_.InitBuffer(checkpointIn_, MATRIX * sizeof(half));
        pipe_.InitBuffer(checkpointOut_, MATRIX * sizeof(half));
        pipe_.InitBuffer(keyIn_, HISTORY * DIM * sizeof(half));
        pipe_.InitBuffer(deltaIn_, HISTORY * DIM * sizeof(half));
        pipe_.InitBuffer(keys_, HISTORY * DIM * sizeof(float));
        pipe_.InitBuffer(deltas_, HISTORY * DIM * sizeof(float));
        pipe_.InitBuffer(state_, MATRIX * sizeof(float));
        pipe_.InitBuffer(outer_, MATRIX * sizeof(float));
        pipe_.InitBuffer(gates_, 2 * RING * sizeof(float));
    }

    __aicore__ inline int32_t Validate(int32_t slot, int32_t ringSlot, int32_t pos,
                                      int32_t base, int32_t flush) const
    {
        if (slot < 0 || static_cast<uint32_t>(slot) >= numSlots_) return 1;
        if (slot == 0) return (pos != 0 || base != 0 || flush != 0) ? 4 : 0;
        if (flush != 0 && (ringSlot < 0 || static_cast<uint32_t>(ringSlot) >= numRingSlots_)) return 6;
        if (pos < 0 || static_cast<uint32_t>(pos) > HISTORY) return 2;
        if (base < 0 || static_cast<uint32_t>(base) >= RING) return 3;
        return (flush != 0 && flush != 1) ? 5 : 0;
    }

    __aicore__ inline void ReadRing(LocalTensor<half> dest, GlobalTensor<half>& source,
                                    uint64_t offset, uint32_t base, uint32_t pos)
    {
        const uint32_t first = pos < RING - base ? pos : RING - base;
        DataCopy(dest, source[offset + base * DIM], first * DIM);
        if (pos > first) {
            DataCopy(dest[first * DIM], source[offset], (pos - first) * DIM);
        }
    }

    __aicore__ inline void FoldPair(uint32_t keyHead, uint32_t slot, uint32_t ringSlot,
                                     uint32_t pos, uint32_t base)
    {
        auto keyIn = keyIn_.Get<half>();
        auto keys = keys_.Get<float>();
        const uint64_t kOffset = static_cast<uint64_t>(ringSlot) * kStride_ + keyHead * RING * DIM;
        ReadRing(keyIn, kCache_, kOffset, base, pos);
        Sync<HardEvent::MTE2_V>();
        Cast(keys, keyIn, RoundMode::CAST_NONE, pos * DIM);
        for (uint32_t h = 0; h < PAIR; ++h) {
            FoldHead(keyHead * PAIR + h, slot, ringSlot, pos, base, keys);
        }
        // All reads of the bulk inputs complete before the next pair reuses UB.
        Sync<HardEvent::V_MTE2>();
    }

    __aicore__ inline void FoldHead(uint32_t head, uint32_t slot, uint32_t ringSlot,
                                     uint32_t pos, uint32_t base, LocalTensor<float> keys)
    {
        const uint64_t stateOffset = static_cast<uint64_t>(slot) * checkpointStride_ + head * MATRIX;
        const uint64_t dOffset = static_cast<uint64_t>(ringSlot) * dStride_ + head * RING * DIM;
        const uint64_t gOffset = static_cast<uint64_t>(ringSlot) * gStride_ + head * RING;
        auto checkpointIn = checkpointIn_.Get<half>();
        auto checkpointOut = checkpointOut_.Get<half>();
        auto deltaIn = deltaIn_.Get<half>();
        auto deltas = deltas_.Get<float>();
        auto gates = gates_.Get<float>();
        auto decay = gates[RING];
        auto state = state_.Get<float>();
        auto outer = outer_.Get<float>();
        DataCopy(checkpointIn, checkpoint_[stateOffset], MATRIX);
        ReadRing(deltaIn, dCache_, dOffset, base, pos);
        DataCopy(gates, gCache_[gOffset], RING);
        Sync<HardEvent::MTE2_V>();
        Cast(state, checkpointIn, RoundMode::CAST_NONE, MATRIX);
        Cast(deltas, deltaIn, RoundMode::CAST_NONE, pos * DIM);
        Exp(decay, gates, RING);
        Sync<HardEvent::V_S>();
        // H_end = H_start * product(decay) + sum_i (suffix_i * D_i) outer K_i.
        // Weight the D rows in FP32 and leave the state tile on the vector pipe.
        float suffix = 1.0f;
        for (uint32_t i = pos; i != 0; --i) {
            const uint32_t physical = (base + i - 1) & (RING - 1);
            auto delta = deltas[(i - 1) * DIM];
            Muls(delta, delta, suffix, DIM);
            suffix *= decay.GetValue(physical);
        }
        Muls(state, state, suffix, MATRIX);
        PipeBarrier<PIPE_V>();
        constexpr uint32_t stateShape[2] = {DIM, DIM};
        constexpr uint32_t deltaShape[2] = {DIM, 1};
        for (uint32_t i = 0; i < pos; ++i) {
            auto key = keys[i * DIM];
            Broadcast<float, 2, 1>(outer, deltas[i * DIM], stateShape, deltaShape);
            PipeBarrier<PIPE_V>();
            Mul(outer, outer, key, 64, DIM, {1, 1, 1, 16, 16, 0});
            Mul(outer[64], outer[64], key[64], 64, DIM, {1, 1, 1, 16, 16, 0});
            PipeBarrier<PIPE_V>();
            Add(state, state, outer, MATRIX);
            PipeBarrier<PIPE_V>();
        }
        Cast(checkpointOut, state, RoundMode::CAST_NONE, MATRIX);
        Sync<HardEvent::V_MTE3>();
        DataCopy(checkpoint_[stateOffset], checkpointOut, MATRIX);
        Sync<HardEvent::MTE3_V>();
        Sync<HardEvent::V_MTE2>();
    }

    __aicore__ inline void WriteStatus(uint32_t offset, uint32_t count, int32_t error)
    {
        auto status = statusBuffer_.Get<int32_t>();
        Duplicate(status, error, count);
        Sync<HardEvent::V_MTE3>();
        DataCopy(status_[offset], status, count);
        Sync<HardEvent::MTE3_V>();
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> checkpointIn_, checkpointOut_, keyIn_, deltaIn_;
    TBuf<QuePosition::VECCALC> keys_, deltas_, gates_, state_, outer_, statusBuffer_;
    GlobalTensor<half> checkpoint_, dCache_, kCache_;
    GlobalTensor<float> gCache_;
    GlobalTensor<int32_t> slots_, ringSlots_, positions_, bases_, flush_, status_;
    uint32_t batch_, numSlots_, numRingSlots_, heads_;
    uint64_t checkpointStride_, dStride_, kStride_, gStride_;
    bool heavyReady_ = false;
};
}  // namespace

extern "C" __global__ __aicore__ void replayssm_sd_310p_compact_batch(
    GM_ADDR checkpoint, GM_ADDR dCache, GM_ADDR kCache, GM_ADDR gCache,
    GM_ADDR slots, GM_ADDR ringSlots, GM_ADDR positions, GM_ADDR bases, GM_ADDR flush, GM_ADDR status,
    uint32_t batch, uint32_t numSlots, uint32_t numRingSlots, uint32_t heads, uint64_t checkpointStride,
    uint64_t dStride, uint64_t kStride, uint64_t gStride)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    ReplayCompactBatch op;
    op.Init(checkpoint, dCache, kCache, gCache, slots, ringSlots, positions, bases, flush,
            status, batch, numSlots, numRingSlots, heads, checkpointStride, dStride, kStride, gStride);
    op.Process();
}

// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t DIM = 128;
constexpr uint32_t C0 = 16;
constexpr uint32_t FRACTAL = C0 * C0;
constexpr uint32_t MATRIX = DIM * DIM;
constexpr uint32_t PANEL = DIM * C0;
constexpr uint32_t BLOCKS = DIM / C0;
constexpr uint32_t HISTORY = 16;
constexpr uint32_t RING = 32;
constexpr uint32_t MAX_TOKENS = 8;
constexpr uint32_t HK = 8;
constexpr uint32_t HV = 16;
constexpr uint32_t PAIR = 2;
constexpr uint32_t KEYS_PER_STATUS = 2;

template <HardEvent Event>
__aicore__ inline void Sync()
{
    const auto id = static_cast<event_t>(GetTPipePtr()->FetchEventID(Event));
    SetFlag<Event>(id);
    WaitFlag<Event>(id);
}

class ReplayCubeVerify {
public:
    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR checkpoint,
        GM_ADDR dCache, GM_ADDR kCache, GM_ADDR gCache, GM_ADDR slots, GM_ADDR ringSlots,
        GM_ADDR positions, GM_ADDR bases, GM_ADDR lengths, GM_ADDR out, GM_ADDR status,
        uint32_t batch, uint32_t numSlots, uint32_t numRingSlots, uint64_t checkpointStride,
        uint64_t dStride, uint64_t kStride, uint64_t gStride, uint32_t width, float scale)
    {
        q_.SetGlobalBuffer((__gm__ half*)q);
        k_.SetGlobalBuffer((__gm__ half*)k);
        v_.SetGlobalBuffer((__gm__ half*)v);
        beta_.SetGlobalBuffer((__gm__ half*)beta);
        checkpoint_.SetGlobalBuffer((__gm__ half*)checkpoint);
        dCache_.SetGlobalBuffer((__gm__ half*)dCache);
        kCache_.SetGlobalBuffer((__gm__ half*)kCache);
        out_.SetGlobalBuffer((__gm__ half*)out);
        g_.SetGlobalBuffer((__gm__ float*)g);
        gCache_.SetGlobalBuffer((__gm__ float*)gCache);
        slots_.SetGlobalBuffer((__gm__ int32_t*)slots);
        ringSlots_.SetGlobalBuffer((__gm__ int32_t*)ringSlots);
        positions_.SetGlobalBuffer((__gm__ int32_t*)positions);
        bases_.SetGlobalBuffer((__gm__ int32_t*)bases);
        lengths_.SetGlobalBuffer((__gm__ int32_t*)lengths);
        status_.SetGlobalBuffer((__gm__ int32_t*)status);
        batch_ = batch;
        numSlots_ = numSlots;
        numRingSlots_ = numRingSlots;
        checkpointStride_ = checkpointStride;
        dStride_ = dStride;
        kStride_ = kStride;
        gStride_ = gStride;
        width_ = width;
        scale_ = scale;

        pipe_.InitBuffer(a1_, MATRIX * sizeof(half));
        pipe_.InitBuffer(b1_, PANEL * sizeof(half));
        pipe_.InitBuffer(a2_, 1, MATRIX * sizeof(half));
        pipe_.InitBuffer(b2_, 1, PANEL * sizeof(half));
        pipe_.InitBuffer(c1_, 1, PANEL * sizeof(float));
        pipe_.InitBuffer(c2_, 1, PANEL * sizeof(float));
        pipe_.InitBuffer(keys_, (HISTORY + MAX_TOKENS) * DIM * sizeof(half));
        pipe_.InitBuffer(rhs_, PANEL * sizeof(half));
        pipe_.InitBuffer(rhsFp32_, PANEL * sizeof(float));
        pipe_.InitBuffer(packed_, PANEL * sizeof(half));
        pipe_.InitBuffer(halfScratch_, PANEL * sizeof(half));
        pipe_.InitBuffer(vectors_, (HISTORY + MAX_TOKENS) * DIM * sizeof(float));
        pipe_.InitBuffer(dotProduct_, PANEL * sizeof(float));
        pipe_.InitBuffer(dotFold_, (PANEL / 2) * sizeof(float));
        pipe_.InitBuffer(dots_, (HISTORY + MAX_TOKENS) * C0 * sizeof(float));
        pipe_.InitBuffer(projection_, PANEL * sizeof(float));
        pipe_.InitBuffer(outer_, PANEL * sizeof(float));
        pipe_.InitBuffer(delta_, DIM * sizeof(float));
        pipe_.InitBuffer(result_, DIM * sizeof(float));
        pipe_.InitBuffer(gates_, (RING + 2 * MAX_TOKENS * HV) * sizeof(float));
        pipe_.InitBuffer(rates_, MAX_TOKENS * HV * sizeof(half));
        pipe_.InitBuffer(ratesFp32_, MAX_TOKENS * HV * sizeof(float));
        pipe_.InitBuffer(output_, 1, DIM * sizeof(half));
        pipe_.InitBuffer(gather_, (MAX_TOKENS * PAIR * DIM) * sizeof(uint32_t));
        pipe_.InitBuffer(packOffsets_, FRACTAL * sizeof(uint32_t));
        pipe_.InitBuffer(statusBuffer_, 8 * sizeof(int32_t));
        auto gather = gather_.Get<uint32_t>();
        for (uint32_t col = 0; col < MAX_TOKENS * PAIR; ++col) {
            for (uint32_t row = 0; row < DIM; ++row) {
                gather.SetValue(col * DIM + row, (row * C0 + col) * sizeof(float));
            }
        }
        auto packOffsets = packOffsets_.Get<uint32_t>();
        for (uint32_t lane = 0; lane < C0; ++lane) {
            for (uint32_t col = 0; col < C0; ++col) {
                packOffsets.SetValue(lane * C0 + col, (col * DIM + lane) * sizeof(half));
            }
        }
        Sync<HardEvent::S_V>();
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t groupsPerRow = HK / KEYS_PER_STATUS;
        for (uint32_t group = GetBlockIdx(); group < batch_ * groupsPerRow; group += GetBlockNum()) {
            const uint32_t row = group / groupsPerRow;
            const uint32_t keyHead = (group % groupsPerRow) * KEYS_PER_STATUS;
            const int32_t slot = slots_.GetValue(row);
            const int32_t ringSlot = ringSlots_.GetValue(row);
            const int32_t pos = positions_.GetValue(row);
            const int32_t base = bases_.GetValue(row);
            const int32_t length = lengths_.GetValue(row);
            const int32_t error = Validate(slot, ringSlot, pos, base, length);
            if (error != 0 || slot == 0) {
                ZeroOutputs(row, keyHead * PAIR, 0, KEYS_PER_STATUS * PAIR);
            } else {
                auto gates = gates_.Get<float>()[RING];
                auto decay = gates[MAX_TOKENS * HV];
                auto rates = rates_.Get<half>();
                auto ratesFp32 = ratesFp32_.Get<float>();
                DataCopy(gates, g_[row * width_ * HV], width_ * HV);
                DataCopy(rates, beta_[row * width_ * HV], width_ * HV);
                Sync<HardEvent::MTE2_V>();
                Exp(decay, gates, width_ * HV);
                Cast(ratesFp32, rates, RoundMode::CAST_NONE, width_ * HV);
                for (uint32_t pair = 0; pair < KEYS_PER_STATUS; ++pair) {
                    ProcessPair(row, keyHead + pair, slot, ringSlot, pos, base, length);
                }
                ZeroOutputs(row, keyHead * PAIR, length, KEYS_PER_STATUS * PAIR);
            }
            WriteStatus(group, error);
        }
    }

private:
    __aicore__ inline int32_t Validate(int32_t slot, int32_t ringSlot, int32_t pos,
                                      int32_t base, int32_t length) const
    {
        if (slot < 0 || static_cast<uint32_t>(slot) >= numSlots_) return 1;
        if (slot == 0) return (pos != 0 || base != 0 || length != 0) ? 4 : 0;
        if (ringSlot < 0 || static_cast<uint32_t>(ringSlot) >= numRingSlots_) return 5;
        if (pos < 0 || pos > static_cast<int32_t>(HISTORY) ||
            length < 1 || length > static_cast<int32_t>(width_)) return 2;
        if (base < 0 || base >= static_cast<int32_t>(RING)) return 3;
        return 0;
    }

    __aicore__ inline uint64_t OutputOffset(uint32_t row, uint32_t token, uint32_t head) const
    {
        return ((static_cast<uint64_t>(row) * width_ + token) * HV + head) * DIM;
    }

    __aicore__ inline uint32_t FirstSegment(uint32_t base, uint32_t count) const
    {
        return count < RING - base ? count : RING - base;
    }

    __aicore__ inline void ReadRing(LocalTensor<half> dst, GlobalTensor<half>& source,
                                    uint64_t headBase, uint32_t base, uint32_t count)
    {
        const uint32_t first = FirstSegment(base, count);
        if (first != 0) {
            DataCopy(dst, source[headBase + base * DIM], first * DIM);
        }
        if (count > first) {
            DataCopy(dst[first * DIM], source[headBase], (count - first) * DIM);
        }
    }

    __aicore__ inline void PreparePair(uint32_t row, uint32_t keyHead, uint64_t keyBase,
                                       uint32_t pos, uint32_t base, uint32_t length)
    {
        auto keys = keys_.Get<half>();
        auto rhs = rhs_.Get<half>();
        auto vectors = vectors_.Get<float>();
        auto rhsFp32 = rhsFp32_.Get<float>();
        const uint64_t offset = (static_cast<uint64_t>(row) * width_ * HK + keyHead) * DIM;
        const DataCopyParams copy{static_cast<uint16_t>(length), DIM / C0, (HK - 1) * DIM / C0, 0};
        ReadRing(keys, kCache_, keyBase, base, pos);
        DataCopy(keys[pos * DIM], k_[offset], copy);
        DataCopy(rhs[MAX_TOKENS * DIM], q_[offset], copy);
        Sync<HardEvent::MTE2_V>();
        Duplicate(rhs, static_cast<half>(0), MAX_TOKENS * DIM);
        if (length < MAX_TOKENS) {
            Duplicate(rhs[(MAX_TOKENS + length) * DIM], static_cast<half>(0), (MAX_TOKENS - length) * DIM);
        }
        PipeBarrier<PIPE_V>();
        Adds(rhs, keys[pos * DIM], static_cast<half>(0), length * DIM);
        Cast(vectors, keys, RoundMode::CAST_NONE, (pos + length) * DIM);
        PipeBarrier<PIPE_V>();
        Cast(rhsFp32, rhs, RoundMode::CAST_NONE, PANEL);
        PipeBarrier<PIPE_V>();

        auto packed = packed_.Get<half>();
        auto offsets = packOffsets_.Get<uint32_t>();
        for (uint32_t kb = 0; kb < BLOCKS; ++kb) {
            Gather(packed[kb * FRACTAL], rhs, offsets, kb * C0 * sizeof(half), FRACTAL);
        }
        PipeBarrier<PIPE_V>();

        auto product = dotProduct_.Get<float>();
        auto fold = dotFold_.Get<float>();
        auto dots = dots_.Get<float>();
        for (uint32_t i = 0; i < pos + length; ++i) {
            auto key = vectors[i * DIM];
            Mul(product, rhsFp32, key, 64, C0, {1, 1, 1, 16, 16, 0});
            Mul(product[64], rhsFp32[64], key[64], 64, C0, {1, 1, 1, 16, 16, 0});
            PipeBarrier<PIPE_V>();
            Add(fold, product, product[64], 64, C0, {1, 1, 1, 8, 16, 16});
            PipeBarrier<PIPE_V>();
            WholeReduceSum(dots[i * C0], fold, 64, C0, 1, 1, 8);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline LocalTensor<half> LoadRhs()
    {
        auto b1 = b1_.Get<half>();
        auto b2 = b2_.AllocTensor<half>();
        Sync<HardEvent::V_MTE3>();
        DataCopy(b1, packed_.Get<half>(), DataCopyParams{1, PANEL / C0, 0, 0});
        Sync<HardEvent::MTE3_MTE1>();
        LoadData2DParams params{};
        params.repeatTimes = BLOCKS;
        params.srcStride = 1;
        params.ifTranspose = true;
        LoadData(b2, b1, params);
        b2_.EnQue(b2);
        return b2_.DeQue<half>();
    }

    __aicore__ inline void Project(uint64_t stateOffset, LocalTensor<half> b2)
    {
        auto a1 = a1_.Get<half>();
        for (uint32_t kb = 0; kb < BLOCKS; ++kb) {
            DataCopy(a1[kb * PANEL], checkpoint_[stateOffset + kb * C0],
                     DataCopyParams{DIM, 1, DIM / C0 - 1, 0});
        }
        Sync<HardEvent::MTE2_MTE1>();
        auto a2 = a2_.AllocTensor<half>();
        LoadData2DParams load{};
        load.repeatTimes = BLOCKS;
        load.srcStride = BLOCKS;
        for (uint32_t mb = 0; mb < BLOCKS; ++mb) {
            LoadData(a2[mb * BLOCKS * FRACTAL], a1[mb * FRACTAL], load);
        }
        a2_.EnQue(a2);
        a2 = a2_.DeQue<half>();
        auto c1 = c1_.AllocTensor<float>();
        MmadParams params{};
        params.m = DIM;
        params.n = C0;
        params.k = DIM;
        params.cmatrixInitVal = true;
        Mmad(c1, a2, b2, params);
        c1_.EnQue(c1);
        a2_.FreeTensor(a2);
        c1 = c1_.DeQue<float>();
        auto c2 = c2_.AllocTensor<float>();
        DataCopyEnhancedParams drain{};
        drain.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(c2, c1, DataCopyParams{1, BLOCKS, 0, 0}, drain);
        c2_.EnQue(c2);
        c2 = c2_.DeQue<float>();
        PipeBarrier<PIPE_V>();
        Adds(projection_.Get<float>(), c2, 0.0f, PANEL);
        PipeBarrier<PIPE_V>();
        c2_.FreeTensor(c2);
        c1_.FreeTensor(c1);
        Sync<HardEvent::MTE1_MTE2>();
    }

    __aicore__ inline void AddOuter(LocalTensor<float> vector, LocalTensor<float> dots)
    {
        auto outer = outer_.Get<float>();
        auto projection = projection_.Get<float>();
        const uint32_t dstShape[2] = {DIM, C0};
        const uint32_t srcShape[2] = {DIM, 1};
        Broadcast<float, 2, 1>(outer, vector, dstShape, srcShape);
        PipeBarrier<PIPE_V>();
        Mul(outer, outer, dots, C0, DIM, {1, 1, 1, 2, 2, 0});
        PipeBarrier<PIPE_V>();
        Add(projection, projection, outer, PANEL);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessHead(uint32_t row, uint32_t head, uint64_t dBase, uint64_t gBase,
                                      uint32_t pos, uint32_t base, uint32_t length)
    {
        auto projection = projection_.Get<float>();
        auto dots = dots_.Get<float>();
        auto dHistory = vectors_.Get<float>();
        auto dAndValues = halfScratch_.Get<half>();
        auto historyGates = gates_.Get<float>();
        auto gates = historyGates[RING];
        auto decay = gates[MAX_TOKENS * HV];
        auto rates = ratesFp32_.Get<float>();
        auto delta = delta_.Get<float>();
        auto result = result_.Get<float>();
        auto gather = gather_.Get<uint32_t>();

        ReadRing(dAndValues, dCache_, dBase, base, pos);
        Sync<HardEvent::MTE2_V>();
        if (pos != 0) {
            Cast(dHistory, dAndValues, RoundMode::CAST_NONE, pos * DIM);
            Sync<HardEvent::V_MTE2>();
        }
        DataCopy(historyGates, gCache_[gBase], RING);
        DataCopy(dAndValues, v_[OutputOffset(row, 0, head)],
                 DataCopyParams{static_cast<uint16_t>(length), DIM / C0, (HV - 1) * DIM / C0, 0});
        Sync<HardEvent::MTE2_V>();
        // Use separate scratch so the raw gate bits can be preserved on writeback.
        auto alpha = result;
        Exp(alpha, historyGates, RING);
        Sync<HardEvent::V_S>();
        for (uint32_t i = 0; i < pos; ++i) {
            Muls(projection, projection, alpha.GetValue((base + i) & (RING - 1)), PANEL);
            PipeBarrier<PIPE_V>();
            AddOuter(dHistory[i * DIM], dots[i * C0]);
        }

        for (uint32_t t = 0; t < length; ++t) {
            const uint32_t lane = t * HV + head;
            const uint32_t phys = (base + pos + t) & (RING - 1);
            Muls(projection, projection, decay.GetValue(lane), PANEL);
            Cast(delta, dAndValues[t * DIM], RoundMode::CAST_NONE, DIM);
            PipeBarrier<PIPE_V>();
            Gather(result, projection, gather[t * DIM], 0, DIM);
            PipeBarrier<PIPE_V>();
            Sub(delta, delta, result, DIM);
            PipeBarrier<PIPE_V>();
            Muls(delta, delta, rates.GetValue(lane), DIM);
            PipeBarrier<PIPE_V>();
            Store(dCache_, dBase + phys * DIM, delta);
            AddOuter(delta, dots[(pos + t) * C0]);
            Gather(result, projection, gather[(MAX_TOKENS + t) * DIM], 0, DIM);
            PipeBarrier<PIPE_V>();
            Muls(result, result, scale_, DIM);
            PipeBarrier<PIPE_V>();
            Store(out_, OutputOffset(row, t, head), result);
            historyGates.SetValue(phys, gates.GetValue(lane));
        }
        Sync<HardEvent::S_MTE3>();
        DataCopy(gCache_[gBase], historyGates, RING);
        Sync<HardEvent::MTE3_MTE2>();
        Sync<HardEvent::V_MTE2>();
    }

    __aicore__ inline void ProcessPair(uint32_t row, uint32_t keyHead, uint32_t slot, uint32_t ringSlot,
                                      uint32_t pos, uint32_t base, uint32_t length)
    {
        const uint64_t keyBase = static_cast<uint64_t>(ringSlot) * kStride_ + keyHead * RING * DIM;
        PreparePair(row, keyHead, keyBase, pos, base, length);
        auto b2 = LoadRhs();
        for (uint32_t member = 0; member < PAIR; ++member) {
            const uint32_t head = keyHead * PAIR + member;
            Project(static_cast<uint64_t>(slot) * checkpointStride_ + head * MATRIX, b2);
            const uint64_t dBase = static_cast<uint64_t>(ringSlot) * dStride_ + head * RING * DIM;
            const uint64_t gBase = static_cast<uint64_t>(ringSlot) * gStride_ + head * RING;
            ProcessHead(row, head, dBase, gBase, pos, base, length);
        }
        b2_.FreeTensor(b2);
        const uint32_t start = (base + pos) & (RING - 1);
        const uint32_t first = FirstSegment(start, length);
        auto keys = keys_.Get<half>()[pos * DIM];
        DataCopy(kCache_[keyBase + start * DIM], keys, first * DIM);
        if (length > first) {
            DataCopy(kCache_[keyBase], keys[first * DIM], (length - first) * DIM);
        }
        Sync<HardEvent::MTE3_MTE2>();
        Sync<HardEvent::MTE3_V>();
    }

    __aicore__ inline void Store(GlobalTensor<half>& dest, uint64_t offset, LocalTensor<float> src)
    {
        auto tmp = output_.AllocTensor<half>();
        Cast(tmp, src, RoundMode::CAST_NONE, DIM);
        output_.EnQue(tmp);
        tmp = output_.DeQue<half>();
        DataCopy(dest[offset], tmp, DIM);
        output_.FreeTensor(tmp);
    }

    __aicore__ inline void ZeroOutputs(uint32_t row, uint32_t head, uint32_t firstToken, uint32_t count)
    {
        if (firstToken == width_) return;
        auto zeros = output_.AllocTensor<half>();
        Duplicate(zeros, static_cast<half>(0), DIM);
        output_.EnQue(zeros);
        zeros = output_.DeQue<half>();
        for (uint32_t t = firstToken; t < width_; ++t) {
            for (uint32_t h = 0; h < count; ++h) {
                DataCopy(out_[OutputOffset(row, t, head + h)], zeros, DIM);
            }
        }
        output_.FreeTensor(zeros);
    }

    __aicore__ inline void WriteStatus(uint32_t group, int32_t error)
    {
        auto status = statusBuffer_.Get<int32_t>();
        Duplicate(status, error, 8);
        Sync<HardEvent::V_MTE3>();
        DataCopy(status_[group * 8], status, 8);
        Sync<HardEvent::MTE3_V>();
    }

    TPipe pipe_;
    TBuf<TPosition::A1> a1_;
    TBuf<TPosition::B1> b1_;
    TQue<TPosition::A2, 1> a2_;
    TQue<TPosition::B2, 1> b2_;
    TQue<TPosition::CO1, 1> c1_;
    TQue<TPosition::CO2, 1> c2_;
    TQue<TPosition::VECOUT, 1> output_;
    TBuf<TPosition::VECCALC> keys_, rhs_, rhsFp32_, packed_, halfScratch_, vectors_;
    TBuf<TPosition::VECCALC> dotProduct_, dotFold_, dots_, projection_, outer_, delta_, result_;
    TBuf<TPosition::VECCALC> gates_, rates_, ratesFp32_, gather_, packOffsets_, statusBuffer_;
    GlobalTensor<half> q_, k_, v_, beta_, checkpoint_, dCache_, kCache_, out_;
    GlobalTensor<float> g_, gCache_;
    GlobalTensor<int32_t> slots_, ringSlots_, positions_, bases_, lengths_, status_;
    uint64_t checkpointStride_, dStride_, kStride_, gStride_;
    uint32_t batch_, numSlots_, numRingSlots_, width_;
    float scale_;
};
}  // namespace

extern "C" __global__ __aicore__ void replayssm_sd_310p_verify_cube(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR checkpoint,
    GM_ADDR dCache, GM_ADDR kCache, GM_ADDR gCache, GM_ADDR slots, GM_ADDR ringSlots,
    GM_ADDR positions, GM_ADDR bases, GM_ADDR lengths, GM_ADDR out, GM_ADDR status,
    uint32_t batch, uint32_t numSlots, uint32_t numRingSlots, uint64_t checkpointStride,
    uint64_t dStride, uint64_t kStride, uint64_t gStride, float scale, uint32_t width)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
    ReplayCubeVerify op;
    op.Init(q, k, v, g, beta, checkpoint, dCache, kCache, gCache, slots, ringSlots, positions,
            bases, lengths, out, status, batch, numSlots, numRingSlots, checkpointStride,
            dStride, kStride, gStride, width, scale);
    op.Process();
}

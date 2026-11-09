// SPDX-License-Identifier: Apache-2.0
// Same 16x128 dot problem as cube_matvec, using the native V310 vector pattern.
#include "kernel_operator.h"

using namespace AscendC;

class VectorMatvecProbe {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR vec, GM_ADDR out, uint32_t tiles)
    {
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(a));
        vecGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(vec));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
        tiles_ = tiles;
        pipe_.InitBuffer(aIn_, 16 * 128 * sizeof(half));
        pipe_.InitBuffer(vecIn_, 128 * sizeof(half));
        pipe_.InitBuffer(aFloat_, 16 * 128 * sizeof(float));
        pipe_.InitBuffer(vecFloat_, 128 * sizeof(float));
        pipe_.InitBuffer(product_, 16 * 128 * sizeof(float));
        pipe_.InitBuffer(fold_, 16 * 64 * sizeof(float));
        pipe_.InitBuffer(sum_, 16 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        auto aIn = aIn_.Get<half>();
        auto vecIn = vecIn_.Get<half>();
        auto aFloat = aFloat_.Get<float>();
        auto vecFloat = vecFloat_.Get<float>();
        auto product = product_.Get<float>();
        auto fold = fold_.Get<float>();
        auto sum = sum_.Get<float>();
        for (uint32_t tile = GetBlockIdx(); tile < tiles_; tile += GetBlockNum()) {
            DataCopy(aIn, aGm_[tile * 16 * 128], 16 * 128);
            DataCopy(vecIn, vecGm_[tile * 128], 128);
            event_t mte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(mte2ToV);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV);
            Cast(aFloat, aIn, RoundMode::CAST_NONE, 16 * 128);
            Cast(vecFloat, vecIn, RoundMode::CAST_NONE, 128);
            PipeBarrier<PIPE_V>();
            Mul(product, aFloat, vecFloat, 64, 16, {1, 1, 1, 16, 16, 0});
            Mul(product[64], aFloat[64], vecFloat[64], 64, 16, {1, 1, 1, 16, 16, 0});
            PipeBarrier<PIPE_V>();
            Add(fold, product, product[64], 64, 16, {1, 1, 1, 8, 16, 16});
            PipeBarrier<PIPE_V>();
            WholeReduceSum(sum, fold, 64, 16, 1, 1, 8);
            // The next iteration reuses the input buffers. Finish vector reads
            // before a following MTE2 transfer can overwrite them.
            event_t vToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(vToMte2);
            WaitFlag<HardEvent::V_MTE2>(vToMte2);
            event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(vToMte3);
            WaitFlag<HardEvent::V_MTE3>(vToMte3);
            DataCopy(outGm_[tile * 16], sum, 16);
            event_t mte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
        }
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> aIn_;
    TBuf<TPosition::VECCALC> vecIn_;
    TBuf<TPosition::VECCALC> aFloat_;
    TBuf<TPosition::VECCALC> vecFloat_;
    TBuf<TPosition::VECCALC> product_;
    TBuf<TPosition::VECCALC> fold_;
    TBuf<TPosition::VECCALC> sum_;
    GlobalTensor<half> aGm_;
    GlobalTensor<half> vecGm_;
    GlobalTensor<float> outGm_;
    uint32_t tiles_;
};

extern "C" __global__ __aicore__ void rgdr_v310_vector_matvec(
    GM_ADDR a, GM_ADDR vec, GM_ADDR out, uint32_t tiles)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    VectorMatvecProbe op;
    op.Init(a, vec, out, tiles);
    op.Process();
}

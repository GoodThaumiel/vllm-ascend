// SPDX-License-Identifier: Apache-2.0
// Ascend310P coupled-core probe for one 16x128 state tile times a 128-vector.
// B is padded to 128x16 with only column 0 populated. Inputs are ND FP16.
#include "kernel_operator.h"

using namespace AscendC;

class CubeMatvecProbe {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t tiles)
    {
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(a));
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b));
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c));
        tiles_ = tiles;
        pipe_.InitBuffer(a1Queue_, 1, 16 * 128 * sizeof(half));
        pipe_.InitBuffer(b1Queue_, 1, 128 * 16 * sizeof(half));
        pipe_.InitBuffer(a2Queue_, 1, 16 * 128 * sizeof(half));
        pipe_.InitBuffer(b2Queue_, 1, 128 * 16 * sizeof(half));
        pipe_.InitBuffer(c1Queue_, 1, 16 * 16 * sizeof(float));
        pipe_.InitBuffer(c2Queue_, 1, 16 * 16 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t tile = GetBlockIdx(); tile < tiles_; tile += GetBlockNum()) {
            auto a1 = a1Queue_.AllocTensor<half>();
            auto b1 = b1Queue_.AllocTensor<half>();
            // ND -> NZ. The B matrix has N=16, so its ND and NZ storage
            // coincide; A requires one 16-column transfer per K fractal.
            for (uint32_t kBlock = 0; kBlock < 8; ++kBlock) {
                DataCopy(a1[kBlock * 16 * 16],
                         aGm_[tile * 16 * 128 + kBlock * 16],
                         DataCopyParams{16, 1, 7, 0});
            }
            DataCopy(b1, bGm_[tile * 128 * 16], DataCopyParams{128, 1, 0, 0});
            a1Queue_.EnQue(a1);
            b1Queue_.EnQue(b1);

            a1 = a1Queue_.DeQue<half>();
            b1 = b1Queue_.DeQue<half>();
            auto a2 = a2Queue_.AllocTensor<half>();
            auto b2 = b2Queue_.AllocTensor<half>();
            LoadData2DParams aLoad{};
            aLoad.repeatTimes = 8;
            aLoad.srcStride = 1;
            LoadData(a2, a1, aLoad);
            LoadData2DParams bLoad{};
            bLoad.repeatTimes = 8;
            bLoad.srcStride = 1;
            bLoad.ifTranspose = true;
            LoadData(b2, b1, bLoad);
            a2Queue_.EnQue(a2);
            b2Queue_.EnQue(b2);
            a1Queue_.FreeTensor(a1);
            b1Queue_.FreeTensor(b1);

            a2 = a2Queue_.DeQue<half>();
            b2 = b2Queue_.DeQue<half>();
            auto c1 = c1Queue_.AllocTensor<float>();
            MmadParams params{};
            params.m = 16;
            params.n = 16;
            params.k = 128;
            params.cmatrixInitVal = true;
            Mmad(c1, a2, b2, params);
            c1Queue_.EnQue(c1);
            a2Queue_.FreeTensor(a2);
            b2Queue_.FreeTensor(b2);

            c1 = c1Queue_.DeQue<float>();
            auto c2 = c2Queue_.AllocTensor<float>();
            DataCopyEnhancedParams enhanced{};
            enhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
            DataCopy(c2, c1, DataCopyParams{1, 1, 0, 0}, enhanced);
            c2Queue_.EnQue(c2);
            c1Queue_.FreeTensor(c1);

            c2 = c2Queue_.DeQue<float>();
            // Follow the coupled-mode NZ -> ND copy-out path even for one
            // fractal; the transfer is 16 rows of 16 FP32 values (2 blocks).
            DataCopy(cGm_[tile * 16 * 16], c2, DataCopyParams{16, 2, 0, 0});
            c2Queue_.FreeTensor(c2);
        }
    }

private:
    TPipe pipe_;
    TQue<TPosition::A1, 1> a1Queue_;
    TQue<TPosition::B1, 1> b1Queue_;
    TQue<TPosition::A2, 1> a2Queue_;
    TQue<TPosition::B2, 1> b2Queue_;
    TQue<TPosition::CO1, 1> c1Queue_;
    TQue<TPosition::CO2, 1> c2Queue_;
    GlobalTensor<half> aGm_;
    GlobalTensor<half> bGm_;
    GlobalTensor<float> cGm_;
    uint32_t tiles_;
};

extern "C" __global__ __aicore__ void rgdr_v310_cube_matvec(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t tiles)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
    CubeMatvecProbe op;
    op.Init(a, b, c, tiles);
    op.Process();
}

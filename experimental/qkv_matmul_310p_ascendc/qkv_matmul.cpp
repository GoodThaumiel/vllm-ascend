// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

#include <type_traits>

#include "geometry.h"

namespace {
using namespace AscendC;
using namespace qkv310p;

template <typename Accumulator>
class QkvMatmul {
    static constexpr bool FP32_ACCUM = std::is_same_v<Accumulator, float>;
    static constexpr uint32_t ACC_BYTES = FP32_ACCUM ? C_TILE_FP32_BYTES : C_TILE_BYTES;

public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR out, uint32_t rows)
    {
        rows_ = rows;
        rowBlocks_ = (rows + C0 - 1) / C0;
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x), rowBlocks_ * C0 * K);
        weight_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(weight), N * K);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(out), rows * N);
        pipe_.InitBuffer(a1_, 1, A_SLICE_BYTES);
        pipe_.InitBuffer(b1_, 1, W_PANEL_BYTES);
        pipe_.InitBuffer(a2_, 1, A_SLICE_BYTES);
        pipe_.InitBuffer(b2_, 1, W_SLICE_BYTES);
        pipe_.InitBuffer(c1_, 1, ACC_BYTES);
        pipe_.InitBuffer(c2_, 1, ACC_BYTES);
        if constexpr (FP32_ACCUM) {
            pipe_.InitBuffer(castOut_, 1, C_TILE_BYTES);
        }
    }

    __aicore__ inline void Run()
    {
        // One core owns a disjoint set of 128-column panels of the ND output.
        for (uint32_t tile = GetBlockIdx(); tile < N_TILES; tile += GetBlockNum()) {
            const uint32_t col = tile * TILE_N;
            auto panel = LoadWeightPanel(col);
            for (uint32_t row = 0; row < rows_; row += TILE_M) {
                const uint32_t remaining = rows_ - row;
                const uint16_t validRows = remaining < TILE_M ? remaining : TILE_M;
                const uint16_t mBlocks = (validRows + C0 - 1) / C0;
                ComputeTile(panel, row, col, validRows, mBlocks);
            }
            b1_.FreeTensor(panel);
        }
    }

private:
    __aicore__ inline LocalTensor<half> LoadWeightPanel(uint32_t col)
    {
        auto panel = b1_.AllocTensor<half>();
        // W[N,K] NZ is [K/16,N/16,N0,K0]. Each burst loads one strip of N.
        // Ordinary DataCopy uses 32B blocks, one FP16 fractal occupies 16.
        DataCopy(panel, weight_[col * C0],
                 DataCopyParams{K / C0, TILE_N, N - TILE_N, 0});
        b1_.EnQue(panel);
        return b1_.DeQue<half>();
    }

    __aicore__ inline void ComputeTile(LocalTensor<half> panel, uint32_t row, uint32_t col,
                                       uint16_t validRows, uint16_t mBlocks)
    {
        auto acc = c1_.AllocTensor<Accumulator>();
        MmadParams mma{};
        mma.m = mBlocks * C0;
        mma.n = TILE_N;
        mma.k = TILE_K;
        for (uint32_t k = 0; k < K; k += TILE_K) {
            auto xSlice = a1_.AllocTensor<half>();
            const uint64_t xOffset = (static_cast<uint64_t>(k / C0) * rowBlocks_ + row / C0) * FRACTAL;
            const uint16_t copyRows = mBlocks * C0;
            DataCopy(xSlice, x_[xOffset],
                     DataCopyParams{TILE_K / C0, copyRows,
                                    static_cast<uint16_t>(rowBlocks_ * C0 - copyRows), 0});
            a1_.EnQue(xSlice);
            xSlice = a1_.DeQue<half>();

            auto a = a2_.AllocTensor<half>();
            auto b = b2_.AllocTensor<half>();
            LoadData2DParams loadA{};
            loadA.repeatTimes = TILE_K / C0;
            loadA.srcStride = mBlocks;
            // A: NZ to ZZ, reorder the outer fractals.
            for (uint32_t m = 0; m < mBlocks; ++m) {
                LoadData(a[m * (TILE_K / C0) * FRACTAL], xSlice[m * FRACTAL], loadA);
            }
            // W[N,K] NZ bytes already represent ZN for B=W^T[K,N].
            LoadData2DParams loadB{};
            loadB.repeatTimes = (TILE_K / C0) * (TILE_N / C0);
            loadB.srcStride = 1;
            LoadData(b, panel[k * TILE_N], loadB);
            a2_.EnQue(a);
            b2_.EnQue(b);
            a1_.FreeTensor(xSlice);

            a = a2_.DeQue<half>();
            b = b2_.DeQue<half>();
            mma.cmatrixInitVal = (k == 0);
            Mmad(acc, a, b, mma);
            a2_.FreeTensor(a);
            b2_.FreeTensor(b);
        }
        c1_.EnQue(acc);
        StoreTile(row, col, validRows, mBlocks);
    }

    __aicore__ inline void StoreTile(uint32_t row, uint32_t col, uint16_t validRows,
                                    uint16_t mBlocks)
    {
        auto acc = c1_.DeQue<Accumulator>();
        auto ub = c2_.AllocTensor<Accumulator>();
        DataCopyEnhancedParams drain{};
        drain.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        if constexpr (FP32_ACCUM) {
            // FP32 transfer in NZ strips, as in M200 hand_mmad_310p.
            DataCopy(ub, acc, DataCopyParams{TILE_N / C0, mBlocks, 0, 0}, drain);
        } else {
            const uint16_t fractals = mBlocks * (TILE_N / C0);
            DataCopy(ub, acc, DataCopyParams{1, fractals, 0, 0}, drain);
        }
        c2_.EnQue(ub);
        c1_.FreeTensor(acc);

        ub = c2_.DeQue<Accumulator>();
        if constexpr (FP32_ACCUM) {
            auto converted = castOut_.AllocTensor<half>();
            const uint32_t stripElems = mBlocks * FRACTAL;
            PipeBarrier<PIPE_V>();
            for (uint32_t n = 0; n < TILE_N / C0; ++n) {
                const uint32_t offset = n * stripElems;
                Cast(converted[offset], ub[offset], RoundMode::CAST_NONE, stripElems);
            }
            castOut_.EnQue(converted);
            converted = castOut_.DeQue<half>();
            StoreNd(converted, row, col, validRows, mBlocks);
            castOut_.FreeTensor(converted);
        } else {
            StoreNd(ub, row, col, validRows, mBlocks);
        }
        c2_.FreeTensor(ub);
    }

    __aicore__ inline void StoreNd(LocalTensor<half> ub, uint32_t row, uint32_t col,
                                  uint16_t validRows, uint16_t mBlocks)
    {
        // Each NZ column strip has a contiguous 32B burst for each matrix row.
        // Write just the logical rows, even if the Cube used a padded M tail.
        const uint64_t dst = static_cast<uint64_t>(row) * N + col;
        const uint32_t stripStride = mBlocks * FRACTAL;
        for (uint32_t n = 0; n < TILE_N / C0; ++n) {
            DataCopy(out_[dst + n * C0], ub[n * stripStride],
                     DataCopyParams{validRows, 1, 0, N_BLOCKS - 1});
        }
    }

    TPipe pipe_;
    TQue<TPosition::A1, 1> a1_;
    TQue<TPosition::B1, 1> b1_;
    TQue<TPosition::A2, 1> a2_;
    TQue<TPosition::B2, 1> b2_;
    TQue<TPosition::CO1, 1> c1_;
    TQue<TPosition::CO2, 1> c2_;
    TQue<TPosition::VECOUT, 1> castOut_;
    GlobalTensor<half> x_, weight_, out_;
    uint32_t rows_, rowBlocks_;
};
}  // namespace

extern "C" __global__ __aicore__ void qkv_matmul_310p_fp16(
    GM_ADDR x, GM_ADDR weight, GM_ADDR out, uint32_t rows)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
    QkvMatmul<half> op;
    op.Init(x, weight, out, rows);
    op.Run();
}

extern "C" __global__ __aicore__ void qkv_matmul_310p_fp32_accum(
    GM_ADDR x, GM_ADDR weight, GM_ADDR out, uint32_t rows)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
    QkvMatmul<float> op;
    op.Init(x, weight, out, rows);
    op.Run();
}

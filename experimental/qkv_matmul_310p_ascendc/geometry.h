// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace qkv310p {
constexpr uint32_t K = 2048;
constexpr uint32_t N = 6144;
constexpr uint32_t TILE_M = 256;
constexpr uint32_t TILE_N = 128;
constexpr uint32_t TILE_K = 128;
constexpr uint32_t C0 = 16;
constexpr uint32_t FRACTAL = C0 * C0;
constexpr uint32_t N_TILES = N / TILE_N;
constexpr uint32_t N_BLOCKS = N / C0;
constexpr uint32_t FP16_BYTES = 2;
constexpr uint32_t A_SLICE_BYTES = TILE_M * TILE_K * FP16_BYTES;
constexpr uint32_t W_SLICE_BYTES = TILE_K * TILE_N * FP16_BYTES;
constexpr uint32_t W_PANEL_BYTES = K * TILE_N * FP16_BYTES;
constexpr uint32_t C_TILE_BYTES = TILE_M * TILE_N * FP16_BYTES;
constexpr uint32_t C_TILE_FP32_BYTES = TILE_M * TILE_N * sizeof(float);

static_assert(N % TILE_N == 0 && K % TILE_K == 0);
static_assert(TILE_M % C0 == 0 && TILE_N % C0 == 0 && TILE_K % C0 == 0);
}  // namespace qkv310p

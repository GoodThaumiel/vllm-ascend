// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>

constexpr int64_t WY_CHUNK_SIZE = 64;
constexpr int64_t WY_MAX_HEAD_DIM = 128;
constexpr uint32_t WY_LOCAL_WORKSPACE_BYTES = 32 * 1024;
// The two FP16 Cube input slices and the FP32 snapshot retain v24's stride.
constexpr uint32_t WY_PER_CORE_WORKSPACE_BYTES =
    2 * WY_CHUNK_SIZE * WY_MAX_HEAD_DIM * sizeof(uint16_t) +
    WY_CHUNK_SIZE * WY_CHUNK_SIZE * sizeof(float);

#pragma pack(push, 1)
struct WyLaunchTiling {
  int64_t batch;
  int64_t seqlen;
  int64_t kNumHead;
  int64_t vNumHead;
  int64_t kHeadDim;
  int64_t vHeadDim;
  int64_t chunkSize;
  int64_t numChunks;
  int64_t groupSize;
  int64_t totalTasks;
  uint32_t localWorkspaceSize;
  uint32_t perCoreWorkspaceBytes;
  uint32_t usedCoreNum;
  uint64_t workspaceOffset;
};
#pragma pack(pop)
static_assert(sizeof(WyLaunchTiling) == 100);

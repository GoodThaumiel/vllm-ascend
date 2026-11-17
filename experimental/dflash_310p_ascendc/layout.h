// SPDX-License-Identifier: Apache-2.0
#ifndef DFLASH_310P_LAYOUT_H
#define DFLASH_310P_LAYOUT_H

#include <cstdint>

namespace dflash310 {
constexpr uint32_t kDmaLanes = 32 / sizeof(int32_t);
constexpr uint32_t kMaxDraftTokens = 7;
constexpr uint32_t kMaxGroups = 2;  // Full attention and sliding attention.
constexpr int32_t kPadSlot = -1;

enum RequestLane : uint32_t { ContextStart, ContextEnd, Rejected, BonusId };
enum GroupLane : uint32_t { PageSize, PageCount, NullPage };
enum QueryField : uint32_t { InputIds, Positions, Slots, SampleIndices, SamplePositions, RequestInfo, QueryFields };
enum InfoLane : uint32_t { QueryStart, QueryEnd, SeqLen, Status };
enum ContextField : uint32_t { ContextPositions, ContextSlots, ContextFields };
}  // namespace dflash310

#endif

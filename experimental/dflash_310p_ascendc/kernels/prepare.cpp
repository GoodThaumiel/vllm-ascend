// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

#include "../layout.h"

using namespace AscendC;
using namespace dflash310;

namespace {

struct Request {
  int32_t start;
  int32_t end;
  int32_t rejected;
  int32_t bonus;
  int32_t accepted_end;
  int32_t query_position;
  int32_t status;
  bool valid;
};

class PrepareDFlash {
 public:
  __aicore__ inline void Init(GM_ADDR requests, GM_ADDR positions, GM_ADDR tables, GM_ADDR group_info,
                              GM_ADDR query_out, GM_ADDR context_out, uint32_t batch, uint32_t context_capacity,
                              uint32_t table_width, uint32_t spec, uint32_t max_model_len, int32_t mask_id,
                              uint32_t from_anchor) {
    const uint32_t group = GetBlockIdx();
    batch_ = batch;
    capacity_ = context_capacity;
    table_width_ = table_width;
    spec_ = spec;
    width_ = spec + 1;
    max_model_len_ = static_cast<int32_t>(max_model_len);
    mask_id_ = mask_id;
    from_anchor_ = from_anchor;
    requests_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(requests), batch * kDmaLanes);
    positions_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(positions), capacity_);
    tables_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tables) + group * batch * table_width,
                            batch * table_width);
    group_info_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(group_info) + group * kDmaLanes, kDmaLanes);
    query_out_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(query_out) + group * batch * QueryFields * kDmaLanes,
                              batch * QueryFields * kDmaLanes);
    context_out_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(context_out) + group * ContextFields * capacity_,
                                ContextFields * capacity_);
    page_size_ = group_info_.GetValue(PageSize);
    page_count_ = group_info_.GetValue(PageCount);
    null_page_ = group_info_.GetValue(NullPage);
    pipe_.InitBuffer(query_buf_, QueryFields * kDmaLanes * sizeof(int32_t));
    pipe_.InitBuffer(context_buf_, ContextFields * kDmaLanes * sizeof(int32_t));
  }

  __aicore__ inline void Process() {
    LocalTensor<int32_t> query = query_buf_.Get<int32_t>();
    for (uint32_t row = 0; row < batch_; ++row) {
      Request req = ReadRequest(row);
      for (uint32_t i = 0; i < kDmaLanes; ++i) {
        InitQueryLane(query, i);
        if (!req.valid) {
          continue;
        }
        if (i < width_) {
          const int32_t pos = req.query_position + static_cast<int32_t>(i);
          query.SetValue(InputIds * kDmaLanes + i, i == 0 ? req.bonus : mask_id_);
          query.SetValue(Positions * kDmaLanes + i, CapPosition(pos));
          query.SetValue(Slots * kDmaLanes + i, Slot(row, pos, false, req.status));
        }
        if (i < spec_) {
          const uint32_t sample_lane = i + (from_anchor_ ? 0 : 1);
          query.SetValue(SampleIndices * kDmaLanes + i, row * width_ + sample_lane);
          query.SetValue(SamplePositions * kDmaLanes + i, req.query_position + static_cast<int32_t>(i) + 1);
        }
      }
      if (req.valid) {
        for (int32_t i = req.start; i < req.accepted_end; ++i) {
          Slot(row, positions_.GetValue(i), true, req.status);
        }
      }
      const uint32_t info = RequestInfo * kDmaLanes;
      query.SetValue(info + QueryStart, row * width_);
      query.SetValue(info + QueryEnd, (row + req.valid) * width_);
      const int32_t end = req.query_position + static_cast<int32_t>(width_);
      query.SetValue(info + SeqLen, req.valid ? (end < max_model_len_ ? end : max_model_len_) : 0);
      query.SetValue(info + Status, req.status);
      BeginWrite();
      DataCopy(query_out_[row * QueryFields * kDmaLanes], query, QueryFields * kDmaLanes);
      EndWrite();
    }

    LocalTensor<int32_t> context = context_buf_.Get<int32_t>();
    uint32_t row = 0;
    Request req = ReadRequest(row);
    for (uint32_t offset = 0; offset < capacity_; offset += kDmaLanes) {
      for (uint32_t lane = 0; lane < kDmaLanes; ++lane) {
        const int32_t index = static_cast<int32_t>(offset + lane);
        while (row < batch_ && (!req.valid || index >= req.end)) {
          ++row;
          if (row < batch_) {
            req = ReadRequest(row);
          }
        }
        int32_t pos = 0;
        int32_t slot = kPadSlot;
        if (row < batch_ && index >= req.start && index < req.accepted_end) {
          const int32_t value = positions_.GetValue(index);
          int32_t status = 0;
          slot = Slot(row, value, true, status);
          pos = status == 0 ? value : 0;
        }
        context.SetValue(lane, pos);
        context.SetValue(kDmaLanes + lane, slot);
      }
      BeginWrite();
      DataCopy(context_out_[offset], context, kDmaLanes);
      DataCopy(context_out_[capacity_ + offset], context[kDmaLanes], kDmaLanes);
      EndWrite();
    }
  }

 private:
  __aicore__ inline Request ReadRequest(uint32_t row) {
    const uint32_t offset = row * kDmaLanes;
    Request req{};
    req.start = requests_.GetValue(offset + ContextStart);
    if (req.start == -1) {
      return req;
    }
    req.end = requests_.GetValue(offset + ContextEnd);
    req.rejected = requests_.GetValue(offset + Rejected);
    req.bonus = requests_.GetValue(offset + BonusId);
    if (req.start < 0 || req.end <= req.start || req.end > static_cast<int32_t>(capacity_) ||
        req.rejected < 0 || req.rejected > req.end - req.start || req.bonus < 0) {
      req.status = 1;
      return req;
    }
    req.accepted_end = req.end - req.rejected;
    const bool has_accepted = req.accepted_end > req.start;
    const int32_t pos = positions_.GetValue(has_accepted ? req.accepted_end - 1 : req.start);
    if (pos < 0 || pos >= max_model_len_) {
      req.status = 1;
      return req;
    }
    req.query_position = pos + static_cast<int32_t>(has_accepted);
    req.valid = true;
    return req;
  }

  __aicore__ inline int32_t CapPosition(int32_t pos) {
    return pos < max_model_len_ ? pos : max_model_len_ - 1;
  }

  __aicore__ inline int32_t Slot(uint32_t row, int32_t pos, bool is_context, int32_t& status) {
    if (pos >= max_model_len_) {
      if (is_context) {
        status = 2;
      }
      return kPadSlot;
    }
    if (pos < 0 || page_size_ <= 0 || page_count_ <= 0 || pos / page_size_ >= static_cast<int32_t>(table_width_)) {
      status = 2;
      return kPadSlot;
    }
    const int32_t page = tables_.GetValue(row * table_width_ + pos / page_size_);
    if (page >= 0 && page == null_page_ && is_context) {
      return kPadSlot;
    }
    if (page < 0 || page >= page_count_ || page == null_page_) {
      status = 2;
      return kPadSlot;
    }
    const int64_t slot = static_cast<int64_t>(page) * page_size_ + pos % page_size_;
    if (slot > 2147483647LL) {
      status = 2;
      return kPadSlot;
    }
    return static_cast<int32_t>(slot);
  }

  __aicore__ inline void InitQueryLane(LocalTensor<int32_t> query, uint32_t lane) {
    for (uint32_t field = 0; field < QueryFields; ++field) {
      query.SetValue(field * kDmaLanes + lane, field == Slots ? kPadSlot : 0);
    }
  }

  __aicore__ inline void BeginWrite() {
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
  }

  __aicore__ inline void EndWrite() {
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
  }

  TPipe pipe_;
  TBuf<TPosition::VECCALC> query_buf_, context_buf_;
  GlobalTensor<int32_t> requests_, positions_, tables_, group_info_, query_out_, context_out_;
  uint32_t batch_, capacity_, table_width_, spec_, width_, from_anchor_;
  int32_t max_model_len_, mask_id_, page_size_, page_count_, null_page_;
};

}  // namespace

extern "C" __global__ __aicore__ void dflash_310p_prepare(
    GM_ADDR requests, GM_ADDR positions, GM_ADDR tables, GM_ADDR group_info, GM_ADDR query_out, GM_ADDR context_out,
    uint32_t batch, uint32_t context_capacity, uint32_t table_width, uint32_t spec, uint32_t max_model_len,
    int32_t mask_id, uint32_t from_anchor) {
  PrepareDFlash op;
  op.Init(requests, positions, tables, group_info, query_out, context_out, batch, context_capacity,
          table_width, spec, max_model_len, mask_id, from_anchor);
  op.Process();
}

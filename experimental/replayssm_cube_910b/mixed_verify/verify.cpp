// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t DIM = 128;
constexpr uint32_t TILE = DIM / 2;
constexpr uint32_t COLS = 16;
constexpr uint32_t TOKENS = 4;
constexpr uint32_t HISTORY = 16;
constexpr uint32_t RING = 32;
constexpr uint32_t FRACTAL = 16 * 16;
constexpr uint32_t FLOAT_LANES_PER_BLOCK = 32 / sizeof(float);
constexpr uint32_t CANDIDATE_SCALARS = TOKENS * FLOAT_LANES_PER_BLOCK;
constexpr uint32_t MIX_MODE = 2;
constexpr uint16_t READY = 6;
constexpr uint16_t CONSUMED = 7;
constexpr float SCALE = 0.08838834764831845f;

template <HardEvent Event>
__aicore__ inline void Fence(event_t id) {
  SetFlag<Event>(id);
  WaitFlag<Event>(id);
}

template <HardEvent Event>
__aicore__ inline void Fence() {
  Fence<Event>(static_cast<event_t>(GetTPipePtr()->FetchEventID(Event)));
}

__aicore__ inline uint32_t Physical(uint32_t base, uint32_t p) {
  return (base + p) & (RING - 1);
}

struct Inputs {
  GlobalTensor<bfloat16_t> q, k, v, checkpoint, out, zeros;
  GlobalTensor<half> d, keys;
  GlobalTensor<float> g, beta, gates, scratch;
  GlobalTensor<int32_t> slots, ringSlots, pos, base, length, status, flush, tokenOffsets;
  uint32_t batch, hk, hv, numSlots, numRingSlots, blocks, tokenCount;
  uint64_t checkpointStride;
  bool perHeadWorkspace, balanceHeads;

  __aicore__ inline uint32_t FirstTask(uint32_t block, uint32_t group) const {
    return balanceHeads ? (batch * hv * block) / blocks : block * group;
  }

  __aicore__ inline uint32_t EndTask(uint32_t block) const {
    return balanceHeads ? (batch * hv * (block + 1)) / blocks : batch * hv;
  }

  __aicore__ inline uint32_t GroupEnd(uint32_t task, uint32_t end, uint32_t group) const {
    if (!balanceHeads) {
      return task + group;
    }
    const uint32_t groupEnd = (task / group + 1) * group;
    return groupEnd < end ? groupEnd : end;
  }

  __aicore__ inline uint32_t NextTask(uint32_t task, uint32_t groupEnd, uint32_t group) const {
    return balanceHeads ? groupEnd : task + blocks * group;
  }

  __aicore__ inline int32_t ReadWindow(uint32_t row, uint32_t &tokenStart) {
    const int32_t len = length.GetValue(row);
    if (tokenCount == 0) {
      tokenStart = row * TOKENS;
      return len;
    }
    const int32_t begin = tokenOffsets.GetValue(row);
    const int32_t end = tokenOffsets.GetValue(row + 1);
    if (begin < 0 || end < begin || static_cast<uint32_t>(end) > tokenCount ||
        end - begin != len) {
      return -1;
    }
    tokenStart = begin;
    return len;
  }

  __aicore__ inline uint64_t KeyInput(uint32_t tokenStart, uint32_t key) const {
    return (static_cast<uint64_t>(tokenStart) * hk + key) * DIM;
  }
  __aicore__ inline uint64_t Output(uint32_t tokenStart, uint32_t t, uint32_t head) const {
    return ((static_cast<uint64_t>(tokenStart) + t) * hv + head) * DIM;
  }

  __aicore__ inline uint64_t Checkpoint(uint32_t slot, uint32_t head, uint32_t vStart = 0) const {
    return slot * checkpointStride + (head * DIM + vStart) * DIM;
  }

  __aicore__ inline uint64_t Projection(uint32_t block, uint32_t row, uint32_t head) const {
    return (perHeadWorkspace ? static_cast<uint64_t>(row) * hv + head : block) * DIM * COLS;
  }
};

#ifdef __DAV_C220_CUBE__
class Projector {
 public:
  __aicore__ inline void Run(Inputs &x) {
    const uint32_t block = GetBlockIdx();
    const uint32_t group = x.hv / x.hk;
    pipe_.InitBuffer(a1_, DIM * DIM * sizeof(bfloat16_t));
    pipe_.InitBuffer(b1_, COLS * DIM * sizeof(bfloat16_t));
    pipe_.InitBuffer(a2_, DIM * DIM * sizeof(bfloat16_t));
    pipe_.InitBuffer(b2_, COLS * DIM * sizeof(bfloat16_t));
    pipe_.InitBuffer(c1_, DIM * COLS * sizeof(float));

    const uint32_t end = x.EndTask(block);
    for (uint32_t task = x.FirstTask(block, group); task < end;) {
      const uint32_t row = task / x.hv;
      const uint32_t firstHead = task - row * x.hv;
      const uint32_t key = firstHead / group;
      const uint32_t groupEnd = x.GroupEnd(task, end, group);
      const int32_t slot = x.slots.GetValue(row);
      uint32_t tokenStart = row * TOKENS;
      const int32_t len = x.tokenCount ? x.ReadWindow(row, tokenStart) : TOKENS;
      const bool valid = slot > 0 && static_cast<uint32_t>(slot) < x.numSlots &&
                         len > 0 && len <= TOKENS;
      if (valid) {
        StageRhs(x, tokenStart, key, len);
      }
      for (uint32_t head = firstHead; head < firstHead + groupEnd - task; ++head) {
        if (valid) {
          const uint64_t stateOff = x.Checkpoint(slot, head);
          Project(x, stateOff, x.Projection(block, row, head), head == firstHead);
        }
        // Pad/error rows have the same handshakes. The AIVs skip the read.
        CrossCoreSetFlag<MIX_MODE, PIPE_FIX>(READY);
        // One ack at the AIC after both AIVs have loaded their tiles.
        CrossCoreWaitFlag(CONSUMED);
      }
      task = x.NextTask(task, groupEnd, group);
    }
  }

 private:
  __aicore__ inline void ToL1(LocalTensor<bfloat16_t> dst, GlobalTensor<bfloat16_t> src,
                              uint16_t rows, uint16_t pitch, uint16_t nzRows) {
    Nd2NzParams p{};
    p.ndNum = 1;
    p.nValue = rows;
    p.dValue = DIM;
    p.srcDValue = pitch;
    p.dstNzC0Stride = nzRows;
    p.dstNzNStride = 1;
    DataCopy(dst, src, p);
  }

  __aicore__ inline void StageRhs(Inputs &x, uint32_t tokenStart, uint32_t key, uint32_t len) {
    // NZ for [N,K]. Each tile has 16 rows; K goes in 0..3 and Q in 4..7.
    // Clear only the unused lanes. A full L1 clear would overlap the K/Q
    // destinations, and successive MTE2 writes can be in flight together.
    auto b1 = b1_.Get<bfloat16_t>();
    const auto off = x.KeyInput(tokenStart, key);
    ToL1(b1, x.k[off], len, x.hk * DIM, COLS);
    ToL1(b1[TOKENS * 16], x.q[off], len, x.hk * DIM, COLS);
    if (len < TOKENS) {
      ToL1(b1[len * 16], x.zeros, TOKENS - len, DIM, COLS);
    }
    ToL1(b1[(TOKENS + len) * 16], x.zeros, COLS - TOKENS - len, DIM, COLS);
  }

  __aicore__ inline void LoadRhsFromL1() {
    LoadData2DParams p{};
    p.repeatTimes = DIM / 16;
    p.srcStride = 1;
    // Already NZ for [N,K], matching arch22 SFA's B load.
    LoadData(b2_.Get<bfloat16_t>(), b1_.Get<bfloat16_t>(), p);
  }

  __aicore__ inline void Project(Inputs &x, uint64_t off, uint64_t scratchOffset, bool loadRhs) {
    auto a1 = a1_.Get<bfloat16_t>();
    auto a2 = a2_.Get<bfloat16_t>();
    auto c = c1_.Get<float>();
    ToL1(a1, x.checkpoint[off], DIM, DIM, DIM);
    // RHS staging and checkpoint staging use the same MTE2 pipe.
    // Wait once for both, so an outstanding wait is never signalled twice.
    Fence<HardEvent::MTE2_MTE1>();
    if (loadRhs) {
      LoadRhsFromL1();
    }
    LoadData2DParams p{};
    p.repeatTimes = DIM / 16;
    p.srcStride = DIM / 16;
    for (uint32_t m = 0; m < DIM / 16; ++m) {
      LoadData(a2[m * (DIM / 16) * FRACTAL], a1[m * FRACTAL], p);
    }
    Fence<HardEvent::MTE1_M>();
    MmadParams mm{};
    mm.m = DIM;
    mm.n = COLS;
    mm.k = DIM;
    mm.cmatrixInitVal = true;
    Mmad(c, a2, b2_.Get<bfloat16_t>(), mm);
    // The 128x16 tile covers eight output fractals, below the arch22
    // small-tile threshold used by the repository's SFA Mmad path.
    PipeBarrier<PIPE_M>();
    Fence<HardEvent::M_FIX>();
    FixpipeParamsV220 fx{};
    fx.mSize = DIM;
    fx.nSize = COLS;
    fx.srcStride = DIM;
    fx.dstStride = COLS;
    fx.ndNum = 1;
    Fixpipe<float, float, CFG_ROW_MAJOR>(x.scratch[scratchOffset], c, fx);
  }

  TPipe pipe_;
  TBuf<TPosition::A1> a1_;
  TBuf<TPosition::B1> b1_;
  TBuf<TPosition::A2> a2_;
  TBuf<TPosition::B2> b2_;
  TBuf<TPosition::CO1> c1_;
};
#endif

#ifdef __DAV_C220_VEC__
template <uint32_t PanelCols, bool SuffixHistory = false, bool BulkQkv = false,
          bool BatchHistory = false, bool TriangularSolve = false, bool FuseFold = false>
class Consumer {
  static constexpr uint32_t PANEL = TILE * PanelCols;
  static_assert(!BatchHistory || (PanelCols == 2 * TOKENS && SuffixHistory));
  static_assert(!TriangularSolve || (PanelCols == 2 * TOKENS && BulkQkv));
  static_assert(!FuseFold || (BatchHistory && !TriangularSolve));

 public:
  __aicore__ inline void Run(Inputs &x) {
    const uint32_t block = GetBlockIdx() / 2;
    const uint32_t sub = GetSubBlockIdx();
    const uint32_t vStart = sub * TILE;
    const uint32_t group = x.hv / x.hk;
    Init();
    const uint32_t end = x.EndTask(block);
    for (uint32_t task = x.FirstTask(block, group); task < end;) {
      const uint32_t row = task / x.hv;
      const uint32_t firstHead = task - row * x.hv;
      const uint32_t key = firstHead / group;
      const uint32_t groupEnd = x.GroupEnd(task, end, group);
      const int32_t slot = x.slots.GetValue(row);
      const int32_t ring = x.ringSlots.GetValue(row);
      const int32_t pos = x.pos.GetValue(row);
      const int32_t base = x.base.GetValue(row);
      uint32_t tokenStart = 0;
      const int32_t len = x.ReadWindow(row, tokenStart);
      int32_t error = x.tokenCount && len < 0 ? 6 : Validate(x, slot, ring, pos, base, len);
      bool fold = false;
      if constexpr (FuseFold) {
        const int32_t flush = x.flush.GetValue(row);
        if (error == 0 && (flush < 0 || flush > 1 || (slot == 0 && flush != 0))) {
          error = 6;
        }
        fold = error == 0 && slot != 0 && flush == 1 && pos > 0;
      }
      const bool valid = error == 0 && slot != 0;
      if (valid) {
        MakeDots(x, tokenStart, key, ring, pos, base, len);
      }
      for (uint32_t head = firstHead; head < firstHead + groupEnd - task; ++head) {
        if (valid && pos > 0) {
          const uint64_t dOffset = (static_cast<uint64_t>(ring) * x.hv + head) * RING * DIM;
          StageHistory(x.d[dOffset + vStart], base, pos, TILE);
        }
        CrossCoreWaitFlag(READY);
        if (valid) {
          // The previous head may still have Vector reads of proj_ or the
          // gate buffers in flight. Order the next DMA writes after them.
          Fence<HardEvent::V_MTE2>();
          const auto from = x.scratch[x.Projection(block, row, head) + vStart * COLS];
          if constexpr (PanelCols == COLS) {
            DataCopy(proj_.Get<float>(), from, PANEL);
          } else {
            // The Cube fractal is 16 columns wide. Only K0..3 and Q0..3
            // are used by AIV, so skip the zero half of each row during DMA.
            constexpr uint32_t bytes = PanelCols * sizeof(float);
            const DataCopyExtParams copy{TILE, bytes, (COLS - PanelCols) * sizeof(float), 0, 0};
            const DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(proj_.Get<float>(), from, copy, pad);
          }
          Fence<HardEvent::MTE2_V>(projectionReady_);
        }
        // After the read, the projection lives in this AIV's private UB.
        CrossCoreSetFlag<MIX_MODE, PIPE_MTE2>(CONSUMED);
        if (valid) {
          uint64_t stateTile = 0;
          if constexpr (FuseFold) {
            if (fold) {
              stateTile = x.Checkpoint(slot, head, vStart);
              // READY covers the AIC's checkpoint read for this head.
              auto dst = foldIn_.AllocTensor<bfloat16_t>();
              DataCopy(dst, x.checkpoint[stateTile], TILE * DIM);
              foldIn_.EnQue(dst);
            }
          }
          const float alpha = Correct(x, tokenStart, head, ring, pos, base, len, vStart);
          if constexpr (FuseFold) {
            if (fold) {
              FoldCheckpoint(x.checkpoint[stateTile], pos, alpha);
            }
          }
        }
        if (x.tokenCount == 0 && (!valid || len < TOKENS)) {
          ZeroTail(x, tokenStart, head, vStart, valid ? len : 0);
        }
        if (sub == 0) {
          StoreStatus(x, row * x.hv + head, error);
        }
      }
      // With head balancing, one key group can span two Cube blocks. Only
      // the segment containing its first V head owns candidate K writes.
      if (valid && sub == 0 && firstHead == key * group) {
        const uint64_t kRing = (static_cast<uint64_t>(ring) * x.hk + key) * RING;
        for (uint32_t t = 0; t < static_cast<uint32_t>(len); ++t) {
          StoreCast(x.keys[(kRing + Physical(base, pos + t)) * DIM], rhs_.Get<float>()[t * DIM], DIM);
        }
      }
      task = x.NextTask(task, groupEnd, group);
    }
    pipe_.ReleaseEventID<HardEvent::MTE2_V>(projectionReady_);
    pipe_.ReleaseEventID<HardEvent::MTE2_V>(historyReady_);
    pipe_.ReleaseEventID<HardEvent::MTE2_V>(candidatesReady_);
  }

 private:
  __aicore__ inline void Init() {
    pipe_.InitBuffer(in_, 1, (BulkQkv ? TOKENS : 1) * DIM * sizeof(bfloat16_t));
    pipe_.InitBuffer(out_, 1, DIM * sizeof(float));
    pipe_.InitBuffer(rhs_, PanelCols * DIM * sizeof(float));
    pipe_.InitBuffer(products_, PanelCols * DIM * sizeof(float));
    pipe_.InitBuffer(dot_, (HISTORY + TOKENS) * PanelCols * sizeof(float));
    pipe_.InitBuffer(proj_, PANEL * sizeof(float));
    pipe_.InitBuffer(outer_, PANEL * sizeof(float));
    pipe_.InitBuffer(gather_, 2 * TOKENS * TILE * sizeof(uint32_t));
    pipe_.InitBuffer(historyIn_, 1, HISTORY * DIM * sizeof(half));
    pipe_.InitBuffer(historyRows_, HISTORY * DIM * sizeof(float));
    pipe_.InitBuffer(delta_, TILE * sizeof(float));
    pipe_.InitBuffer(selected_, TILE * sizeof(float));
    pipe_.InitBuffer(gates_, 2 * RING * sizeof(float));
    pipe_.InitBuffer(candidate_, 3 * CANDIDATE_SCALARS * sizeof(float));
    if constexpr (BulkQkv) {
      pipe_.InitBuffer(values_, TOKENS * TILE * sizeof(float));
    }
    if constexpr (BatchHistory) {
      pipe_.InitBuffer(historyPanels_, HISTORY * PANEL * sizeof(float));
    }
    if constexpr (TriangularSolve) {
      pipe_.InitBuffer(solvedDeltas_, TOKENS * TILE * sizeof(float));
    }
    if constexpr (FuseFold) {
      pipe_.InitBuffer(historyKeys_, HISTORY * DIM * sizeof(float));
      pipe_.InitBuffer(foldState_, TILE * DIM * sizeof(float));
      pipe_.InitBuffer(foldIn_, 1, TILE * DIM * sizeof(bfloat16_t));
      pipe_.InitBuffer(foldOut_, 1, TILE * DIM * sizeof(bfloat16_t));
    }
    pipe_.InitBuffer(scalarOut_, 32);
    // MTE2 can enqueue gate copies while the V wait on the projection is
    // still outstanding. Reserve distinct token IDs for these dependencies.
    projectionReady_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE2_V>());
    historyReady_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE2_V>());
    candidatesReady_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE2_V>());
    auto offsets = gather_.Get<uint32_t>();
    for (uint32_t col = 0; col < 2 * TOKENS; ++col) {
      for (uint32_t v = 0; v < TILE; ++v) {
        offsets.SetValue(col * TILE + v, (v * PanelCols + col) * sizeof(float));
      }
    }
    Fence<HardEvent::S_V>();
  }

  __aicore__ inline int32_t Validate(const Inputs &x, int32_t slot, int32_t ring, int32_t pos,
                                     int32_t base, int32_t len) {
    if (slot < 0 || static_cast<uint32_t>(slot) >= x.numSlots) return 1;
    if (slot == 0) return (pos == 0 && base == 0 && len == 0) ? 0 : 4;
    if (ring < 0 || static_cast<uint32_t>(ring) >= x.numRingSlots) return 5;
    if (pos < 0 || pos > HISTORY || len < 1 || len > TOKENS) return 2;
    return (base < 0 || base >= RING) ? 3 : 0;
  }

  template <typename T>
  __aicore__ inline void LoadCast(LocalTensor<float> dst, GlobalTensor<T> src, uint32_t count) {
    auto in = in_.AllocTensor<T>();
    DataCopy(in, src, count);
    in_.EnQue(in);
    in = in_.DeQue<T>();
    Cast(dst, in, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    in_.FreeTensor(in);
  }

  __aicore__ inline void LoadCandidateRows(LocalTensor<float> dst, GlobalTensor<bfloat16_t> src,
                                          uint32_t rows, uint32_t width, uint32_t pitch) {
    auto in = in_.AllocTensor<bfloat16_t>();
    const DataCopyExtParams copy{static_cast<uint16_t>(rows), static_cast<uint32_t>(width * sizeof(bfloat16_t)),
                                 static_cast<uint32_t>((pitch - width) * sizeof(bfloat16_t)), 0, 0};
    const DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, 0};
    DataCopyPad(in, src, copy, pad);
    in_.EnQue(in);
    in = in_.DeQue<bfloat16_t>();
    Cast(dst, in, RoundMode::CAST_NONE, rows * width);
    PipeBarrier<PIPE_V>();
    in_.FreeTensor(in);
  }

  template <typename T>
  __aicore__ inline void StoreCast(GlobalTensor<T> dst, LocalTensor<float> src, uint32_t count) {
    auto out = out_.AllocTensor<T>();
    Cast(out, src, RoundMode::CAST_RINT, count);
    out_.EnQue(out);
    out = out_.DeQue<T>();
    DataCopy(dst, out, count);
    out_.FreeTensor(out);
  }

  template <typename T>
  __aicore__ inline void StoreScalar(GlobalTensor<T> dst, T value) {
    auto out = scalarOut_.Get<T>();
    out.SetValue(0, value);
    Fence<HardEvent::S_MTE3>();
    DataCopyExtParams p{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPad(dst, out, p);
    Fence<HardEvent::MTE3_S>();
  }

  __aicore__ inline void StoreStatus(Inputs &x, uint32_t index, int32_t value) {
    StoreScalar(x.status[index], value);
  }

  // src starts at physical row zero, already offset to this AIV's lane.
  // GM rows have DIM elements. Pack just width elements per row into UB.
  __aicore__ inline void StageHistory(GlobalTensor<half> src, uint32_t base, uint32_t pos, uint32_t width) {
    auto input = historyIn_.AllocTensor<half>();
    const uint32_t first = pos < RING - base ? pos : RING - base;
    DataCopyExtParams cp{static_cast<uint16_t>(first), static_cast<uint32_t>(width * sizeof(half)),
                         static_cast<uint32_t>((DIM - width) * sizeof(half)), 0, 0};
    DataCopyPadExtParams<half> pad{false, 0, 0, 0};
    DataCopyPad(input, src[base * DIM], cp, pad);
    if (pos > first) {
      cp.blockCount = static_cast<uint16_t>(pos - first);
      DataCopyPad(input[first * width], src, cp, pad);
    }
    historyIn_.EnQue(input);
  }

  __aicore__ inline void CastHistory(LocalTensor<float> dst, uint32_t count) {
    auto input = historyIn_.DeQue<half>();
    Cast(dst, input, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    historyIn_.FreeTensor(input);
  }

  __aicore__ inline void MakeDots(Inputs &x, uint32_t tokenStart, uint32_t key, uint32_t ring, uint32_t pos,
                                  uint32_t base, uint32_t len) {
    if (pos) {
      const uint64_t kOffset = (static_cast<uint64_t>(ring) * x.hk + key) * RING * DIM;
      StageHistory(x.keys[kOffset], base, pos, DIM);
    }
    auto rhs = rhs_.Get<float>();
    Duplicate(rhs, 0.0f, PanelCols * DIM);
    PipeBarrier<PIPE_V>();
    const uint64_t off = x.KeyInput(tokenStart, key);
    if constexpr (BulkQkv) {
      LoadCandidateRows(rhs, x.k[off], len, DIM, x.hk * DIM);
      LoadCandidateRows(rhs[TOKENS * DIM], x.q[off], len, DIM, x.hk * DIM);
    } else {
      for (uint32_t t = 0; t < len; ++t) {
        LoadCast(rhs[t * DIM], x.k[off + t * x.hk * DIM], DIM);
        LoadCast(rhs[(TOKENS + t) * DIM], x.q[off + t * x.hk * DIM], DIM);
      }
    }
    auto rows = historyRows_.Get<float>();
    if constexpr (FuseFold) {
      rows = historyKeys_.Get<float>();
    }
    if (pos) {
      CastHistory(rows, pos * DIM);
    }
    for (uint32_t i = 0; i < pos; ++i) {
      Dot(i, rows[i * DIM]);
    }
    for (uint32_t t = 0; t < len; ++t) {
      Dot(pos + t, rhs[t * DIM]);
    }
  }

  __aicore__ inline void Dot(uint32_t i, LocalTensor<float> k) {
    auto p = products_.Get<float>();
    auto rhs = rhs_.Get<float>();
    Mul(p, rhs, k, 64, PanelCols, {1, 1, 1, 16, 16, 0});
    Mul(p[64], rhs[64], k[64], 64, PanelCols, {1, 1, 1, 16, 16, 0});
    PipeBarrier<PIPE_V>();
    Add(p, p, p[64], 64, PanelCols, {1, 1, 1, 16, 16, 16});
    PipeBarrier<PIPE_V>();
    WholeReduceSum(dot_.Get<float>()[i * PanelCols], p, 64, PanelCols, 1, 1, 16);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void Outer(uint32_t i, LocalTensor<float> d) {
    auto outer = outer_.Get<float>();
    auto proj = proj_.Get<float>();
    const uint32_t dstShape[2] = {TILE, PanelCols};
    const uint32_t srcShape[2] = {TILE, 1};
    Broadcast<float, 2, 1>(outer, d, dstShape, srcShape);
    PipeBarrier<PIPE_V>();
    constexpr uint32_t rowStride = PanelCols / FLOAT_LANES_PER_BLOCK;
    Mul(outer, outer, dot_.Get<float>()[i * PanelCols], PanelCols, TILE, {1, 1, 1, rowStride, rowStride, 0});
    PipeBarrier<PIPE_V>();
    Add(proj, proj, outer, PANEL);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void CorrectHistoryPanel(LocalTensor<float> rows, uint32_t pos) {
    if (pos == 0) {
      return;
    }
    auto expanded = historyPanels_.Get<float>();
    const uint32_t depth = pos <= 4 ? 4 : pos <= 8 ? 8 : HISTORY;
    const uint32_t dstShape[2] = {pos * TILE, PanelCols};
    const uint32_t srcShape[2] = {pos * TILE, 1};
    Broadcast<float, 2, 1>(expanded, rows, dstShape, srcShape);
    PipeBarrier<PIPE_V>();
    constexpr uint32_t rowStride = PanelCols / FLOAT_LANES_PER_BLOCK;
    for (uint32_t i = 0; i < pos; ++i) {
      auto panel = expanded[i * PANEL];
      Mul(panel, panel, dot_.Get<float>()[i * PanelCols], PanelCols, TILE,
          {1, 1, 1, rowStride, rowStride, 0});
    }
    if (pos < depth) {
      Duplicate(expanded[pos * PANEL], 0.0f, (depth - pos) * PANEL);
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t step = depth / 2; step > 0; step /= 2) {
      Add(expanded, expanded, expanded[step * PANEL], step * PANEL);
      PipeBarrier<PIPE_V>();
    }
    auto proj = proj_.Get<float>();
    Add(proj, proj, expanded, PANEL);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void AddCandidate(LocalTensor<float> selected, LocalTensor<float> delta,
                                       float coefficient) {
    auto tmp = delta_.Get<float>();
    Muls(tmp, delta, coefficient, TILE);
    PipeBarrier<PIPE_V>();
    Add(selected, selected, tmp, TILE);
    PipeBarrier<PIPE_V>();
  }

  template <uint32_t Step>
  __aicore__ inline void SolveToken(Inputs &x, uint64_t output, uint64_t ringHead, uint32_t base,
                                    uint32_t pos, uint32_t vStart, float prefix, const float *kWeights,
                                    const float *qWeights, float beta, float logGate) {
    auto panel = outer_.Get<float>();
    auto deltas = solvedDeltas_.Get<float>();
    auto selected = selected_.Get<float>();
    auto delta = deltas[Step * TILE];

    Muls(selected, panel[Step * TILE], prefix, TILE);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t j = 0; j < Step; ++j) {
      AddCandidate(selected, deltas[j * TILE], kWeights[j]);
    }
    Sub(delta, values_.Get<float>()[Step * TILE], selected, TILE);
    PipeBarrier<PIPE_V>();
    Muls(delta, delta, beta, TILE);
    PipeBarrier<PIPE_V>();
    const uint32_t record = Physical(base, pos + Step);
    StoreCast(x.d[(ringHead + record) * DIM + vStart], delta, TILE);

    Muls(selected, panel[(TOKENS + Step) * TILE], prefix, TILE);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t j = 0; j <= Step; ++j) {
      AddCandidate(selected, deltas[j * TILE], qWeights[j]);
    }
    Muls(selected, selected, SCALE, TILE);
    PipeBarrier<PIPE_V>();
    StoreCast(x.out[output + Step * x.hv * DIM + vStart], selected, TILE);
    if (vStart == 0) {
      StoreScalar(x.gates[ringHead + record], logGate);
    }
  }

  __aicore__ inline void SolveCandidates(Inputs &x, uint64_t output, uint64_t ringHead, uint32_t base,
                                         uint32_t pos, uint32_t len, uint32_t vStart,
                                         LocalTensor<float> decay, LocalTensor<float> beta,
                                         LocalTensor<float> logGate) {
    // One transpose/gather replaces two per candidate. The projected panel
    // then stays read-only while the four FP32 candidate deltas are solved.
    Gather(outer_.Get<float>(), proj_.Get<float>(), gather_.Get<uint32_t>(), 0, PANEL);
    PipeBarrier<PIPE_V>();
    Fence<HardEvent::V_S>();
    const auto dot = dot_.Get<float>();
    float kWeights[TOKENS][TOKENS];
    float qWeights[TOKENS][TOKENS];
    float prefix[TOKENS];
    float product = 1.0f;
    for (uint32_t t = 0; t < len; ++t) {
      product *= decay.GetValue(t * FLOAT_LANES_PER_BLOCK);
      prefix[t] = product;
      float suffix = 1.0f;
      for (uint32_t j = t + 1; j > 0; --j) {
        const uint32_t prior = j - 1;
        const uint32_t offset = (pos + prior) * PanelCols;
        if (prior < t) {
          kWeights[t][prior] = suffix * dot.GetValue(offset + t);
        }
        qWeights[t][prior] = suffix * dot.GetValue(offset + TOKENS + t);
        suffix *= decay.GetValue(prior * FLOAT_LANES_PER_BLOCK);
      }
    }

    SolveToken<0>(x, output, ringHead, base, pos, vStart, prefix[0], kWeights[0], qWeights[0],
                  beta.GetValue(0), logGate.GetValue(0));
    if (len > 1) {
      SolveToken<1>(x, output, ringHead, base, pos, vStart, prefix[1], kWeights[1], qWeights[1],
                    beta.GetValue(FLOAT_LANES_PER_BLOCK), logGate.GetValue(FLOAT_LANES_PER_BLOCK));
    }
    if (len > 2) {
      SolveToken<2>(x, output, ringHead, base, pos, vStart, prefix[2], kWeights[2], qWeights[2],
                    beta.GetValue(2 * FLOAT_LANES_PER_BLOCK), logGate.GetValue(2 * FLOAT_LANES_PER_BLOCK));
    }
    if (len > 3) {
      SolveToken<3>(x, output, ringHead, base, pos, vStart, prefix[3], kWeights[3], qWeights[3],
                    beta.GetValue(3 * FLOAT_LANES_PER_BLOCK), logGate.GetValue(3 * FLOAT_LANES_PER_BLOCK));
    }
  }

  __aicore__ inline float Correct(Inputs &x, uint32_t tokenStart, uint32_t head, uint32_t ring, uint32_t pos,
                                 uint32_t base, uint32_t len, uint32_t vStart) {
    auto proj = proj_.Get<float>();
    auto d = delta_.Get<float>();
    auto selected = selected_.Get<float>();
    auto offsets = gather_.Get<uint32_t>();
    auto gates = gates_.Get<float>();
    auto logGate = candidate_.Get<float>();
    auto decay = logGate[CANDIDATE_SCALARS];
    auto beta = logGate[2 * CANDIDATE_SCALARS];
    const uint64_t ringHead = (static_cast<uint64_t>(ring) * x.hv + head) * RING;
    const uint32_t first = pos < RING - base ? pos : RING - base;
    // Read only committed G. The peer AIV can write candidate G later
    // without sharing a GM location with these reads.
    if (first) {
      LoadGates(gates, x.gates[ringHead + base], first);
    }
    if (pos > first) {
      LoadGates(gates[RING], x.gates[ringHead], pos - first);
    }
    LoadCandidates(x, x.Output(tokenStart, 0, head) / DIM, logGate, decay, beta, x.tokenCount ? len : TOKENS);
    if constexpr (BulkQkv) {
      LoadCandidateRows(values_.Get<float>(), x.v[x.Output(tokenStart, 0, head) + vStart], len, TILE, x.hv * DIM);
    }
    auto rows = historyRows_.Get<float>();
    if (pos) {
      CastHistory(rows, pos * TILE);
    }
    float historyDecay = 1.0f;
    if constexpr (SuffixHistory) {
      if (pos != 0) {
        // Move the per-history gate scales to D and scale the whole panel
        // once. This shortens the projected recurrence's dependency chain.
        float suffix = 1.0f;
        for (uint32_t n = pos; n > 0; --n) {
          const uint32_t i = n - 1;
          auto delta = rows[i * TILE];
          Muls(delta, delta, suffix, TILE);
          suffix *= gates.GetValue(i < first ? i : RING + i - first);
        }
        historyDecay = suffix;
        Muls(proj, proj, historyDecay, PANEL);
        PipeBarrier<PIPE_V>();
      }
    }
    if constexpr (BatchHistory) {
      CorrectHistoryPanel(rows, pos);
    } else {
      for (uint32_t i = 0; i < pos; ++i) {
        if constexpr (!SuffixHistory) {
          const uint32_t gateIndex = i < first ? i : RING + i - first;
          Muls(proj, proj, gates.GetValue(gateIndex), PANEL);
          PipeBarrier<PIPE_V>();
        }
        Outer(i, rows[i * TILE]);
      }
    }
    if constexpr (TriangularSolve) {
      SolveCandidates(x, x.Output(tokenStart, 0, head), ringHead, base, pos, len, vStart, decay, beta, logGate);
      return historyDecay;
    }
    for (uint32_t t = 0; t < len; ++t) {
      const uint64_t out = x.Output(tokenStart, t, head);
      const auto record = Physical(base, pos + t);
      const uint32_t lane = t * FLOAT_LANES_PER_BLOCK;
      Muls(proj, proj, decay.GetValue(lane), PANEL);
      if constexpr (!BulkQkv) {
        LoadCast(d, x.v[out + vStart], TILE);
      }
      Gather(selected, proj, offsets[t * TILE], 0, TILE);
      PipeBarrier<PIPE_V>();
      if constexpr (BulkQkv) {
        Sub(d, values_.Get<float>()[t * TILE], selected, TILE);
      } else {
        Sub(d, d, selected, TILE);
      }
      PipeBarrier<PIPE_V>();
      Muls(d, d, beta.GetValue(lane), TILE);
      PipeBarrier<PIPE_V>();
      StoreCast(x.d[(ringHead + record) * DIM + vStart], d, TILE);
      Outer(pos + t, d);
      Gather(selected, proj, offsets[(TOKENS + t) * TILE], 0, TILE);
      PipeBarrier<PIPE_V>();
      Muls(selected, selected, SCALE, TILE);
      PipeBarrier<PIPE_V>();
      StoreCast(x.out[out + vStart], selected, TILE);
      if (vStart == 0) {
        StoreScalar(x.gates[ringHead + record], logGate.GetValue(lane));
      }
    }
    return historyDecay;
  }

  __aicore__ inline void FoldCheckpoint(GlobalTensor<bfloat16_t> dst, uint32_t pos, float alpha) {
    auto state = foldState_.Get<float>();
    auto input = foldIn_.DeQue<bfloat16_t>();
    Cast(state, input, RoundMode::CAST_NONE, TILE * DIM);
    PipeBarrier<PIPE_V>();
    foldIn_.FreeTensor(input);

    auto products = historyPanels_.Get<float>();
    auto keys = historyKeys_.Get<float>();
    auto weightedD = historyRows_.Get<float>();
    constexpr uint32_t dstShape[2] = {TILE, DIM};
    constexpr uint32_t srcShape[2] = {TILE, 1};
    Muls(state, state, alpha, TILE * DIM);
    for (uint32_t i = 0; i < pos; ++i) {
      Broadcast<float, 2, 1>(products, weightedD[i * TILE], dstShape, srcShape);
      PipeBarrier<PIPE_V>();
      auto key = keys[i * DIM];
      Mul(products, products, key, 64, TILE, {1, 1, 1, 16, 16, 0});
      Mul(products[64], products[64], key[64], 64, TILE, {1, 1, 1, 16, 16, 0});
      PipeBarrier<PIPE_V>();
      Add(state, state, products, TILE * DIM);
      PipeBarrier<PIPE_V>();
    }

    auto output = foldOut_.AllocTensor<bfloat16_t>();
    Cast(output, state, RoundMode::CAST_RINT, TILE * DIM);
    foldOut_.EnQue(output);
    output = foldOut_.DeQue<bfloat16_t>();
    DataCopy(dst, output, TILE * DIM);
    foldOut_.FreeTensor(output);
  }

  __aicore__ inline void LoadCandidates(Inputs &x, uint64_t offset, LocalTensor<float> logGate,
                                        LocalTensor<float> decay, LocalTensor<float> beta, uint32_t len) {
    // One FP32 scalar per token, padded to eight FP32 lanes in UB.
    DataCopyExtParams p{static_cast<uint16_t>(len), sizeof(float), static_cast<uint32_t>((x.hv - 1) * sizeof(float)), 0, 0};
    DataCopyPadExtParams<float> padding{true, 0, FLOAT_LANES_PER_BLOCK - 1, 0.0f};
    DataCopyPad(logGate, x.g[offset], p, padding);
    DataCopyPad(beta, x.beta[offset], p, padding);
    Fence<HardEvent::MTE2_V>(candidatesReady_);
    Exp(decay, logGate, len * FLOAT_LANES_PER_BLOCK);
    Fence<HardEvent::V_S>();
  }

  __aicore__ inline void LoadGates(LocalTensor<float> dst, GlobalTensor<float> src, uint32_t count) {
    DataCopyExtParams p{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padding{true, 0, 0, 0.0f};
    DataCopyPad(dst, src, p, padding);
    Fence<HardEvent::MTE2_V>(historyReady_);
    Exp(dst, dst, count);
    Fence<HardEvent::V_S>();
  }

  __aicore__ inline void ZeroTail(Inputs &x, uint32_t tokenStart, uint32_t head, uint32_t vStart,
                                  uint32_t len) {
    auto zero = selected_.Get<float>();
    Duplicate(zero, 0.0f, TILE);
    PipeBarrier<PIPE_V>();
    for (uint32_t t = len; t < TOKENS; ++t) {
      StoreCast(x.out[x.Output(tokenStart, t, head) + vStart], zero, TILE);
    }
  }

  TPipe pipe_;
  TQue<TPosition::VECIN, 1> in_, historyIn_, foldIn_;
  TQue<TPosition::VECOUT, 1> out_, foldOut_;
  TBuf<TPosition::VECCALC> rhs_, products_, dot_, proj_, outer_, gather_, historyRows_;
  TBuf<TPosition::VECCALC> delta_, selected_, gates_, candidate_, values_;
  TBuf<TPosition::VECCALC> historyPanels_, solvedDeltas_, historyKeys_, foldState_;
  TBuf<TPosition::VECOUT> scalarOut_;
  event_t projectionReady_, historyReady_, candidatesReady_;
};
#endif
}  // namespace

extern "C" __global__ __aicore__ void replayssm_sd_verify_mix_910b(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR checkpoint,
    GM_ADDR d, GM_ADDR keys, GM_ADDR gates, GM_ADDR slots, GM_ADDR ringSlots,
    GM_ADDR pos, GM_ADDR base, GM_ADDR length, GM_ADDR out, GM_ADDR status, GM_ADDR zeros,
    GM_ADDR scratch, GM_ADDR flush, GM_ADDR tokenOffsets, uint32_t tokenCount,
    uint64_t checkpointStride, uint32_t batch, uint32_t hk, uint32_t hv, uint32_t numSlots,
    uint32_t numRingSlots, uint32_t blocks, uint32_t perHeadWorkspace, uint32_t panelMode, uint32_t balanceHeads) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  Inputs x;
  x.q.SetGlobalBuffer((__gm__ bfloat16_t *)q);
  x.k.SetGlobalBuffer((__gm__ bfloat16_t *)k);
  x.v.SetGlobalBuffer((__gm__ bfloat16_t *)v);
  x.g.SetGlobalBuffer((__gm__ float *)g);
  x.beta.SetGlobalBuffer((__gm__ float *)beta);
  x.checkpoint.SetGlobalBuffer((__gm__ bfloat16_t *)checkpoint);
  x.d.SetGlobalBuffer((__gm__ half *)d);
  x.keys.SetGlobalBuffer((__gm__ half *)keys);
  x.gates.SetGlobalBuffer((__gm__ float *)gates);
  x.slots.SetGlobalBuffer((__gm__ int32_t *)slots);
  x.ringSlots.SetGlobalBuffer((__gm__ int32_t *)ringSlots);
  x.pos.SetGlobalBuffer((__gm__ int32_t *)pos);
  x.base.SetGlobalBuffer((__gm__ int32_t *)base);
  x.length.SetGlobalBuffer((__gm__ int32_t *)length);
  x.flush.SetGlobalBuffer((__gm__ int32_t *)flush);
  x.tokenOffsets.SetGlobalBuffer((__gm__ int32_t *)tokenOffsets);
  x.out.SetGlobalBuffer((__gm__ bfloat16_t *)out);
  x.status.SetGlobalBuffer((__gm__ int32_t *)status);
  x.zeros.SetGlobalBuffer((__gm__ bfloat16_t *)zeros);
  x.scratch.SetGlobalBuffer((__gm__ float *)scratch);
  x.tokenCount = tokenCount;
  x.checkpointStride = checkpointStride;
  x.batch = batch;
  x.hk = hk;
  x.hv = hv;
  x.numSlots = numSlots;
  x.numRingSlots = numRingSlots;
  x.blocks = blocks;
  x.perHeadWorkspace = perHeadWorkspace != 0;
  x.balanceHeads = balanceHeads != 0;
#if defined(__DAV_C220_CUBE__)
  Projector op;
  op.Run(x);
#elif defined(__DAV_C220_VEC__)
  if (panelMode == 7) {
    Consumer<2 * TOKENS, true, true, true, false, true> op;
    op.Run(x);
  } else if (panelMode == 6) {
    Consumer<2 * TOKENS, true, true> op;
    op.Run(x);
  } else if (panelMode == 5) {
    Consumer<2 * TOKENS, false, true, false, true> op;
    op.Run(x);
  } else if (panelMode == 4) {
    Consumer<2 * TOKENS, true, true, true> op;
    op.Run(x);
  } else if (panelMode == 3) {
    Consumer<2 * TOKENS, false, true> op;
    op.Run(x);
  } else if (panelMode == 2) {
    Consumer<2 * TOKENS, true> op;
    op.Run(x);
  } else if (panelMode == 1) {
    Consumer<2 * TOKENS> op;
    op.Run(x);
  } else {
    Consumer<COLS> op;
    op.Run(x);
  }
#endif
}

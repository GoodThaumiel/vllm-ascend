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
constexpr uint32_t PANEL = TILE * COLS;
constexpr uint32_t FRACTAL = 16 * 16;
constexpr uint32_t FLOAT_LANES_PER_BLOCK = 32 / sizeof(float);
constexpr uint32_t CANDIDATE_SCALARS = TOKENS * FLOAT_LANES_PER_BLOCK;
constexpr uint32_t MIX_MODE = 2;
constexpr uint16_t READY = 6;
constexpr uint16_t CONSUMED = 7;
constexpr float SCALE = 0.08838834764831845f;

template <HardEvent Event>
__aicore__ inline void Fence() {
  const auto id = static_cast<event_t>(GetTPipePtr()->FetchEventID(Event));
  SetFlag<Event>(id);
  WaitFlag<Event>(id);
}

__aicore__ inline uint32_t Physical(uint32_t base, uint32_t p) {
  return (base + p) & (RING - 1);
}

struct Inputs {
  GlobalTensor<bfloat16_t> q, k, v, checkpoint, out, zeros;
  GlobalTensor<half> d, keys;
  GlobalTensor<float> g, beta, gates, scratch;
  GlobalTensor<int32_t> slots, ringSlots, pos, base, length, status;
  uint32_t batch, hk, hv, numSlots, numRingSlots, blocks;

  __aicore__ inline uint64_t KeyInput(uint32_t row, uint32_t key) const {
    return (static_cast<uint64_t>(row) * TOKENS * hk + key) * DIM;
  }
  __aicore__ inline uint64_t Output(uint32_t row, uint32_t t, uint32_t head) const {
    return ((static_cast<uint64_t>(row) * TOKENS + t) * hv + head) * DIM;
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

    for (uint32_t task = block; task < x.batch * x.hk; task += x.blocks) {
      const uint32_t row = task / x.hk;
      const uint32_t key = task % x.hk;
      const int32_t slot = x.slots.GetValue(row);
      const bool valid = slot > 0 && static_cast<uint32_t>(slot) < x.numSlots;
      if (valid) {
        LoadRhs(x, row, key);
      }
      for (uint32_t member = 0; member < group; ++member) {
        if (valid) {
          Project(x, (static_cast<uint64_t>(slot) * x.hv + key * group + member) * DIM * DIM);
        }
        // Pad/error rows have the same handshakes. The AIVs skip the read.
        CrossCoreSetFlag<MIX_MODE, PIPE_FIX>(READY);
        // One ack at the AIC after both AIVs have loaded their tiles.
        CrossCoreWaitFlag(CONSUMED);
      }
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

  __aicore__ inline void LoadRhs(Inputs &x, uint32_t row, uint32_t key) {
    // NZ for [N,K]. Each tile has 16 rows; K goes in 0..3 and Q in 4..7.
    auto b1 = b1_.Get<bfloat16_t>();
    DataCopy(b1, x.zeros, COLS * DIM);
    const auto off = x.KeyInput(row, key);
    ToL1(b1, x.k[off], TOKENS, x.hk * DIM, COLS);
    ToL1(b1[TOKENS * 16], x.q[off], TOKENS, x.hk * DIM, COLS);
    Fence<HardEvent::MTE2_MTE1>();
    LoadData2DParams p{};
    p.repeatTimes = DIM / 16;
    p.srcStride = 1;
    // Already NZ for [N,K], matching arch22 SFA's B load.
    LoadData(b2_.Get<bfloat16_t>(), b1, p);
  }

  __aicore__ inline void Project(Inputs &x, uint64_t off) {
    auto a1 = a1_.Get<bfloat16_t>();
    auto a2 = a2_.Get<bfloat16_t>();
    auto c = c1_.Get<float>();
    ToL1(a1, x.checkpoint[off], DIM, DIM, DIM);
    Fence<HardEvent::MTE2_MTE1>();
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
    Fixpipe<float, float, CFG_ROW_MAJOR>(x.scratch[GetBlockIdx() * DIM * COLS], c, fx);
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
class Consumer {
 public:
  __aicore__ inline void Run(Inputs &x) {
    const uint32_t block = GetBlockIdx() / 2;
    const uint32_t sub = GetSubBlockIdx();
    const uint32_t vStart = sub * TILE;
    const uint32_t group = x.hv / x.hk;
    Init();
    for (uint32_t task = block; task < x.batch * x.hk; task += x.blocks) {
      const uint32_t row = task / x.hk;
      const uint32_t key = task % x.hk;
      const int32_t slot = x.slots.GetValue(row);
      const int32_t ring = x.ringSlots.GetValue(row);
      const int32_t pos = x.pos.GetValue(row);
      const int32_t base = x.base.GetValue(row);
      const int32_t len = x.length.GetValue(row);
      const int32_t error = Validate(x, slot, ring, pos, base, len);
      const bool valid = error == 0 && slot != 0;
      if (valid) {
        MakeDots(x, row, key, ring, pos, base, len);
      }
      for (uint32_t member = 0; member < group; ++member) {
        const uint32_t head = key * group + member;
        CrossCoreWaitFlag(READY);
        if (valid) {
          // The previous head may still have Vector reads of proj_ or the
          // gate buffers in flight. Order the next DMA writes after them.
          Fence<HardEvent::V_MTE2>();
          DataCopy(proj_.Get<float>(), x.scratch[(block * DIM + vStart) * COLS], PANEL);
          Fence<HardEvent::MTE2_V>();
        }
        // After the read, the projection lives in this AIV's private UB.
        CrossCoreSetFlag<MIX_MODE, PIPE_MTE2>(CONSUMED);
        if (valid) {
          Correct(x, row, head, ring, pos, base, len, vStart);
        }
        if (!valid || len < TOKENS) {
          ZeroTail(x, row, head, vStart, valid ? len : 0);
        }
        if (sub == 0) {
          StoreStatus(x, row * x.hv + head, error);
        }
      }
      if (valid && sub == 0) {
        const uint64_t kRing = (static_cast<uint64_t>(ring) * x.hk + key) * RING;
        for (uint32_t t = 0; t < static_cast<uint32_t>(len); ++t) {
          StoreCast(x.keys[(kRing + Physical(base, pos + t)) * DIM], rhs_.Get<float>()[t * DIM], DIM);
        }
      }
    }
  }

 private:
  __aicore__ inline void Init() {
    pipe_.InitBuffer(in_, 1, DIM * sizeof(bfloat16_t));
    pipe_.InitBuffer(out_, 1, DIM * sizeof(float));
    pipe_.InitBuffer(rhs_, COLS * DIM * sizeof(float));
    pipe_.InitBuffer(products_, COLS * DIM * sizeof(float));
    pipe_.InitBuffer(dot_, (HISTORY + TOKENS) * COLS * sizeof(float));
    pipe_.InitBuffer(proj_, PANEL * sizeof(float));
    pipe_.InitBuffer(outer_, PANEL * sizeof(float));
    pipe_.InitBuffer(gather_, 2 * TOKENS * TILE * sizeof(uint32_t));
    pipe_.InitBuffer(key_, DIM * sizeof(float));
    pipe_.InitBuffer(delta_, TILE * sizeof(float));
    pipe_.InitBuffer(selected_, TILE * sizeof(float));
    pipe_.InitBuffer(gates_, 2 * RING * sizeof(float));
    pipe_.InitBuffer(candidate_, 3 * CANDIDATE_SCALARS * sizeof(float));
    pipe_.InitBuffer(scalarOut_, 32);
    auto offsets = gather_.Get<uint32_t>();
    for (uint32_t col = 0; col < 2 * TOKENS; ++col) {
      for (uint32_t v = 0; v < TILE; ++v) {
        offsets.SetValue(col * TILE + v, (v * COLS + col) * sizeof(float));
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

  __aicore__ inline void MakeDots(Inputs &x, uint32_t row, uint32_t key, uint32_t ring, uint32_t pos,
                                  uint32_t base, uint32_t len) {
    auto rhs = rhs_.Get<float>();
    Duplicate(rhs, 0.0f, COLS * DIM);
    PipeBarrier<PIPE_V>();
    const uint64_t off = x.KeyInput(row, key);
    for (uint32_t t = 0; t < len; ++t) {
      LoadCast(rhs[t * DIM], x.k[off + t * x.hk * DIM], DIM);
      LoadCast(rhs[(TOKENS + t) * DIM], x.q[off + t * x.hk * DIM], DIM);
    }
    const uint64_t kRing = (static_cast<uint64_t>(ring) * x.hk + key) * RING;
    auto keyRow = key_.Get<float>();
    for (uint32_t i = 0; i < pos; ++i) {
      LoadCast(keyRow, x.keys[(kRing + Physical(base, i)) * DIM], DIM);
      Dot(i, keyRow);
    }
    for (uint32_t t = 0; t < len; ++t) {
      Dot(pos + t, rhs[t * DIM]);
    }
  }

  __aicore__ inline void Dot(uint32_t i, LocalTensor<float> k) {
    auto p = products_.Get<float>();
    auto rhs = rhs_.Get<float>();
    Mul(p, rhs, k, 64, COLS, {1, 1, 1, 16, 16, 0});
    Mul(p[64], rhs[64], k[64], 64, COLS, {1, 1, 1, 16, 16, 0});
    PipeBarrier<PIPE_V>();
    Add(p, p, p[64], 64, COLS, {1, 1, 1, 16, 16, 16});
    PipeBarrier<PIPE_V>();
    WholeReduceSum(dot_.Get<float>()[i * COLS], p, 64, COLS, 1, 1, 16);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void Outer(uint32_t i, LocalTensor<float> d) {
    auto outer = outer_.Get<float>();
    auto proj = proj_.Get<float>();
    const uint32_t dstShape[2] = {TILE, COLS};
    const uint32_t srcShape[2] = {TILE, 1};
    Broadcast<float, 2, 1>(outer, d, dstShape, srcShape);
    PipeBarrier<PIPE_V>();
    Mul(outer, outer, dot_.Get<float>()[i * COLS], COLS, TILE, {1, 1, 1, 2, 2, 0});
    PipeBarrier<PIPE_V>();
    Add(proj, proj, outer, PANEL);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void Correct(Inputs &x, uint32_t row, uint32_t head, uint32_t ring, uint32_t pos,
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
    LoadCandidates(x, x.Output(row, 0, head) / DIM, logGate, decay, beta);
    for (uint32_t i = 0; i < pos; ++i) {
      const auto record = Physical(base, i);
      LoadCast(d, x.d[(ringHead + record) * DIM + vStart], TILE);
      const uint32_t gateIndex = i < first ? i : RING + i - first;
      Muls(proj, proj, gates.GetValue(gateIndex), PANEL);
      PipeBarrier<PIPE_V>();
      Outer(i, d);
    }
    for (uint32_t t = 0; t < len; ++t) {
      const uint64_t out = x.Output(row, t, head);
      const auto record = Physical(base, pos + t);
      const uint32_t lane = t * FLOAT_LANES_PER_BLOCK;
      Muls(proj, proj, decay.GetValue(lane), PANEL);
      LoadCast(d, x.v[out + vStart], TILE);
      Gather(selected, proj, offsets[t * TILE], 0, TILE);
      PipeBarrier<PIPE_V>();
      Sub(d, d, selected, TILE);
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
  }

  __aicore__ inline void LoadCandidates(Inputs &x, uint64_t offset, LocalTensor<float> logGate,
                                        LocalTensor<float> decay, LocalTensor<float> beta) {
    // One FP32 scalar per token, padded to eight FP32 lanes in UB.
    DataCopyExtParams p{TOKENS, sizeof(float), static_cast<uint32_t>((x.hv - 1) * sizeof(float)), 0, 0};
    DataCopyPadExtParams<float> padding{true, 0, FLOAT_LANES_PER_BLOCK - 1, 0.0f};
    DataCopyPad(logGate, x.g[offset], p, padding);
    DataCopyPad(beta, x.beta[offset], p, padding);
    Fence<HardEvent::MTE2_V>();
    Exp(decay, logGate, CANDIDATE_SCALARS);
    Fence<HardEvent::V_S>();
  }

  __aicore__ inline void LoadGates(LocalTensor<float> dst, GlobalTensor<float> src, uint32_t count) {
    DataCopyExtParams p{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padding{true, 0, 0, 0.0f};
    DataCopyPad(dst, src, p, padding);
    Fence<HardEvent::MTE2_V>();
    Exp(dst, dst, count);
    Fence<HardEvent::V_S>();
  }

  __aicore__ inline void ZeroTail(Inputs &x, uint32_t row, uint32_t head, uint32_t vStart,
                                  uint32_t len) {
    auto zero = selected_.Get<float>();
    Duplicate(zero, 0.0f, TILE);
    PipeBarrier<PIPE_V>();
    for (uint32_t t = len; t < TOKENS; ++t) {
      StoreCast(x.out[x.Output(row, t, head) + vStart], zero, TILE);
    }
  }

  TPipe pipe_;
  TQue<TPosition::VECIN, 1> in_;
  TQue<TPosition::VECOUT, 1> out_;
  TBuf<TPosition::VECCALC> rhs_, products_, dot_, proj_, outer_, gather_, key_, delta_, selected_, gates_, candidate_;
  TBuf<TPosition::VECOUT> scalarOut_;
};
#endif
}  // namespace

extern "C" __global__ __aicore__ void replayssm_sd_verify_mix_910b(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR checkpoint,
    GM_ADDR d, GM_ADDR keys, GM_ADDR gates, GM_ADDR slots, GM_ADDR ringSlots,
    GM_ADDR pos, GM_ADDR base, GM_ADDR length, GM_ADDR out, GM_ADDR status, GM_ADDR zeros,
    GM_ADDR scratch, uint32_t batch, uint32_t hk, uint32_t hv, uint32_t numSlots,
    uint32_t numRingSlots, uint32_t blocks) {
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
  x.out.SetGlobalBuffer((__gm__ bfloat16_t *)out);
  x.status.SetGlobalBuffer((__gm__ int32_t *)status);
  x.zeros.SetGlobalBuffer((__gm__ bfloat16_t *)zeros);
  x.scratch.SetGlobalBuffer((__gm__ float *)scratch);
  x.batch = batch;
  x.hk = hk;
  x.hv = hv;
  x.numSlots = numSlots;
  x.numRingSlots = numRingSlots;
  x.blocks = blocks;
#if defined(__DAV_C220_CUBE__)
  Projector op;
  op.Run(x);
#elif defined(__DAV_C220_VEC__)
  Consumer op;
  op.Run(x);
#endif
}

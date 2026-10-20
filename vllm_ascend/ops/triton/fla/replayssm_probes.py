# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Small Triton-Ascend feature probes used to diagnose ReplaySSM kernels."""

from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_element


@triton.jit
def cumsum_probe_kernel(x, out, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK).to(tl.int32)
    values = tl.load(x + offsets).to(tl.float32)
    tl.store(out + offsets, tl.cumsum(values, axis=0))


@triton.jit
def serial_prefix_probe_kernel(x, out, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK).to(tl.int32)
    values = tl.load(x + offsets).to(tl.float32)
    prefix = tl.zeros([BLOCK], dtype=tl.float32)
    running = 0.0
    for lane in range(BLOCK):
        running += get_element(values, (lane,))
        prefix = tl.where(offsets == lane, running, prefix)
    tl.store(out + offsets, prefix)


@triton.jit
def dot_probe_kernel(a, b, out, BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
    offsets_m = tl.arange(0, BM).to(tl.int32)
    offsets_k = tl.arange(0, BK).to(tl.int32)
    offsets_n = tl.arange(0, BN).to(tl.int32)
    a_tile = tl.load(a + offsets_m[:, None] * BK + offsets_k[None, :])
    b_tile = tl.load(b + offsets_k[:, None] * BN + offsets_n[None, :])
    result = tl.dot(a_tile, b_tile).to(tl.float32)
    tl.store(out + offsets_m[:, None] * BN + offsets_n[None, :], result)


@triton.jit
def masked_zero_dot_probe_kernel(
    a,
    b,
    out,
    valid_length,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    offsets_m = tl.arange(0, BM).to(tl.int32)
    offsets_k = tl.arange(0, BK).to(tl.int32)
    offsets_n = tl.arange(0, BN).to(tl.int32)
    active_k = offsets_k < tl.load(valid_length).to(tl.int32)
    a_tile = tl.load(
        a + offsets_m[:, None] * BK + offsets_k[None, :],
        mask=active_k[None, :],
        other=0.0,
    )
    b_tile = tl.load(
        b + offsets_k[:, None] * BN + offsets_n[None, :],
        mask=active_k[:, None],
        other=0.0,
    )
    result = tl.dot(a_tile, b_tile).to(tl.float32)
    tl.store(out + offsets_m[:, None] * BN + offsets_n[None, :], result)


@triton.jit
def runtime_branch_probe_kernel(x, flag, out, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK).to(tl.int32)
    values = tl.load(x + offsets).to(tl.float32)
    if tl.load(flag).to(tl.int32) != 0:
        values += 1.0
    else:
        values -= 1.0
    tl.store(out + offsets, values)


@triton.jit
def masked_2d_io_probe_kernel(
    x,
    out,
    rows,
    cols,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    offsets_m = tl.arange(0, BM).to(tl.int32)
    offsets_n = tl.arange(0, BN).to(tl.int32)
    mask = (offsets_m[:, None] < rows) & (offsets_n[None, :] < cols)
    values = tl.load(x + offsets_m[:, None] * BN + offsets_n[None, :], mask=mask, other=0.0)
    tl.store(out + offsets_m[:, None] * BN + offsets_n[None, :], values, mask=mask)

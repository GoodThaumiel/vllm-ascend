# SPDX-License-Identifier: Apache-2.0
"""910B eager/graph demo, uniform MTP window and statically known active rows."""

import torch
import torch.nn.functional as F

DIM = 128
PANEL = 16


def panel(case, t, grouped=False):
    # [B,HK,K,T] -> [B,HV,K,16]. Keep K then Q, share across three HV.
    columns = torch.cat((case["k"][:, :t], case["q"][:, :t]), dim=1).permute(0, 2, 3, 1)
    rhs = F.pad(columns, (0, PANEL - 2 * t))
    if not grouped:
        rhs = rhs.repeat_interleave(3, dim=1)
    return rhs.flatten(0, 1).contiguous()


def history(case, pos, base):
    # Narrow contiguous ring spans; each call accounts for wrap/pack cost.
    def gather(name):
        x = case[name][1:]
        first = min(pos, 32 - base)
        a = x[:, :, base : base + first]
        return torch.cat((a, x[:, :, : pos - first]), dim=2) if first < pos else a

    g = gather("g_cache").float()
    total = g.sum(2)
    suffix = (total.unsqueeze(2) - g.cumsum(2)).exp()
    d = gather("d_cache").float() * suffix.unsqueeze(-1)
    return total.exp(), d, gather("k_cache").repeat_interleave(3, dim=1)


def checkpoint_projection(state, rhs):
    # Grouped RHS folds three contiguous value heads into GEMM M=384.
    projected = torch.bmm(state.reshape(rhs.shape[0], -1, DIM), rhs)
    return projected.view(*state.shape[:2], DIM, PANEL).float()


def verify(case, t, pos, base):
    state = case["checkpoint"][5:]
    rhs = panel(case, t, grouped=True)
    y = checkpoint_projection(state, rhs)
    cols = rhs.view(state.shape[0], -1, DIM, PANEL).repeat_interleave(3, dim=1).float()
    if pos:
        alpha, d, k = history(case, pos, base)
        # FP32 dot and rank updates in this functional demo.
        y = y * alpha[..., None, None]
        for i in range(pos):
            dot = (k[:, :, i, :, None].float() * cols).sum(2)
            y = y + d[:, :, i, :, None] * dot.unsqueeze(2)
    for s in range(t):
        kh = case["k"][:, s]
        dot = (kh.repeat_interleave(3, dim=1).float().unsqueeze(-1) * cols).sum(2)
        y = y * case["g"][:, s].exp()[..., None, None]
        delta = (case["v"][:, s].float() - y[..., s]) * case["beta"][:, s, :, None]
        y = y + delta.unsqueeze(-1) * dot.unsqueeze(2)
        case["out"][:, s].copy_(y[..., t + s] * DIM**-0.5)
        physical = (base + pos + s) % 32
        case["d_cache"][1:, :, physical].copy_(delta)
        case["k_cache"][1:, :, physical].copy_(kh)
        case["g_cache"][1:, :, physical].copy_(case["g"][:, s])
    case["out"][:, t:].zero_()
    case["status"].zero_()


def compact_inputs(case, active, pos, base):
    # Controlled upper-potential experiment: flush rows form a known prefix.
    selected = {name: case[name][: active + 1] for name in ("d_cache", "k_cache", "g_cache")}
    alpha, d, k = history(selected, pos, base)
    lhs = F.pad(d.half().transpose(2, 3), (0, PANEL - pos)).flatten(0, 1).contiguous()
    rhs = F.pad(k, (0, 0, 0, PANEL - pos)).flatten(0, 1).contiguous()
    return alpha, lhs, rhs


def compact(case, active, pos, base, status):
    if active and pos:
        alpha, lhs, rhs = compact_inputs(case, active, pos, base)
        state = case["checkpoint"][5 : 5 + active]
        update = torch.bmm(lhs, rhs).view_as(state).float()
        state.copy_(state.float() * alpha[..., None, None] + update)
    status.zero_()


def cube_fold_operands(case, ring_ids, physical, position):
    # Keep the ring gather, suffix scaling and [V, history] packing in the
    # timed path. The historical GEMM uses FP16 inputs and an FP16 result.
    def gather(name):
        return case[name].index_select(0, ring_ids).index_select(2, physical)

    g = gather("g_cache")
    total = g.sum(dim=2)
    scale = (total.unsqueeze(2) - g.cumsum(dim=2)).exp()
    weighted_d = gather("d_cache").float() * scale.unsqueeze(-1)
    k = gather("k_cache")
    group = weighted_d.size(1) // k.size(1)
    k = k.repeat_interleave(group, dim=1)
    lhs = F.pad(weighted_d.half().transpose(2, 3), (0, PANEL - position)).flatten(0, 1).contiguous()
    rhs = F.pad(k, (0, 0, 0, PANEL - position)).flatten(0, 1).contiguous()
    return total.exp(), lhs, rhs

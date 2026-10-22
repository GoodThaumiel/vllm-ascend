# SPDX-License-Identifier: Apache-2.0
"""CPU checks for server-shaped native/Replay operator profile inputs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/replayssm/profile_ar_operator_compare.py"
SPEC = importlib.util.spec_from_file_location("profile_ar_operator_compare", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


def test_tp4_layout_matches_observed_server_page_stride():
    conv_bytes, page_bytes, regions = profile.layout_geometry(4, 16)
    assert conv_bytes == 15360
    assert page_bytes == 524288
    assert regions == {
        "checkpoint": 393216,
        "d_cache": 49152,
        "k_cache": 16384,
        "g_cache": 768,
    }
    views, _ = profile.make_page_views(device=torch.device("cpu"), num_slots=2, tp=4, capacity=16, page_bytes=0)
    assert views["checkpoint"].stride(0) == 262144
    assert views["checkpoint"].storage_offset() == 7680
    assert views["d_cache"].storage_offset() == 204288
    assert views["g_cache"].storage_offset() == 118528
    assert len({view.untyped_storage().data_ptr() for view in views.values()}) == 1


def test_tp8_layout_halves_page_size():
    conv_bytes, page_bytes, _ = profile.layout_geometry(8, 16)
    assert conv_bytes == 7680
    assert page_bytes == 262144


def test_reconstruction_maps_replay_history_to_native_state():
    import sys

    sys.path.insert(0, str(ROOT / "experimental/replayssm_ar_ascendc"))
    from lab import make_case

    case = make_case(tp=4, batch=1, capacity=16, position=1, flush=False)
    case["checkpoint"][1].zero_()
    case["d_cache"][1, :, 0].fill_(1)
    case["k_cache"][1, :, 0].fill_(2)
    case["g_cache"][1, :, 0].zero_()
    state = profile.reconstruct_state(case, 1)
    assert state.shape == (1, 12, 128, 128)
    assert torch.all(state == 2)

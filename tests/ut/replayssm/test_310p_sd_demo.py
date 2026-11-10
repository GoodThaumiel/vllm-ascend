# SPDX-License-Identifier: Apache-2.0
"""CPU checks for 310P request-owned ring capacity and lifecycle."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch


def _load_cpu_modules():
    root = Path(__file__).resolve().parents[3] / "vllm_ascend"
    packages = (
        "vllm_ascend",
        "vllm_ascend.ops",
        "vllm_ascend.ops.replayssm",
        "vllm_ascend._310p",
        "vllm_ascend._310p.ops",
    )
    modules = (
        ("vllm_ascend.ops.replayssm.cache", root / "ops/replayssm/cache.py"),
        ("vllm_ascend._310p.ops.replayssm_sd", root / "_310p/ops/replayssm_sd.py"),
        ("vllm_ascend._310p.ops.replayssm_sd_demo", root / "_310p/ops/replayssm_sd_demo.py"),
    )
    names = (*packages, *(name for name, _ in modules))
    saved = {name: sys.modules.get(name) for name in names}
    try:
        for name in packages:
            package = ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package
        for name, path in modules:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return tuple(sys.modules[name] for name, _ in modules)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


_cache, _serve, _demo = _load_cpu_modules()
ReplaySSM310PGeometry = _demo.ReplaySSM310PGeometry
ReplaySSM310PRingDemo = _demo.ReplaySSM310PRingDemo
estimate_memory = _demo.estimate_memory
geometry_from_hf_config = _demo.geometry_from_hf_config


class TestReplaySSM310PDemo(unittest.TestCase):
    def test_capacity_and_allocation(self):
        geometry = ReplaySSM310PGeometry(1, 8, 16, 2, 3)
        demo = ReplaySSM310PRingDemo(geometry)
        report = estimate_memory(geometry, active_requests=2)
        self.assertEqual(geometry.ring_len, 32)
        self.assertEqual(demo.allocated_bytes(demo.allocate_rings()), report["ring_bytes_allocated"])

    def test_slot_reuse_and_no_active_alias(self):
        geometry = ReplaySSM310PGeometry(1, 8, 16, 2, 5)
        demo = ReplaySSM310PRingDemo(geometry)
        self.assertEqual(demo.update(["a", "b"]), {"ring_slots": [1, 2], "do_reset": [1, 1]})
        self.assertEqual(demo.update(["b", "c"], ["a"]), {"ring_slots": [2, 1], "do_reset": [0, 1]})
        with self.assertRaises(RuntimeError):
            demo.update(["b", "c", "d"])

    def test_projection_uses_fixed_capacity(self):
        geometry = ReplaySSM310PGeometry(30, 8, 16, 32, 3)
        at_one = estimate_memory(geometry, active_requests=1)
        at_full = estimate_memory(geometry, active_requests=32)
        self.assertEqual(at_one["ring_bytes_allocated"], at_full["ring_bytes_allocated"])
        self.assertLess(at_one["projected_saved_bytes"], 0)
        self.assertGreater(at_full["projected_saved_bytes"], 0)

    def test_hf_config_geometry(self):
        config = {
            "text_config": {
                "layer_types": ["linear_attention", "full_attention", "linear_attention"],
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
            }
        }
        geometry = geometry_from_hf_config(config, tp=2, max_num_seqs=32, mtp_tokens=5)
        self.assertEqual((geometry.linear_layers, geometry.local_key_heads, geometry.local_value_heads), (2, 8, 16))
        with self.assertRaises(ValueError):
            geometry_from_hf_config(config, tp=3, max_num_seqs=32, mtp_tokens=5)


class TestReplaySSM310PServe(unittest.TestCase):
    def test_metadata_backing_covers_last_m200_block(self):
        # A 64-request service uses physical slot 64 plus reserved slot 0.
        for dtype in (torch.int32, torch.int8):
            elem_size = torch.empty(0, dtype=dtype).element_size()
            backing = torch.empty(_serve.aligned_metadata_size(65, elem_size), dtype=dtype)
            view = backing[:65]
            block_elems = 32 // elem_size
            last_block_end = (64 // block_elems + 1) * block_elems
            self.assertGreaterEqual(view.untyped_storage().nbytes(), last_block_end * elem_size)
            self.assertEqual(view.numel(), 65)

    def test_verify_dispatch_for_full_windows(self):
        self._check_verify_dispatch((4, 4))

    def test_verify_dispatch_for_short_windows(self):
        self._check_verify_dispatch((4, 1, 2))

    def _check_verify_dispatch(self, lengths):
        rows = len(lengths)
        tokens = sum(lengths)
        lengths = torch.tensor(lengths, dtype=torch.int32)
        pack = _cache.replayssm_sd_pack_indices(lengths, 4)
        v = torch.arange(tokens, dtype=torch.float16).view(1, tokens, 1, 1).expand(1, tokens, 16, 128)
        q = torch.ones(1, tokens, 8, 128, dtype=torch.float16)
        gates = torch.ones(1, tokens, 16)
        slots = torch.arange(1, rows + 1, dtype=torch.int32)
        ring_slots = slots.flip(0)
        zero = torch.zeros(rows, dtype=torch.int32)
        meta = SimpleNamespace(
            replayssm_sd_block_slots=slots,
            replayssm_sd_ring_slots=ring_slots,
            replayssm_sd_positions=zero,
            replayssm_sd_bases=zero,
            replayssm_sd_flush=zero,
            replayssm_sd_lengths=lengths,
            replayssm_sd_pack_indices=pack,
        )
        if tokens != rows * 4:
            meta.replayssm_sd_token_gather = _serve.replay_token_gather(pack, rows)
        checkpoint = torch.empty(rows + 1, 16, 128, 128, dtype=torch.float16)
        layer = SimpleNamespace(
            head_k_dim=128,
            kv_cache=(None, checkpoint),
            _replayssm_sd_d_ring=torch.empty(0),
            _replayssm_sd_k_ring=torch.empty(0),
            _replayssm_sd_g_ring=torch.empty(0),
            _replayssm_sd_verify_status=torch.empty(rows * 16 * 2, dtype=torch.int32),
            _replayssm_sd_compact_status=torch.empty(rows * 16 * 4, dtype=torch.int32),
        )
        ops = Mock()

        def fake_verify(*args):
            self.assertEqual(args[0].shape, (rows, 4, 8, 128))
            self.assertEqual(args[3].dtype, torch.float32)
            self.assertEqual(args[4].dtype, torch.float16)
            self.assertIs(args[5], checkpoint)
            self.assertIs(args[9], slots)
            self.assertEqual(args[14].numel(), rows * 16 * 2)
            self.assertEqual(args[16:18], (64, 7))
            self.assertIs(args[18], ring_slots)
            self.assertEqual(args[19], 1)
            args[13].copy_(args[2] + 1)

        ops.verify_.side_effect = fake_verify
        with patch.object(torch.ops, "replayssm_sd_310p", ops):
            output = _serve.replayssm_sd_310p_decode(layer, q, q, v, gates, gates, meta)

        torch.testing.assert_close(output, v + 1)
        self.assertEqual([call[0] for call in ops.mock_calls], ["verify_", "compact_"])
        compact = ops.compact_.call_args.args
        self.assertIs(compact[0], checkpoint)
        self.assertIs(compact[4], slots)
        self.assertEqual(compact[8].numel(), rows * 16 * 4)
        self.assertEqual(compact[9:11], (32, 7))
        self.assertIs(compact[11], ring_slots)


if __name__ == "__main__":
    unittest.main()

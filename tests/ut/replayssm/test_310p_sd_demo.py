# SPDX-License-Identifier: Apache-2.0
"""CPU checks for 310P request-owned ring capacity and lifecycle."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType


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
        return sys.modules["vllm_ascend._310p.ops.replayssm_sd_demo"]
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


_demo = _load_cpu_modules()
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


if __name__ == "__main__":
    unittest.main()

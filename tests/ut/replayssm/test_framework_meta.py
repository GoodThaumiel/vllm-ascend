# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Run with the optional CPU-only C++ test library path as the sole argument."""

import importlib.util
import sys
import unittest
from pathlib import Path


class FrameworkMetaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if len(sys.argv) != 2 or not Path(sys.argv[1]).is_file():
            raise unittest.SkipTest("build and pass the CPU-only Meta library to run these checks")
        import torch

        torch.ops.load_library(sys.argv[1])
        cls.torch = torch
        root = Path(__file__).resolve().parents[3]
        spec = importlib.util.spec_from_file_location(
            "replay_ar_wrapper", root / "vllm_ascend/ops/replayssm/ascendc.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.wrapper = staticmethod(module.replayssm_ar_step)

    def arguments(self, device):
        t = self.torch
        tensor = lambda shape, dtype: t.empty(shape, dtype=dtype, device=device)
        bf, fp, half, integer = t.bfloat16, t.float32, t.float16, t.int32
        return [
            tensor((2, 4, 128), bf),
            tensor((2, 4, 128), bf),
            tensor((2, 12, 128), bf),
            tensor((2, 12), fp),
            tensor((2, 12), fp),
            tensor((4, 12, 128, 128), bf),
            tensor((4, 12, 16, 128), half),
            tensor((4, 4, 16, 128), half),
            tensor((4, 12, 16), fp),
            tensor((2,), integer),
            tensor((2,), integer),
            tensor((2,), integer),
            tensor((2, 12, 128), bf),
            tensor((96,), integer),
            128**-0.5,
            32,
            8,
        ]

    def test_cpp_schema_declares_all_six_mutations(self):
        schema = self.torch.ops._C_ascend.replayssm_ar_step_.default._schema
        self.assertEqual(
            [i for i, arg in enumerate(schema.arguments) if arg.alias_info and arg.alias_info.is_write],
            [5, 6, 7, 8, 12, 13],
        )
        self.assertEqual(len(schema.returns), 0)

    def test_meta_dispatch_has_no_device_allocation_or_output(self):
        args = self.arguments("meta")
        self.assertIsNone(self.wrapper(*args))

    def test_fake_tensor_tracing_retains_the_mutating_operator(self):
        from torch.fx.experimental.proxy_tensor import make_fx

        args = self.arguments("cpu")

        def forward(*values):
            self.wrapper(*values)
            return values[12]

        graph = make_fx(forward, tracing_mode="fake")(*args)
        self.assertTrue(
            any(node.target == self.torch.ops._C_ascend.replayssm_ar_step_.default for node in graph.graph.nodes)
        )


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FrameworkMetaTests)
    raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)

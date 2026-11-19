# SPDX-License-Identifier: Apache-2.0
import unittest
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[3]


class ComputeWy310DispatchTest(unittest.TestCase):
    def setUp(self):
        self.envs = SimpleNamespace(VLLM_ASCEND_310P_COMPUTE_WY_LIBRARY=None)
        logger = SimpleNamespace(logger=Mock())
        with patch.dict("sys.modules", {"vllm.logger": logger, "vllm_ascend": SimpleNamespace(envs=self.envs)}):
            self.api = run_path(str(ROOT / "vllm_ascend/_310p/ops/compute_wy.py"))

    def test_opt_in_library_must_register_the_expected_operator(self):
        load = self.api["load_compute_wy_library"]
        self.assertIsNone(load())
        self.envs.VLLM_ASCEND_310P_COMPUTE_WY_LIBRARY = "relative.so"
        with self.assertRaisesRegex(ValueError, "absolute path"):
            load()
        with TemporaryDirectory() as directory:
            library = Path(directory) / "compute_wy.so"
            library.touch()
            loader = Mock()
            ops = SimpleNamespace(load_library=loader, replayssm_310p_compute_wy=SimpleNamespace())
            with patch.object(torch, "ops", ops):
                self.envs.VLLM_ASCEND_310P_COMPUTE_WY_LIBRARY = str(library)
                with self.assertRaisesRegex(RuntimeError, "did not register"):
                    load()
                ops.replayssm_310p_compute_wy.prepare = Mock()
                self.assertEqual(load(), library.resolve())
                loader.assert_called_with(str(library.resolve()))

    def test_dispatches_production_prefill_shape_and_preserves_fallback(self):
        def tensor(shape, dtype=torch.float16):
            return SimpleNamespace(shape=shape, dtype=dtype, device=SimpleNamespace(type="npu"), contiguous=Mock())

        q = tensor((1, 8192, 8, 128))
        v = tensor((1, 8192, 16, 128))
        g = tensor((1, 8192, 16), torch.float32)
        beta = tensor((1, 8192, 16))
        dispatch = self.api["compute_wy_for_prefill"]
        op = Mock()
        with patch.object(torch, "ops", SimpleNamespace(replayssm_310p_compute_wy=SimpleNamespace(prepare=op))):
            self.assertIsNone(dispatch(q, q, v, g, beta, 64))
            op.assert_not_called()
            self.envs.VLLM_ASCEND_310P_COMPUTE_WY_LIBRARY = "/compute_wy.so"
            self.assertIs(dispatch(q, q, v, g, beta, 64), op.return_value)
            op.assert_called_once_with(
                q.contiguous.return_value,
                q.contiguous.return_value,
                v.contiguous.return_value,
                g.contiguous.return_value,
                beta.contiguous.return_value,
                64,
            )
            op.reset_mock()
            self.assertIsNone(dispatch(q, q, v, g, beta, 32))
            self.assertIsNone(dispatch(q, q, v, g, tensor(beta.shape, torch.float32), 64))
            op.assert_not_called()

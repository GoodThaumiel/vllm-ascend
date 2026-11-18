# SPDX-License-Identifier: Apache-2.0
import unittest
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = run_path(str(ROOT / "vllm_ascend/_310p/dflash_config.py"))
RGDR = run_path(str(ROOT / "vllm_ascend/_310p/ops/rgdr_grouped.py"))


class DFlash310PServiceContractTest(unittest.TestCase):
    def test_mixed_layer_bypass_is_scoped_and_retains_each_layer(self):
        resolve = CONTRACT["dflash_layer_attention"]
        loading = CONTRACT["DFLASH_MRV1_LOADING"]
        config = SimpleNamespace(
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=4096,
            dflash_config={"swa_window_size": 2048},
        )

        def upstream(config, index):
            return "upstream", index

        def causal(config, index):
            return config.layer_types[index] == "sliding_attention"

        self.assertEqual(resolve(config, 0, upstream, causal), ("upstream", 0))
        token = loading.set(True)
        try:
            self.assertEqual(resolve(config, 0, upstream, causal), (2048, True))
            self.assertEqual(resolve(config, 1, upstream, causal), (None, False))
            config.layer_types = ["full_attention"] * 2
            self.assertEqual(resolve(config, 0, upstream, causal), ("upstream", 0))
        finally:
            loading.reset(token)

    def test_sparse_exact_buckets_do_not_require_all_request_counts(self):
        selected = CONTRACT["usable_decode_sizes"]([1, 8, 64, 384, 392], 48, 8192, 8)
        self.assertEqual(selected, {8, 64, 384})
        self.assertNotIn(16, selected)

    def test_exact_capture_sizes_respect_token_budget(self):
        sizes = CONTRACT["usable_decode_sizes"]
        self.assertEqual(sizes([8, 192, 384], 48, 256, 8), {8, 192})
        self.assertFalse(sizes([1, 4, 392], 48, 8192, 8))

    def test_all_k_use_the_live_per_row_lanes_and_exact_capture_sizes(self):
        sizes = CONTRACT["usable_decode_sizes"]
        pack = CONTRACT["pack_query_lanes"]
        lanes = CONTRACT["DFLASH_QUERY_DMA_LANES"]
        for k in range(1, 8):
            with self.subTest(k=k):
                width = k + 1
                positions = torch.full((2, lanes), -999)
                positions[:, :width] = torch.arange(2 * width).reshape(2, width)
                self.assertEqual(pack(positions, width).tolist(), list(range(2 * width)))
                samples = torch.full((2, lanes), -999)
                samples[:, :k] = positions[:, 1:width]
                self.assertEqual(pack(samples, k).tolist(), [*range(1, width), *range(width + 1, 2 * width)])
                capture_sizes = [width, 3 * width - 1, 3 * width, 5 * width]
                self.assertEqual(sizes(capture_sizes, 4, 8192, width), {width, 3 * width})

    def test_grouped_rgdr_selects_validated_probe_and_retains_the_old_mtp_abi(self):
        configure = RGDR["grouped_rgdr_launch_config"]
        for fields, expected in ((["row_offsets", "probe"], (8, 11)), (["row_offsets"], (7, None))):
            configure.cache_clear()
            schema = SimpleNamespace(arguments=[SimpleNamespace(name=name) for name in fields])
            dispatch = SimpleNamespace(default=SimpleNamespace(_schema=schema))
            namespace = SimpleNamespace(rgdr_v310_cube_fp16=SimpleNamespace(run_=dispatch))
            with patch.object(torch, "ops", namespace):
                self.assertEqual(configure(), expected)
        configure.cache_clear()
        with (
            patch.object(torch, "ops", SimpleNamespace(rgdr_v310_cube_fp16=SimpleNamespace())),
            self.assertRaisesRegex(RuntimeError, "register run_"),
        ):
            configure()

    def test_crop_keeps_a_boundary_page_for_first_query(self):
        retained = CONTRACT["retained_page_range"]
        self.assertEqual(retained(4096 + 128 + 8 - 2, 8, 128, 4096), (0, 34))
        self.assertEqual(retained(4096 + 128 + 8 - 1, 8, 128, 4096), (1, 34))
        self.assertEqual(retained(4096 + 128 + 8 - 1, 8, 128, None), (0, 34))
        with self.assertRaisesRegex(ValueError, "shorter than"):
            retained(7, 8, 128, 4096)

    def test_context_can_exceed_draft_descriptor_but_cannot_exceed_its_own_storage(self):
        check = CONTRACT["validate_packed_context"]
        check([0, 2_000, 2_008], 2, 2_008)
        for offsets in ([0, 2000, 2010], [0, 2008, 2000], [0, 0, 2008], [1, 2000, 2008], [0, 2008]):
            with self.subTest(offsets=offsets), self.assertRaisesRegex(ValueError, "packed context bounds"):
                check(offsets, 2, 2_008)


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
import unittest
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = run_path(str(ROOT / "vllm_ascend/_310p/dflash_config.py"))


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

# SPDX-License-Identifier: Apache-2.0
"""Local checks for the serve report's queue metric parser."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("validate_sd_serve", ROOT / "scripts/replayssm/validate_sd_serve.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SDServeValidatorTests(unittest.TestCase):
    def test_prompt_jsonl_rejects_missing_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text(json.dumps({"prompt": "one"}) + "\n")
            self.assertEqual(MODULE.read_prompt_jsonl(path), ["one"])
            path.write_text(json.dumps({"text": "two"}) + "\n")
            with self.assertRaisesRegex(ValueError, "nonempty prompt"):
                MODULE.read_prompt_jsonl(path)

    def test_first_divergence_records_margin_and_token_ids(self) -> None:
        native = {"token_ids": [1, 2], "logprobs": {"top_logprobs": [{}, {"a": -0.1, "b": -0.2}]}}
        replay = {"token_ids": [1, 3], "logprobs": {"top_logprobs": [{}, {"a": -0.3, "b": -0.4}]}}
        self.assertEqual(
            MODULE.first_divergence(native, replay),
            {
                "index": 1,
                "native_token_id": 2,
                "replay_token_id": 3,
                "native_top_logprob_margin": 0.1,
                "replay_top_logprob_margin": 0.1,
            },
        )

    def test_latency_summary_uses_successful_requests_only(self) -> None:
        rows = [{"status": "PASSED", "latency_s": value} for value in (1.0, 2.0, 3.0, 4.0)]
        rows.append({"status": "FAILED", "latency_s": 100.0})
        self.assertEqual(MODULE.latency_summary(rows), {"p50_s": 2.5, "p90_s": 4.0})

    def test_waiting_metric_handles_multiple_workers_and_missing_samples(self) -> None:
        self.assertIsNone(MODULE.waiting_count("# HELP vllm:num_requests_waiting\n"))
        self.assertEqual(
            MODULE.waiting_count(
                'vllm:num_requests_waiting{engine="0"} 3\n'
                'vllm:num_requests_waiting{engine="1"} 7\n'
                'vllm:num_requests_running{engine="0"} 9\n'
            ),
            7,
        )


if __name__ == "__main__":
    unittest.main()

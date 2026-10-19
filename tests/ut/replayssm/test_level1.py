# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU tests for validation failures that must never release the next level.

These tests use unittest directly; repository pytest mocks must not simulate NPU
success for the user-operated hardware acceptance scripts.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "replayssm"
sys.path.insert(0, str(SCRIPTS))

from level1_common import CASES, cann_versions, matches_version, model_kwargs, summarize  # noqa: E402
from run_level1 import run_case  # noqa: E402


class AcceptanceGateTests(unittest.TestCase):
    def test_passed_subset_cannot_release_next_level(self):
        report = summarize([{"case": "environment", "status": "PASSED"}])
        self.assertEqual(report["status"], "FAILED_OR_INCOMPLETE")
        self.assertFalse(report["next_level_allowed"])

    def test_duplicate_case_cannot_hide_missing_graph_case(self):
        report = summarize([{"case": "environment", "status": "PASSED"} for _ in CASES])
        self.assertEqual(report["status"], "FAILED_OR_INCOMPLETE")

    def test_all_passed_still_requires_human_confirmation(self):
        records = [{"case": case, "status": "PASSED"} for case in CASES]
        report = summarize(records)
        self.assertEqual(report["status"], "AUTOMATED_CHECKS_PASSED")
        self.assertEqual(report["user_confirmation"], "PENDING")
        self.assertFalse(report["next_level_allowed"])

    def test_one_failure_is_not_hidden_by_other_passes(self):
        records = [{"case": case, "status": "PASSED"} for case in CASES]
        records[-1]["status"] = "FAILED"
        self.assertEqual(summarize(records)["status"], "FAILED_OR_INCOMPLETE")

    def test_wrong_release_or_torch_npu_post_is_rejected(self):
        self.assertTrue(matches_version("0.30.0rc2+local", "0.30.0"))
        for value in (None, "0.23.0", "0.30.01", "0.30.0.post1"):
            self.assertFalse(matches_version(value, "0.30.0"))
        self.assertFalse(matches_version("2.10.0.post3", "2.10.0.post4"))

    def test_compiler_version_retains_path_and_full_build_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            info = Path(folder) / "version.info"
            info.write_text("Version=9.2.0_20260930220649\nrequired_package_runtime_version=9.2.0\n")
            records = cann_versions([info, info, Path(folder) / "missing"])
            self.assertEqual(records, [{"path": str(info.resolve()), "version": "9.2.0_20260930220649"}])

    def test_baseline_never_enables_replay_or_low_precision_state(self):
        for case in CASES[2:]:
            kwargs = model_kwargs("/model", 4, case, "v1", 0.85, 3)
            self.assertEqual(kwargs["mamba_ssm_cache_dtype"], "float32")
            self.assertNotIn("use_replayssm", kwargs)
            self.assertFalse(kwargs["enable_prefix_caching"])
            self.assertEqual(kwargs["enforce_eager"], case.endswith("eager"))
            if case.startswith("mtp"):
                self.assertEqual(kwargs["speculative_config"]["method"], "qwen3_5_mtp")

    def test_child_exit_failure_overrides_passed_json(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            report = output / "environment.json"
            report.write_text('{"case":"environment","status":"PASSED"}')
            record = run_case(
                [sys.executable, "-c", "raise SystemExit(7)"], output, "environment", 10, os.environ.copy()
            )
            self.assertEqual(record["status"], "FAILED")
            self.assertEqual(record["returncode"], 7)

    def test_missing_worker_report_is_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            record = run_case([sys.executable, "-c", "pass"], Path(folder), "environment", 10, os.environ.copy())
            self.assertEqual(record["status"], "FAILED")

    def test_malformed_worker_report_is_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "environment.json").write_text("not json")
            record = run_case([sys.executable, "-c", "pass"], output, "environment", 10, os.environ.copy())
            self.assertEqual(record["status"], "FAILED")

    def test_array_report_cannot_crash_gate_or_be_treated_as_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "environment.json").write_text("[]")
            record = run_case([sys.executable, "-c", "pass"], output, "environment", 10, os.environ.copy())
            self.assertEqual(record["status"], "FAILED")

    def test_timed_out_child_is_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            record = run_case(
                [sys.executable, "-c", "import time; time.sleep(5)"],
                Path(folder),
                "environment",
                0.05,
                os.environ.copy(),
            )
            self.assertEqual(record["status"], "FAILED")
            self.assertIn("Timed out", record["error"])

    def test_dry_run_requires_no_torch_and_creates_no_pass_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text('{"model_type":"qwen3_5"}')
            output = root / "results"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "run_level1.py"),
                    "--model",
                    str(model),
                    "--tp",
                    "4",
                    "--output",
                    str(output),
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "DRY_RUN_NOT_VALIDATED")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

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
from argparse import Namespace
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "replayssm"
sys.path.insert(0, str(SCRIPTS))

from level1_common import CASES, cann_versions, matches_version, model_kwargs, summarize  # noqa: E402
from level1_worker import operator_worker_command  # noqa: E402
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
        self.assertTrue(matches_version("0.30.0rc2.dev171+g62cc09b17", "0.30.0"))
        self.assertTrue(matches_version("0.30.0.dev171+source", "0.30.0"))
        self.assertTrue(matches_version("0.30.0+empty", "0.30.0"))
        self.assertTrue(matches_version("2.10.0+cpu", "2.10.0"))
        for value in (None, "0.23.0", "0.30.01", "0.30.0.post1"):
            self.assertFalse(matches_version(value, "0.30.0"))
        self.assertFalse(matches_version("2.10.0.post3", "2.10.0.post4"))
        self.assertFalse(matches_version("2.10.0.post4.dev1", "2.10.0.post4"))

    def test_compiler_version_retains_path_and_full_build_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            info = Path(folder) / "version.info"
            info.write_text("Version=9.2.0_20260930220649\nrequired_package_runtime_version=9.2.0\n")
            records = cann_versions([info, info, Path(folder) / "missing"])
            self.assertEqual(records, [{"path": str(info.resolve()), "version": "9.2.0_20260930220649"}])

    def test_native_baseline_uses_explicit_bf16_caches_without_replay(self):
        for tp in (4, 8):
            for runner in ("v1", "v2"):
                for case in CASES[2:]:
                    with self.subTest(tp=tp, runner=runner, case=case):
                        kwargs = model_kwargs("/model", tp, case, runner, 0.85, 3)
                        self.assertEqual(kwargs["dtype"], "bfloat16")
                        self.assertEqual(kwargs["mamba_cache_dtype"], "bfloat16")
                        self.assertEqual(kwargs["mamba_ssm_cache_dtype"], "bfloat16")
                        self.assertNotIn("use_replayssm", kwargs)
                        self.assertFalse(kwargs["enable_prefix_caching"])
                        self.assertEqual(kwargs["enforce_eager"], case.endswith("eager"))
                        if case.startswith("mtp"):
                            self.assertEqual(kwargs["speculative_config"]["method"], "qwen3_5_mtp")

    def test_operator_probe_uses_explicit_per_device_child(self):
        args = Namespace(model="/model", tp=8, runner="v1", memory_utilization=0.85, spec_tokens=3)
        command = operator_worker_command(args, 7, Path("/tmp/device7.json"))
        self.assertEqual(command[command.index("--device") + 1], "7")
        self.assertEqual(command[command.index("--tp") + 1], "8")
        self.assertEqual(command[command.index("--report") + 1], "/tmp/device7.json")

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

    def test_worker_exit_and_timeout_clean_up_owned_descendants(self):
        for hang in (False, True):
            with self.subTest(hang=hang), tempfile.TemporaryDirectory() as folder:
                output = Path(folder)
                child = "\n".join(
                    [
                        "import signal, sys, time",
                        "from pathlib import Path",
                        "def stop(*args):",
                        "    Path(sys.argv[2]).write_text('stopped')",
                        "    raise SystemExit(0)",
                        "signal.signal(signal.SIGTERM, stop)",
                        "Path(sys.argv[1]).write_text('ready')",
                        "while True: time.sleep(0.05)",
                    ]
                )
                parent = "\n".join(
                    [
                        "import json, subprocess, sys, time",
                        "from pathlib import Path",
                        "root = Path(sys.argv[1])",
                        f"subprocess.Popen([sys.executable, '-c', {child!r}, str(root/'ready'), str(root/'stopped')])",
                        "deadline = time.monotonic() + 3",
                        "while not (root/'ready').exists():",
                        "    if time.monotonic() > deadline: raise RuntimeError('fixture child did not start')",
                        "    time.sleep(0.01)",
                        "(root/'environment.json').write_text(json.dumps({'case':'environment','status':'PASSED'}))",
                        "while True: time.sleep(0.05)" if hang else "raise SystemExit(0)",
                    ]
                )
                record = run_case(
                    [sys.executable, "-c", parent, folder], output, "environment", 1 if hang else 5, os.environ.copy()
                )
                self.assertEqual(record["status"], "FAILED" if hang else "PASSED")
                self.assertTrue((output / "stopped").is_file(), "owned descendant must receive cleanup signal")

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
            plan = json.loads(result.stdout)
            self.assertEqual(plan["status"], "DRY_RUN_NOT_VALIDATED")
            self.assertEqual(set(plan["serve_commands"]), set(CASES[2:]))
            self.assertTrue(all(command[3] == "serve" for command in plan["serve_commands"].values()))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

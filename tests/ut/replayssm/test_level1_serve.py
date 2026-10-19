# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise real loopback HTTP and subprocess cleanup without any NPU packages."""

import copy
import json
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "replayssm"
sys.path.insert(0, str(SCRIPTS))

import level1_serve  # noqa: E402
from level1_common import model_kwargs, serve_command  # noqa: E402


class ServeValidationTests(unittest.TestCase):
    def test_serve_cli_keeps_bf16_graph_and_mtp_options(self):
        for case in ("ar_eager", "ar_graph", "mtp_eager", "mtp_graph"):
            with self.subTest(case=case):
                kwargs = model_kwargs("/models/model with spaces", 8, case, "v1", 0.8, 3)
                command = serve_command(kwargs, 18080, "unique-model")
                self.assertEqual(
                    command[:5], [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", kwargs["model"]]
                )
                for flag in ("--mamba-cache-dtype", "--mamba-ssm-cache-dtype"):
                    self.assertEqual(command[command.index(flag) + 1], "bfloat16")
                self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
                self.assertIn("--no-enable-prefix-caching", command)
                self.assertIn("--no-async-scheduling", command)
                self.assertNotIn("--no-disable-log-stats", command)
                self.assertNotIn("--disable-log-stats", command)
                self.assertEqual("--enforce-eager" in command, case.endswith("eager"))
                if case.endswith("graph"):
                    self.assertEqual(
                        json.loads(command[command.index("--compilation-config") + 1]),
                        {"cudagraph_mode": "FULL_DECODE_ONLY"},
                    )
                if case.startswith("mtp"):
                    self.assertEqual(
                        json.loads(command[command.index("--speculative-config") + 1]), kwargs["speculative_config"]
                    )

    def test_metric_parser_ignores_comments_created_and_unrelated_counters(self):
        metrics = """# TYPE vllm:spec_decode_num_drafts counter
vllm:spec_decode_num_drafts_created{engine="0"} 999
vllm:spec_decode_num_drafts_total{engine="0"} 1.2e1
vllm:spec_decode_num_drafts_total{engine="1"} 3
another_counter 456
"""
        self.assertEqual(level1_serve.draft_count(metrics), 15)
        self.assertIsNone(level1_serve.draft_count("# counter absent\n"))
        for value in ("NaN", "+Inf", "-1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                level1_serve.draft_count(f"vllm:spec_decode_num_drafts_total {value}")

    def test_http_completion_requires_distinct_choices_tokens_and_matching_usage(self):
        response = {
            "choices": [{"index": index, "token_ids": [1], "finish_reason": "length"} for index in range(2)],
            "usage": {"completion_tokens": 2},
        }
        self.assertEqual(level1_serve.validate_completion(response, 2), 2)
        for issue in ("duplicate", "missing_tokens", "bad_token", "usage", "unfinished"):
            broken = copy.deepcopy(response)
            if issue == "duplicate":
                broken["choices"][1]["index"] = 0
            elif issue == "missing_tokens":
                del broken["choices"][0]["token_ids"]
            elif issue == "bad_token":
                broken["choices"][0]["token_ids"] = [True]
            elif issue == "usage":
                broken["usage"]["completion_tokens"] = 0
            else:
                broken["choices"][0]["finish_reason"] = None
            with self.subTest(issue=issue), self.assertRaises(RuntimeError):
                level1_serve.validate_completion(broken, 2)

    def run_fixture(self, mode, case="mtp_graph", expected_error=None, selected_port=None):
        with tempfile.TemporaryDirectory() as directory, socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = selected_port or probe.getsockname()[1]
            probe.close()
            args = SimpleNamespace(
                model="/model",
                tp=4,
                case=case,
                runner="v1",
                memory_utilization=0.85,
                spec_tokens=3,
                port=port,
                report=Path(directory) / f"{case}.json",
                startup_timeout=3,
                request_timeout=2,
            )
            if mode == "unready":
                args.startup_timeout = 0.3
            if mode == "slow_request":
                args.request_timeout = 0.1
            processes = []
            original_popen = subprocess.Popen

            def start(*values, **kwargs):
                process = original_popen(*values, **kwargs)
                processes.append(process)
                return process

            def command(kwargs, selected_port, model_name):
                return [
                    sys.executable,
                    str(Path(__file__).with_name("fake_server.py")),
                    str(selected_port),
                    model_name,
                    mode,
                ]

            with (
                patch.object(level1_serve, "serve_command", command),
                patch.object(level1_serve.subprocess, "Popen", start),
                patch.object(level1_serve, "METRICS_WAIT_SECONDS", 0),
            ):
                if expected_error:
                    with self.assertRaisesRegex((RuntimeError, TimeoutError), expected_error):
                        level1_serve.run_serve_case(args)
                else:
                    result = level1_serve.run_serve_case(args)
                    self.assertEqual(result["status"], "PASSED")
                    self.assertEqual(result["validation_mode"], "vllm_serve_http")
                    if case.startswith("mtp"):
                        self.assertGreater(result["drafts_delta"], 0)
                    self.assertEqual(result["model_graph_dispatch"], "REQUESTED_NOT_INSTRUMENTED")
                    self.assertEqual(result["generation_tokens"], 4)
                    self.assertTrue(args.report.with_suffix(".generation.json").exists())
                self.assertEqual(len(processes), 1)
                self.assertIsNotNone(processes[0].poll(), "owned server must be reaped even on failure")
                with socket.socket() as check:
                    self.assertNotEqual(check.connect_ex(("127.0.0.1", port)), 0, "server port must be closed")
                self.assertTrue(args.report.with_suffix(".server.log").exists())
            return port

    def test_real_http_warmup_generation_and_metrics(self):
        self.run_fixture("ok")

    def test_server_exit_before_ready_is_failure(self):
        self.run_fixture("exit", expected_error="exited before ready")

    def test_health_timeout_cleans_up_owned_server(self):
        self.run_fixture("unready", expected_error="readiness timed out")

    def test_another_models_health_cannot_pass(self):
        self.run_fixture("wrong_model", expected_error="unique served model")

    def test_http_generation_error_cleans_up_owned_server(self):
        self.run_fixture("http_error", expected_error="HTTP 500.*fixture generation failed")

    def test_positive_but_unchanged_mtp_counter_does_not_pass(self):
        self.run_fixture("stale_counter", expected_error="no positive /metrics draft counter delta")

    def test_missing_mtp_counter_does_not_pass(self):
        self.run_fixture("no_counter", expected_error="no positive /metrics draft counter delta")

    def test_ar_does_not_require_mtp_metrics(self):
        self.run_fixture("no_counter", case="ar_graph")

    def test_request_timeout_cleans_up_owned_server(self):
        self.run_fixture("slow_request", expected_error="timed out")

    def test_sequential_servers_can_reuse_the_port(self):
        port = self.run_fixture("ok")
        self.run_fixture("ok", selected_port=port)

    def test_occupied_port_refuses_to_contact_existing_server(self):
        with tempfile.TemporaryDirectory() as directory, socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            args = SimpleNamespace(
                model="/model",
                tp=4,
                case="ar_eager",
                runner="v1",
                memory_utilization=0.85,
                spec_tokens=3,
                port=occupied.getsockname()[1],
                report=Path(directory) / "ar_eager.json",
            )
            with patch.object(level1_serve.subprocess, "Popen") as start, self.assertRaises(OSError):
                level1_serve.run_serve_case(args)
            start.assert_not_called()


if __name__ == "__main__":
    unittest.main()

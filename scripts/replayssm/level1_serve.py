# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Validate an owned vllm serve process via loopback HTTP; no NPU imports here."""

import json
import math
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid

from level1_common import model_kwargs, serve_command, write_json

SERVED_MODEL_PREFIX = "replayssm-level1"
DRAFT_COUNTER = "vllm:spec_decode_num_drafts_total"
METRICS_WAIT_SECONDS = 30


def http_request(base_url: str, path: str, timeout: float, payload: dict | None = None) -> str:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        base_url + path, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    # Local validation must not go through a caller's HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.read().decode()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{path}: HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc


def wait_ready(process, base_url: str, model_name: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last_error = "No HTTP response yet"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vllm serve exited before ready: returncode={process.returncode}; inspect server log")
        try:
            remaining = max(0.01, min(2.0, deadline - time.monotonic()))
            http_request(base_url, "/health", remaining)
            models = json.loads(http_request(base_url, "/v1/models", remaining))
        except (OSError, ValueError, RuntimeError) as exc:
            last_error = str(exc)
            time.sleep(min(0.2, max(0, deadline - time.monotonic())))
            continue
        if not isinstance(models, dict) or not any(
            isinstance(item, dict) and item.get("id") == model_name for item in models.get("data", [])
        ):
            raise RuntimeError("/v1/models does not identify this invocation's unique served model")
        return models
    raise TimeoutError(f"vllm serve readiness timed out after {timeout}s: {last_error}")


def draft_count(metrics: str) -> float | None:
    values = []
    pattern = re.compile(rf"^{re.escape(DRAFT_COUNTER)}(?:\{{.*\}})?\s+(\S+)(?:\s+\d+)?$")
    for line in metrics.splitlines():
        match = pattern.fullmatch(line.strip())
        if match:
            value = float(match[1])
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid speculative draft counter")
            values.append(value)
    return sum(values) if values else None


def validate_completion(response: dict, batch_size: int) -> int:
    choices = response.get("choices", [])
    if len(choices) != batch_size or {item.get("index") for item in choices} != set(range(batch_size)):
        raise RuntimeError("HTTP completion returned an incomplete or duplicate batch")
    total = 0
    for choice in choices:
        tokens = choice.get("token_ids")
        if not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens):
            raise RuntimeError("HTTP completion must return nonempty token_ids for each prompt")
        if choice.get("finish_reason") not in ("length", "stop"):
            raise RuntimeError("HTTP completion did not finish successfully")
        total += len(tokens)
    if response.get("usage", {}).get("completion_tokens") != total:
        raise RuntimeError("HTTP completion usage does not match returned token IDs")
    return total


def run_serve_case(args) -> dict:
    kwargs = model_kwargs(args.model, args.tp, args.case, args.runner, args.memory_utilization, args.spec_tokens)
    model_name = f"{SERVED_MODEL_PREFIX}-{args.case}-{uuid.uuid4().hex}"
    command = serve_command(kwargs, args.port, model_name)
    base_url = f"http://127.0.0.1:{args.port}"
    prefix = args.report.with_suffix("")
    server_log = prefix.with_suffix(".server.log")
    record = {
        "status": "RUNNING",
        "case": args.case,
        "validation_mode": "vllm_serve_http",
        "engine_kwargs": kwargs,
        "serve_command": command,
        "base_url": base_url,
        "served_model_name": model_name,
        "server_log": server_log.name,
        "model_graph_dispatch": "REQUESTED_NOT_INSTRUMENTED" if args.case.endswith("_graph") else "EAGER",
        "scope": "Short local HTTP smoke run; timing is diagnostic, not a performance benchmark.",
    }
    write_json(args.report, record)
    # Refuse an occupied port; the unique model name also rejects unrelated services.
    with socket.socket() as probe:
        # Permit sequential cases to reuse a port with closed HTTP connections in TIME_WAIT.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
    started = time.monotonic()
    with server_log.open("w") as log:
        # Inherit the isolated worker's group so the outer timeout kills all TP workers.
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            record["models"] = wait_ready(process, base_url, model_name, args.startup_timeout)
            record["startup_seconds"] = time.monotonic() - started
            before = http_request(base_url, "/metrics", args.request_timeout)
            prefix.with_suffix(".metrics-before.log").write_text(before)
            record["drafts_before"] = draft_count(before)
            write_json(args.report, record)
            payload = {
                "model": model_name,
                "prompt": ["请用两句话解释什么是矩阵乘法。", "Write a short Python function that adds two integers."],
                "temperature": 0,
                "seed": 42,
                "max_tokens": 64,
                "ignore_eos": True,
                "return_token_ids": True,
                "stream": False,
            }
            write_json(prefix.with_suffix(".request.json"), payload)
            for phase in ("warmup", "generation"):
                started = time.monotonic()
                response = json.loads(http_request(base_url, "/v1/completions", args.request_timeout, payload))
                seconds = time.monotonic() - started
                write_json(prefix.with_suffix(f".{phase}.json"), response)
                tokens = validate_completion(response, len(payload["prompt"]))
                record[f"{phase}_seconds"] = seconds
                record[f"{phase}_tokens"] = tokens
                write_json(args.report, record)
            deadline = time.monotonic() + METRICS_WAIT_SECONDS
            while True:
                after = http_request(base_url, "/metrics", args.request_timeout)
                prefix.with_suffix(".metrics-after.log").write_text(after)
                record["drafts_after"] = draft_count(after)
                delta = (record["drafts_after"] or 0) - (record["drafts_before"] or 0)
                record["drafts_delta"] = delta
                write_json(args.report, record)
                if not args.case.startswith("mtp") or delta > 0:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("MTP requested but no positive /metrics draft counter delta was observed")
                time.sleep(0.5)
            if process.poll() is not None:
                raise RuntimeError("vllm serve exited during validation; inspect server log")
            record["status"] = "PASSED"
            return record
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)

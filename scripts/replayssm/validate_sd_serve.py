# SPDX-License-Identifier: Apache-2.0
"""Exercise an already running vllm serve instance and save SD evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def get_text(base: str, path: str, timeout: float = 10) -> str:
    with urlopen(base + path, timeout=timeout) as response:  # noqa: S310 - explicitly supplied local vllm server
        return response.read().decode("utf-8")


def run_request(base: str, model: str, prompt: str, max_tokens: int, seed: int, timeout: float) -> dict:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "temperature": 0,
            "seed": seed,
            "max_tokens": max_tokens,
            "ignore_eos": True,
            "return_token_ids": True,
            "stream": False,
        }
    ).encode()
    request = Request(base + "/v1/completions", data=payload, headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - explicitly supplied local vllm server
            result = json.load(response)
        choice = result["choices"][0]
        ids = choice.get("token_ids")
        usage = result.get("usage", {})
        if not isinstance(ids, list) or not ids:
            raise ValueError("completion lacks nonempty token_ids")
        if usage.get("completion_tokens") != len(ids):
            raise ValueError("completion token count differs from token_ids")
        return {
            "status": "PASSED",
            "latency_s": round(time.monotonic() - started, 3),
            "token_ids": ids,
            "text": choice.get("text", ""),
            "usage": usage,
        }
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, IndexError) as exc:
        return {"status": "FAILED", "latency_s": round(time.monotonic() - started, 3), "error": str(exc)}


def waiting_count(metrics: str) -> float | None:
    values = [
        float(match.group(1))
        for line in metrics.splitlines()
        if line.startswith("vllm:num_requests_waiting")
        if (match := re.search(r"\s([-+]?\d+(?:\.\d+)?)$", line))
    ]
    return max(values) if values else None


def latency_summary(results: list[dict]) -> dict[str, float | None]:
    latencies = sorted(row["latency_s"] for row in results if row.get("status") == "PASSED")
    if not latencies:
        return {"p50_s": None, "p90_s": None}
    return {
        "p50_s": round(statistics.median(latencies), 3),
        "p90_s": latencies[math.ceil(0.9 * len(latencies)) - 1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18083")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--alternate-prompt-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--compare", type=Path, help="Baseline JSON from the same prompt and request count")
    args = parser.parse_args()
    if args.concurrency < 1 or args.repetitions < 1 or args.max_tokens < 1 or args.timeout <= 0:
        parser.error("concurrency, repetitions, max-tokens and timeout must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not args.url.startswith("http://127.0.0.1:") and not args.url.startswith("http://localhost:"):
        parser.error("only a local vllm serve URL is accepted")
    prompt = args.prompt_file.read_text()
    if not prompt.strip():
        parser.error("prompt file is empty")
    prompts = [prompt]
    if args.alternate_prompt_file is not None:
        alternate = args.alternate_prompt_file.read_text()
        if not alternate.strip():
            parser.error("alternate prompt file is empty")
        prompts.append(alternate)
    try:
        get_text(args.url, "/health")
        models = json.loads(get_text(args.url, "/v1/models"))
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        parser.error(f"server is not ready: {exc}")
    if args.model not in {item.get("id") for item in models.get("data", [])}:
        parser.error(f"served model {args.model!r} is absent from /v1/models")

    stop_polling = threading.Event()
    waiting_samples: list[float] = []

    def poll_waiting() -> None:
        while not stop_polling.wait(0.25):
            try:
                value = waiting_count(get_text(args.url, "/metrics", timeout=5))
                if value is not None:
                    waiting_samples.append(value)
            except (HTTPError, URLError, TimeoutError):
                continue

    before = get_text(args.url, "/metrics")
    poller = threading.Thread(target=poll_waiting, daemon=True)
    poller.start()
    results: list[dict] = [{} for _ in range(args.concurrency * args.repetitions)]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                run_request, args.url, args.model, prompts[i % len(prompts)], args.max_tokens, 42, args.timeout
            ): i
            for i in range(len(results))
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    stop_polling.set()
    poller.join(timeout=6)
    after = get_text(args.url, "/metrics")

    mismatches = []
    if args.compare is not None:
        baseline = json.loads(args.compare.read_text())
        expected = baseline.get("results", [])
        if len(expected) != len(results):
            parser.error("baseline request count differs")
        mismatches = [
            i for i, (old, new) in enumerate(zip(expected, results)) if old.get("token_ids") != new.get("token_ids")
        ]
    report = {
        "status": "PASSED" if all(row.get("status") == "PASSED" for row in results) and not mismatches else "FAILED",
        "model": args.model,
        "concurrency": args.concurrency,
        "repetitions": args.repetitions,
        "max_tokens": args.max_tokens,
        "prompt_chars": [len(value) for value in prompts],
        "peak_waiting_requests": max(waiting_samples, default=None),
        "latency": latency_summary(results),
        "mismatched_request_indices": mismatches,
        "results": results,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    args.output.with_suffix(".metrics.before.txt").write_text(before)
    args.output.with_suffix(".metrics.after.txt").write_text(after)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False))
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

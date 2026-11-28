# SPDX-License-Identifier: Apache-2.0
"""Measure one DFlash workload against an otherwise idle vllm serve instance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from validate_sd_serve import get_text

_COUNTER = re.compile(
    r"^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens|accepted_tokens_per_pos)_total"
    r"(?:\{([^}]*)\})?\s+(\S+)$"
)
_POSITION = re.compile(r'(?:^|,)position="(\d+)"')


def counters(metrics: str) -> dict[str, int]:
    """Sum counters across engine labels and keep separate per-position totals."""
    result: dict[str, int] = {}
    for line in metrics.splitlines():
        match = _COUNTER.fullmatch(line)
        if match is None:
            continue
        name, labels, value = match.groups()
        if name == "accepted_tokens_per_pos":
            position = _POSITION.search(labels or "")
            if position is None:
                raise ValueError(f"missing position label: {line}")
            name += f"_{position[1]}"
        result[name] = result.get(name, 0) + int(float(value))
    return result


def request(args: argparse.Namespace, item: dict) -> dict:
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": item["prompt"]}],
        "temperature": 0,
        "seed": 42,
        "max_tokens": args.max_tokens,
        "ignore_eos": args.ignore_eos,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if schema := item.get("json_schema"):
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema, "strict": True},
        }
    http = Request(args.url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    started = time.monotonic()
    first_token = None
    usage = None
    finish_reason = None
    text = []
    try:
        with urlopen(http, timeout=args.timeout) as response:  # noqa: S310 - benchmark server supplied explicitly
            for raw in response:
                if not raw.startswith(b"data:"):
                    continue
                payload = raw[5:].strip()
                if payload == b"[DONE]":
                    break
                chunk = json.loads(payload)
                if "error" in chunk:
                    raise ValueError(str(chunk["error"]))
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if chunk["choices"]:
                    choice = chunk["choices"][0]
                    delta = choice["delta"]
                    content = delta.get("content")
                    if first_token is None and (content or delta.get("reasoning") or delta.get("reasoning_content")):
                        first_token = time.monotonic()
                    if content:
                        text.append(content)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
        finished = time.monotonic()
        if not usage or not usage["completion_tokens"] or first_token is None or finish_reason is None:
            raise ValueError("stream missing generated tokens, usage or finish reason")
        output = "".join(text)
        result = {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "ttft_s": first_token - started,
            "latency_s": finished - started,
            "tpot_s": (finished - first_token) / (usage["completion_tokens"] - 1)
            if usage["completion_tokens"] > 1
            else None,
            "finish_reason": finish_reason,
            "text": output,
        }
        if schema or item.get("expect_json"):
            try:
                parsed = json.loads(output)
            except json.JSONDecodeError:
                result["valid_json"] = False
            else:
                result["valid_json"] = True
                if "expected_json" in item:
                    result["expected_json_match"] = parsed == item["expected_json"]
        return result
    except (URLError, TimeoutError, ValueError, KeyError) as exc:
        result = {"error": str(exc), "latency_s": time.monotonic() - started}
        if isinstance(exc, HTTPError):
            result["response"] = exc.read().decode(errors="replace")[:2000]
        return result


def percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p90": None}
    return {"p50": statistics.median(values), "p90": sorted(values)[math.ceil(0.9 * len(values)) - 1]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18145")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dataset", type=Path, required=True, help='JSONL objects with "prompt" and optional JSON schema'
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draft-tokens", type=int, choices=range(1, 8), required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= args.requests or args.max_tokens < 1 or args.timeout <= 0:
        parser.error("require 1 <= concurrency <= requests, positive max-tokens and positive timeout")
    raw = args.dataset.read_bytes()
    dataset = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not dataset or any(not isinstance(row.get("prompt"), str) or not row["prompt"].strip() for row in dataset):
        parser.error("every dataset row must contain a nonempty prompt")
    cases = [dataset[i % len(dataset)] for i in range(args.requests)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    before = get_text(args.url, "/metrics")
    args.output.with_suffix(".before.prom").write_text(before)
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(lambda item: request(args, item), cases))
    elapsed = time.monotonic() - start
    after = get_text(args.url, "/metrics")
    args.output.with_suffix(".after.prom").write_text(after)
    old, new = map(counters, (before, after))
    delta = {key: new.get(key, 0) - old.get(key, 0) for key in old.keys() | new.keys()}
    drafts = delta.get("drafts", 0)
    proposed = delta.get("draft_tokens", 0)
    accepted = delta.get("accepted_tokens", 0)
    per_position = [delta.get(f"accepted_tokens_per_pos_{i}", 0) for i in range(args.draft_tokens)]
    valid = [row for row in rows if "error" not in row]
    counts_ok = (
        all(value >= 0 for value in delta.values())
        and 0 <= accepted <= proposed <= args.draft_tokens * drafts
        and sum(per_position) == accepted
    )
    summary = {
        "status": "PASSED" if len(valid) == len(rows) and counts_ok else "FAILED",
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "requests": len(rows),
        "passed": len(valid),
        "concurrency": args.concurrency,
        "draft_tokens_per_window": args.draft_tokens,
        "elapsed_s": elapsed,
        "output_toks_s": sum(row["completion_tokens"] for row in valid) / elapsed,
        "ttft_s": percentiles([row["ttft_s"] for row in valid]),
        "tpot_s": percentiles([row["tpot_s"] for row in valid if row["tpot_s"] is not None]),
        "latency_s": percentiles([row["latency_s"] for row in valid]),
        "drafts": drafts,
        "draft_tokens": proposed,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / proposed if proposed else None,
        "mean_accept_length": 1 + accepted / drafts if drafts else None,
        "accepted_tokens_per_position": per_position,
        "per_position_acceptance_rate": [value / drafts for value in per_position]
        if drafts and proposed == args.draft_tokens * drafts
        else None,
        "json_valid": sum(row.get("valid_json", False) for row in rows),
        "json_requested": sum(bool(item.get("json_schema") or item.get("expect_json")) for item in cases),
        "json_expected_matches": sum(row.get("expected_json_match", False) for row in rows),
    }
    args.output.write_text(json.dumps({**summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary))
    return int(summary["status"] != "PASSED")


if __name__ == "__main__":
    raise SystemExit(main())

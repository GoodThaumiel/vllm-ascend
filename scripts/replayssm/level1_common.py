# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Dependency-free contracts for user-operated ReplaySSM validation."""

import json
import re
from pathlib import Path

CASES = ("environment", "operators", "ar_eager", "ar_graph", "mtp_eager", "mtp_graph")
EXPECTED_PACKAGES = {
    "vllm": "0.30.0",
    "vllm-ascend": "0.30.0",
    "torch": "2.10.0",
    "torch-npu": "2.10.0.post4",
    "triton-ascend": "3.2.2",
}


def matches_version(actual: str | None, expected: str) -> bool:
    if actual is None:
        return False
    # Allow a release candidate or local build, but not another release/post.
    return re.fullmatch(re.escape(expected) + r"(?:(?:rc|a|b)\d+)?(?:\+[^\s]+)?", actual) is not None


def cann_versions(paths: list[Path]) -> list[dict[str, str]]:
    records = []
    for path in dict.fromkeys(paths):
        if not path.is_file():
            continue
        content = path.read_text(errors="replace")
        match = re.search(r"(?mi)^\s*(?:version|cann_version)\s*[=:]\s*[\"']?([^\s\"']+)", content)
        if match:
            records.append({"path": str(path.resolve()), "version": match[1]})
    return records


def write_json(path: Path, value: dict) -> None:
    # Replace only reports owned by this invocation, not a previous run folder.
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def summarize(records: list[dict]) -> dict:
    names = [record["case"] for record in records]
    complete = len(names) == len(CASES) and set(names) == set(CASES)
    all_passed = complete and all(record.get("status") == "PASSED" for record in records)
    return {
        "level": 1,
        "status": "AUTOMATED_CHECKS_PASSED" if all_passed else "FAILED_OR_INCOMPLETE",
        "user_confirmation": "PENDING",
        "next_level_allowed": False,
        "cases": records,
        "limits": [
            "Native baseline only; ReplaySSM is not implemented or enabled at level 1.",
            "Model graph configuration is requested; actual model graph dispatch needs log review.",
            "Small operator checks and generated text do not establish model accuracy or performance.",
        ],
    }


def model_kwargs(model: str, tp: int, case: str, runner: str, memory: float, spec_tokens: int) -> dict:
    if case not in CASES[2:]:
        raise ValueError(f"Not a model case: {case}")
    graph = case.endswith("_graph")
    kwargs = {
        "model": model,
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": 1,
        "dtype": "bfloat16",
        "mamba_ssm_cache_dtype": "float32",
        "mamba_cache_mode": "none",
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "language_model_only": True,
        "max_model_len": 4096,
        "max_num_seqs": 2,
        "max_num_batched_tokens": 4096,
        "gpu_memory_utilization": memory,
        "enforce_eager": not graph,
        "async_scheduling": False,
        "seed": 42,
        "disable_log_stats": False,
        "distributed_executor_backend": "mp",
    }
    if graph:
        kwargs["compilation_config"] = {"cudagraph_mode": "FULL_DECODE_ONLY"}
    if case.startswith("mtp"):
        # Match the repository's Qwen3.6 tutorial: target graph + eager drafter.
        kwargs["speculative_config"] = {
            "method": "qwen3_5_mtp",
            "num_speculative_tokens": spec_tokens,
            "enforce_eager": True,
        }
    return kwargs

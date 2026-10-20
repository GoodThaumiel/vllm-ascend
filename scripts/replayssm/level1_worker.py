# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""One hardware-dependent case per process; import NPU packages only here."""

import argparse
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

from level1_common import CASES, EXPECTED_PACKAGES, cann_versions, matches_version, write_json
from level1_serve import run_serve_case


def environment(args) -> dict:
    checks = []
    details = {"python": platform.python_version(), "platform": platform.platform()}
    versions = {}
    for package, expected in EXPECTED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        versions[package] = actual
        checks.append(
            {"name": package, "passed": matches_version(actual, expected), "expected": expected, "actual": actual}
        )
    details["packages"] = versions
    roots = [Path("/usr/local/Ascend/ascend-toolkit/latest"), Path("/usr/local/Ascend/cann")]
    if os.environ.get("ASCEND_HOME_PATH"):
        roots.insert(0, Path(os.environ["ASCEND_HOME_PATH"]))
    version_paths = []
    for root in roots:
        version_paths.extend(root / name for name in ("compiler/version.info", "version.info", "version.cfg"))
        version_paths.extend(root.glob("*-linux/ascend_toolkit_install.info"))
    evidence = cann_versions(version_paths)
    active_evidence = next((item for item in evidence if item["path"].endswith("compiler/version.info")), None)
    if active_evidence is None and evidence:
        active_evidence = evidence[0]
    checks.append(
        {
            "name": "CANN toolkit version",
            "passed": bool(active_evidence and active_evidence["version"].split("_")[0].split("+")[0] == "9.2.0"),
            "actual": active_evidence,
        }
    )
    details["cann_evidence"] = evidence
    # Only the documented Ascend paths are collected; never dump all environment variables.
    details["ascend_paths"] = {name: os.environ.get(name) for name in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH")}
    try:
        smi = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=20, check=False)
        details["npu_smi"] = {"returncode": smi.returncode, "stdout": smi.stdout, "stderr": smi.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        details["npu_smi"] = {"error": str(exc)}
    write_json(args.report, {"case": args.case, "status": "RUNNING", "checks": checks, "details": details})
    import torch
    import torch_npu
    from fla_npu.ops import ascendc

    details["torch_npu_file"] = torch_npu.__file__
    details["fla_module_file"] = ascendc.__file__
    details["fla_distributions"] = {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name", "").lower().replace("_", "-").startswith("flash-linear-attention-npu")
    }
    available = torch.npu.is_available()
    count = torch.npu.device_count() if available else 0
    checks.append({"name": "visible NPU count", "passed": available and count >= args.tp, "actual": count})
    details["device_names_observed"] = [torch.npu.get_device_name(index) for index in range(count)]
    details["soc_identity_note"] = (
        "Observed device names are not authoritative compiler SoC identities. Confirm 910B4 separately."
    )
    signatures = {}
    for name in ("causal_conv1d_fn", "causal_conv1d_update", "recurrent_gated_delta_rule"):
        op = getattr(ascendc, name, None)
        checks.append({"name": f"fla_npu.{name}", "passed": callable(op)})
        if callable(op):
            try:
                signatures[name] = str(inspect.signature(op))
            except (TypeError, ValueError):
                signatures[name] = "signature unavailable"
    details["fla_signatures"] = signatures
    from vllm_ascend.ops import gdn

    details["gdn_module_file"] = gdn.__file__
    return {
        "status": "PASSED" if all(check["passed"] for check in checks) else "FAILED",
        "checks": checks,
        "details": details,
    }


def operator_worker_command(args, device: int, report: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--case",
        "operators",
        "--model",
        args.model,
        "--tp",
        str(args.tp),
        "--runner",
        args.runner,
        "--report",
        str(report),
        "--memory-utilization",
        str(args.memory_utilization),
        "--spec-tokens",
        str(args.spec_tokens),
        "--device",
        str(device),
    ]


def operators_on_device(args) -> dict:
    import torch
    import torch_npu  # noqa: F401

    from vllm_ascend.ops.triton.fla.l2norm import l2norm_fwd
    from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    device_index = args.device
    torch.npu.set_device(device_index)
    init_device_properties_triton()
    device = f"npu:{device_index}"
    heads = 48 // args.tp
    a = torch.linspace(-1, 1, 4 * heads, device=device).reshape(4, heads).to(torch.bfloat16)
    b = a.clone()
    a_log = torch.full((heads,), -2.0, device=device)
    bias = torch.zeros(heads, device=device)
    x = torch.linspace(-0.5, 0.5, 4 * heads * 128, device=device).reshape(4, heads, 128).to(torch.bfloat16)

    def compute(a=a, b=b, a_log=a_log, bias=bias, x=x):
        g, beta = fused_gdn_gating_patch(a_log, a, b, bias)
        return g, beta, l2norm_fwd(x)

    def reference(a=a, b=b, a_log=a_log, bias=bias, x=x):
        g = -torch.exp(a_log.float())[None, :] * torch.nn.functional.softplus(a.float() + bias.float())
        beta = torch.sigmoid(b.float()).to(b.dtype)
        norm = (x.float() / torch.sqrt(x.float().square().sum(-1, keepdim=True) + 1e-6)).to(x.dtype)
        return g.unsqueeze(0), beta.unsqueeze(0), norm

    for _ in range(3):
        eager = compute()
    expected = reference()
    torch.npu.synchronize()
    for actual, target in zip(eager, expected, strict=True):
        torch.testing.assert_close(actual, target, atol=5e-3, rtol=1e-2)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = compute()
    # Mutate fixed inputs after capture to reject a stale-output pseudo-pass.
    a.fill_(0.25)
    b.fill_(-0.25)
    x.fill_(0.125)
    expected = reference()
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    errors = []
    for actual, target in zip(captured, expected, strict=True):
        torch.testing.assert_close(actual, target, atol=5e-3, rtol=1e-2)
        errors.append(float((actual.float() - target.float()).abs().max().cpu()))
    result = {
        "device": device_index,
        "local_value_heads": heads,
        "status": "PASSED",
        "graph_replays": 3,
        "max_abs_errors": errors,
    }
    del graph, captured, eager
    torch.npu.empty_cache()
    return {
        "status": "PASSED",
        "devices": [result],
        "scope": "Existing gating and L2Norm kernels only; not Replay or recurrent kernel acceptance.",
    }


def operators(args) -> dict:
    if args.device is not None:
        return operators_on_device(args)

    results = []
    for device in range(args.tp):
        report = args.report.with_name(f"{args.report.stem}-device{device}.json")
        log = args.report.with_name(f"{args.report.stem}-device{device}.log")
        with log.open("w") as stream:
            completed = subprocess.run(
                operator_worker_command(args, device, report),
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode != 0 or not report.is_file():
            raise RuntimeError(f"operator probe failed on device {device}; see {log}")
        child = json.loads(report.read_text())
        if child.get("status") != "PASSED" or len(child.get("devices", [])) != 1:
            raise RuntimeError(f"operator probe did not pass on device {device}; see {report}")
        results.extend(child["devices"])
    return {
        "status": "PASSED",
        "devices": results,
        "process_model": "one subprocess per logical NPU, matching TP workers",
        "scope": "Existing gating and L2Norm kernels only; not Replay or recurrent kernel acceptance.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, required=True, choices=(4, 8))
    parser.add_argument("--runner", choices=("v1", "v2"), required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--memory-utilization", type=float, required=True)
    parser.add_argument("--spec-tokens", type=int, required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--device", type=int, choices=range(8))
    args = parser.parse_args()
    record = {"case": args.case, "status": "FAILED"}
    try:
        function = (
            environment if args.case == "environment" else operators if args.case == "operators" else run_serve_case
        )
        record.update(function(args))
        write_json(args.report, record)
    except Exception as exc:
        if args.report.is_file():
            try:
                partial = json.loads(args.report.read_text())
                if isinstance(partial, dict):
                    record.update(partial)
            except (ValueError, OSError):
                pass
        record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        # Strip potentially unserializable partial details after a serialization failure.
        try:
            write_json(args.report, record)
        except (TypeError, ValueError):
            write_json(
                args.report,
                {"case": args.case, "status": "FAILED", "error": record["error"], "traceback": record["traceback"]},
            )
        traceback.print_exc()
    print(json.dumps({"case": args.case, "status": record["status"]}), flush=True)
    return 0 if record["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

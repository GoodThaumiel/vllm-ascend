# Local validation, 2026-10-20

This record concerns the standalone AscendC candidate. It does not accept L2.1.

| Check | Result |
|---|---|
| Standalone FP64 golden / contract / checker CPU tests | 11 passed |
| Existing ReplaySSM CPU regression tests | 37 passed |
| Ruff check and format check of new Python files | Passed, Ruff 0.17.1 |
| clang-format check of new C++ files | Passed, clang-format 23.1.3 |
| Python compileall and JSON parsing | Passed |
| git diff whitespace check | Passed |
| CANN compilation / NPU execution / graphs / latency | Not run; no CANN or NPU on developer Mac |

The standalone CPU tests ran on Python 3.12.15 / CPU PyTorch 2.14.1. NumPy is
absent; these tests use PyTorch directly and do not depend on NumPy. The server
baseline remains torch/torch_npu 2.10, CANN 9.2.0 and Ascend910B4. The local CPU
version does not establish ABI or compiler compatibility with that environment.

Existing HTTP fixture regressions required permission to bind local loopback
ports outside the sandbox. The initial sandbox run reported permission errors;
the authorized rerun passed all 37 tests.

The project's full `bash format.sh ci` check was initially unavailable because
pre-commit was absent. A temporary lint environment was installed to retry it;
its outcome is recorded below when complete.

First hardware action: follow A0/A1 only in
[the server checklist](../../docs/replayssm/checklists/LEVEL2_ASCENDC.md).
Report the full configure/build logs and smoke JSON/logs before advancing.

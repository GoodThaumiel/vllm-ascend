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

The initial full `bash format.sh ci` check required a temporary pre-commit
environment. Refer to the current commit's check results for its latest outcome.

First hardware action: follow A0/A1 only in
[the server checklist](../../docs/replayssm/checklists/LEVEL2_ASCENDC.md).
Report the full configure/build logs and smoke JSON/logs before advancing.

## Framework packaging iteration

The user subsequently reported all six standalone server test groups passed
at `3ff971b` with their CMake Release/RPATH correction. Raw reports were not
attached in that follow-up; timing figures are approximate user observations.
That result accepts the standalone candidate for packaging work. It does not
validate the new full-plugin build or model integration.

The moved kernel source is byte-identical to that commit. The actual shared
C++ schema and Meta function were compiled locally as a CPU-only test library:
six mutation aliases, Meta dispatch and FakeTensor tracing all passed (3 tests).
The 11 golden/tooling tests passed again. The local CPU-only test compile used
C++20 because local PyTorch 2.14 headers require it; production torch 2.10/CANN
build flags remain C++17. This does not establish the NPU ABI or packaging pass.

CPU-only reproduction (use a separate process, without the production extension):

```bash
cmake -S experimental/replayssm_ar_ascendc/tests/meta -B /tmp/replayssm-meta \
  -DPython3_EXECUTABLE="$(command -v python3)" -DCMAKE_CXX_STANDARD=20
cmake --build /tmp/replayssm-meta --parallel 2
python3 tests/ut/replayssm/test_framework_meta.py /tmp/replayssm-meta/libreplayssm_meta_test.dylib
```

Use `.so` for the final path on Linux. The library contains only registration
and Meta; it never links or pretends to execute the NPU kernel.
The next server action is F1 in [the framework checklist](../../docs/replayssm/checklists/LEVEL2_FRAMEWORK.md).

Current local results: 40 existing/budget CPU tests passed (the optional Meta
class is skipped during ordinary discovery), 11 standalone tests passed and
3 compiled Meta/FakeTensor tests passed separately: 54 checks in total.
Ruff, format, codespell, typos, clang-format, markdownlint and whitespace checks
passed. The full repository check still reports missing gitleaks/wget and
shellcheck tools, and existing forbidden `import re` in `level1_common.py` and
`level1_serve.py`. Those scripts and lint rules are outside this packaging change;
the full CI command is not claimed to pass. No NPU packaging result is claimed.

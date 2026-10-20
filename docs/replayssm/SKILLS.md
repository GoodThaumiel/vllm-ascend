# Cannbot skills installation record

Installed for the developer's Codex on 2026-10-18. No skills, agents, server
credentials, or local machine configuration are uploaded into this repository.

- Canonical source: <https://gitcode.com/cann/cannbot-skills.git>
- Audited source commit: `957277c7db3afc84b2bccf4baeba8966e5a9236d`
- Official installation instructions: <https://gitcode.com/cann/cannbot-skills/blob/master/docs/installation-guide.md>
- Installation target: `$CODEX_HOME/skills/` (the developer's default Codex skills directory).
- Method: clone the canonical source, validate each selected directory using the
  Codex skill-installer's validation helpers, and copy the complete skill folder
  using its local-copy helper. The GitHub-only downloader was not used to substitute
  an older mirror for the audited GitCode source.
- Source license and a machine-readable provenance record are retained alongside
  the local installation. Supporting references/scripts are copied; their
  environment setup or hardware scripts are not automatically executed.
- Skills become available for automatic discovery on the next user turn. A skill
  needed during this turn is read explicitly before use.

## Installed independent skills

| Group | Names |
|---|---|
| Environment and architecture | `ascendc-env-check`, `npu-arch` |
| AscendC development | `ascendc-api-best-practices`, `ascendc-tiling-design`, `ascendc-docs-search` |
| Precision and verification | `ascendc-precision-debug`, `ascendc-code-review`, `ascendc-st-design`, `ops-precision-standard` |
| Triton-Ascend | `triton-op-designer`, `triton-op-coding`, `triton-op-verifier` |
| Graph and custom ops | `torch-npugraph-ex-knowledge`, `torch-npugraph-ex-template`, `torch-custom-ops-guide` |

The independent-skills route is used instead of installing the full multi-agent
orchestration plugins. User-controlled, one-level-at-a-time validation takes
precedence over any skill workflow that assumes autonomous hardware access.

## Knowledge dependency

The official <https://gitcode.com/cann/cannbot-knowledge.git> repository is cloned
under the workspace's `.cannbot/dependencies/replayssm/cannbot-knowledge/`, outside
this Git checkout. This follows the skills' ancestor-directory discovery rule.
Do not add that dependency tree, installed skills, or downloaded runtimes to Git.
Pinned dependency revision: `7ea0efbeb79f2559f121fa8514a9fe5c82245807`.

The Cannbot website could not be reached from this execution environment; the
canonical GitCode repository and its installation guide were used instead.

## Application in this project

Level 1 uses `ascendc-env-check` to select the CANN/NPU evidence to collect. It
does not assume the developer has an NPU or infer the exact compiler SoC from a
generic `npu-smi` chip name. Level 2 AR uses `triton-op-designer` and
`triton-op-coding`, together with the pinned knowledge base's global,
linear-recurrent and gated-delta-rule constraints. Later levels will load only
the skills needed for the current operator or graph task, with sources and
versions recorded.

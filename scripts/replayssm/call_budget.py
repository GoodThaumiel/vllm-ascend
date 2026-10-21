# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Estimate pure-decode calls from a local model config, without importing vLLM."""

import argparse
import json
from pathlib import Path


def estimate(config: dict, tp: int, replay_us: float, native_us: float) -> dict:
    text = config.get("text_config", config)
    types = text.get("layer_types")
    if not types or len(types) != text["num_hidden_layers"]:
        raise ValueError("explicit layer_types must match num_hidden_layers")
    if tp <= 0 or replay_us < 0 or native_us < 0:
        raise ValueError("invalid TP or kernel latency")
    layers = types.count("linear_attention")
    return {
        "gdn_layers": layers,
        "tp": tp,
        "calls_per_rank_per_pure_decode_forward": layers,
        "calls_all_ranks": layers * tp,
        "illustrative_replay_ms": layers * replay_us / 1000,
        "illustrative_native_ms": layers * native_us / 1000,
        "illustrative_saved_ms": layers * (native_us - replay_us) / 1000,
        "batch_multiplier": 1,
        "flush_extra_calls": 0,
        "caveat": "Same batch/TP/tiling/timing required; eager Event gaps and overlap prevent end-to-end prediction",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tp", type=int, choices=(4, 8), required=True)
    parser.add_argument("--replay-us", type=float, default=150)
    parser.add_argument("--native-us", type=float, default=220)
    args = parser.parse_args()
    print(json.dumps(estimate(json.loads(args.config.read_text()), args.tp, args.replay_us, args.native_us), indent=2))


if __name__ == "__main__":
    main()

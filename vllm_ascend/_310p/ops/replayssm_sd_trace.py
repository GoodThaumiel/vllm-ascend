# SPDX-License-Identifier: Apache-2.0
"""Capture and replay a single request's exact MTP windows in eager serve."""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import torch

_active_trace: ContextVar = ContextVar("replayssm_310_trace", default=None)


def active_trace():
    return _active_trace.get()


class ReplaySSM310Trace:
    def __init__(self, config_path: str, rank: int):
        config = json.loads(Path(config_path).read_text())
        self.output = Path(config["output"]) / f"rank{rank}"
        self.output.mkdir(parents=True, exist_ok=False)
        reference = config.get("reference")
        self.reference = Path(reference) / f"rank{rank}" if reference else None
        self.tile = config.get("tile", 64)
        self.variant = config.get("status_variant", 1)
        if self.tile not in (32, 64) or self.variant not in (0, 1):
            raise ValueError("Replay trace requires tile 32/64 and status variant 0/1")
        self.step = 0
        self.record: dict = {}
        self.expected: dict | None = None
        self.request_id: str | None = None

    def _read_reference(self, step: int) -> dict | None:
        if self.reference is None:
            return None
        source = self.reference / f"{step:04d}.pt"
        if not source.exists():
            return None
        return torch.load(source, map_location="cpu", weights_only=True, mmap=True)

    def begin(self, request_id: str, drafts: list[int], computed: int, length: int) -> None:
        if self.request_id not in (None, request_id):
            raise ValueError("Replay trace captures one request per server run")
        self.request_id = request_id
        self.record = {
            "drafts": list(drafts),
            "computed": int(computed),
            "length": int(length),
            "tile": self.tile,
            "status_variant": self.variant,
            "layers": {},
        }
        self.expected = self._read_reference(self.step)
        if self.reference is not None and self.expected is None:
            raise ValueError(f"Replay trace exhausted the native reference at step {self.step}")
        if self.expected is not None:
            for key in ("drafts", "computed", "length"):
                if self.record[key] != self.expected[key]:
                    raise ValueError(f"Replay trace scheduler mismatch at step {self.step}: {key}")

    @contextmanager
    def forward(self, input_ids: torch.Tensor):
        tokens = input_ids[: self.record["length"]].detach().cpu().clone()
        self.record["input_ids"] = tokens
        if self.expected is not None and not torch.equal(tokens, self.expected["input_ids"]):
            raise ValueError(f"Replay trace input token mismatch at step {self.step}")
        marker = _active_trace.set(self)
        try:
            yield
        finally:
            _active_trace.reset(marker)

    def layer(self, name: str, **tensors: torch.Tensor) -> None:
        fields = self.record["layers"].setdefault(name, {})
        fields.update((key, value.detach().cpu().clone()) for key, value in tensors.items())

    def finish(self, logits: torch.Tensor, sampled: torch.Tensor, original_sampled: torch.Tensor) -> None:
        self.record["logits"] = logits
        self.record["sampled"] = sampled.detach().cpu().clone()
        self.record["original_sampled"] = original_sampled
        destination = self.output / f"{self.step:04d}.pt"
        torch.save(self.record, destination)
        destination.with_suffix(".json").write_text(json.dumps({"drafts": self.record["drafts"]}))
        self.step += 1
        self.record = {}
        self.expected = None

    def next_drafts(self) -> list[int] | None:
        if self.reference is None:
            return None
        source = self.reference / f"{self.step:04d}.pt"
        if not source.exists():
            return []
        # Only the small schedule is needed here, not the layer snapshots.
        return json.loads(source.with_suffix(".json").read_text())["drafts"]

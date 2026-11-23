# SPDX-License-Identifier: Apache-2.0
"""CPU checks for 310P request-owned ring capacity and lifecycle."""

import ast
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch


def _load_cpu_modules():
    root = Path(__file__).resolve().parents[3] / "vllm_ascend"
    packages = (
        "vllm_ascend",
        "vllm_ascend.ops",
        "vllm_ascend.ops.replayssm",
        "vllm_ascend._310p",
        "vllm_ascend._310p.ops",
    )
    modules = (
        ("vllm_ascend.ops.replayssm.cache", root / "ops/replayssm/cache.py"),
        ("vllm_ascend._310p.ops.replayssm_sd_trace", root / "_310p/ops/replayssm_sd_trace.py"),
        ("vllm_ascend._310p.ops.replayssm_sd", root / "_310p/ops/replayssm_sd.py"),
        ("vllm_ascend._310p.ops.replayssm_sd_demo", root / "_310p/ops/replayssm_sd_demo.py"),
    )
    names = (*packages, *(name for name, _ in modules))
    saved = {name: sys.modules.get(name) for name in names}
    try:
        for name in packages:
            package = ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package
        for name, path in modules:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return tuple(sys.modules[name] for name, _ in modules)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


_cache, _trace, _serve, _demo = _load_cpu_modules()
ReplaySSM310PGeometry = _demo.ReplaySSM310PGeometry
ReplaySSM310PRingDemo = _demo.ReplaySSM310PRingDemo
estimate_memory = _demo.estimate_memory
geometry_from_hf_config = _demo.geometry_from_hf_config


class TestReplaySSM310PDemo(unittest.TestCase):
    def test_capacity_and_allocation(self):
        geometry = ReplaySSM310PGeometry(1, 8, 16, 2, 3)
        demo = ReplaySSM310PRingDemo(geometry)
        report = estimate_memory(geometry, active_requests=2)
        self.assertEqual(geometry.ring_len, 32)
        self.assertEqual(demo.allocated_bytes(demo.allocate_rings()), report["ring_bytes_allocated"])

    def test_slot_reuse_and_no_active_alias(self):
        geometry = ReplaySSM310PGeometry(1, 8, 16, 2, 5)
        demo = ReplaySSM310PRingDemo(geometry)
        self.assertEqual(demo.update(["a", "b"]), {"ring_slots": [1, 2], "do_reset": [1, 1]})
        self.assertEqual(demo.update(["b", "c"], ["a"]), {"ring_slots": [2, 1], "do_reset": [0, 1]})
        with self.assertRaises(RuntimeError):
            demo.update(["b", "c", "d"])

    def test_projection_uses_fixed_capacity(self):
        geometry = ReplaySSM310PGeometry(30, 8, 16, 32, 3)
        at_one = estimate_memory(geometry, active_requests=1)
        at_full = estimate_memory(geometry, active_requests=32)
        self.assertEqual(at_one["ring_bytes_allocated"], at_full["ring_bytes_allocated"])
        self.assertLess(at_one["projected_saved_bytes"], 0)
        self.assertGreater(at_full["projected_saved_bytes"], 0)

    def test_hf_config_geometry(self):
        config = {
            "text_config": {
                "layer_types": ["linear_attention", "full_attention", "linear_attention"],
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
            }
        }
        geometry = geometry_from_hf_config(config, tp=2, max_num_seqs=32, mtp_tokens=5)
        self.assertEqual((geometry.linear_layers, geometry.local_key_heads, geometry.local_value_heads), (2, 8, 16))
        with self.assertRaises(ValueError):
            geometry_from_hf_config(config, tp=3, max_num_seqs=32, mtp_tokens=5)


class TestReplaySSM310PServe(unittest.TestCase):
    def test_deferred_request_keeps_accepted_prefix_and_ring_after_batch_reorder(self):
        source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"
        tree = ast.parse(source.read_text())
        runner_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner"
        )
        runner_class.bases = [ast.Name(id="BaseRunner", ctx=ast.Load())]
        runner_class.body = [
            node
            for node in runner_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("_update_states", "_restore_replayssm_sd_deferred_state")
        ]

        class BaseRunner:
            def _update_states(self, output):
                self.input_batch.req_id_to_index = {req: row for row, req in enumerate(output.num_scheduled_tokens)}
                self.input_batch.num_accepted_tokens_cpu[:] = [1, 1]
                self.input_batch.replayssm_decode_base[:] = [4096, 4096]

        namespace = {
            "BaseRunner": BaseRunner,
            "Callable": object,
            "replayssm_sd_enabled": _cache.replayssm_sd_enabled,
        }
        module = ast.fix_missing_locations(ast.Module(body=[runner_class], type_ignores=[]))
        exec(compile(module, str(source), "exec"), namespace)
        runner = namespace["NPUModelRunner"]()
        runner.vllm_config = SimpleNamespace(additional_config={"replayssm_sd": True})
        runner.use_async_scheduling = False
        runner._replayssm_sd_deferred_state = {}
        slots = runner._replayssm_sd_ring_slots = _cache.ReplaySSMSDRingSlots(2)
        slots.assign(["paused", "finished"])
        runner.input_batch = SimpleNamespace(
            req_id_to_index={"paused": 0, "finished": 1},
            replayssm_decode_base=[2048, 100],
            num_accepted_tokens_cpu=[1, 1],
        )
        accepted = runner.input_batch.num_accepted_tokens_cpu
        runner.num_accepted_tokens = SimpleNamespace(gpu=torch.ones(2, dtype=torch.int32))
        event = runner.num_accepted_tokens_event = Mock()
        event.synchronize.side_effect = lambda: accepted.__setitem__(0, 6)
        runner._apply_pp_sampled_tokens_from_scheduler_output = Mock()
        runner._track_tmp_encoder_cache_refs = Mock()
        output = SimpleNamespace(
            finished_req_ids={"finished"},
            preempted_req_ids=set(),
            scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
            num_scheduled_tokens={"new": 8},
            kv_cache_block_copies=[],
        )
        runner._update_states(output)
        event.synchronize.assert_called_once()
        self.assertEqual(slots.slots_for(["new", "paused"]), [2, 1])
        self.assertEqual(runner._replayssm_sd_deferred_state, {"paused": (2048, 6)})

        output.finished_req_ids = set()
        output.num_scheduled_tokens["paused"] = 8
        runner._update_states(output)
        runner._restore_replayssm_sd_deferred_state()
        self.assertEqual(runner.input_batch.replayssm_decode_base[1], 2048)
        self.assertEqual(accepted, [1, 6])
        self.assertEqual(runner.num_accepted_tokens.gpu.tolist(), [1, 6])
        self.assertEqual(runner._replayssm_sd_deferred_state, {})

    def test_trace_reuses_native_drafts_and_rejects_different_inputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps({"output": str(root / "native")}))
            native = _trace.ReplaySSM310Trace(str(config), 0)
            for drafts in ([], [5, 6, 7]):
                native.begin("native", drafts, 0, 1)
                with native.forward(torch.tensor([3])):
                    self.assertIs(_trace.active_trace(), native)
                native.finish(torch.zeros(1, 8), torch.tensor([[4]]), torch.tensor([[4]]))

            config.write_text(json.dumps({"output": str(root / "replay"), "reference": str(root / "native")}))
            replay = _trace.ReplaySSM310Trace(str(config), 0)
            replay.begin("replay", [], 0, 1)
            with self.assertRaisesRegex(ValueError, "input token mismatch"), replay.forward(torch.tensor([9])):
                self.fail("A mismatched input must be rejected before model forward")
            with replay.forward(torch.tensor([3])):
                self.assertIs(_trace.active_trace(), replay)
            replay.finish(torch.zeros(1, 8), replay.expected["sampled"], torch.tensor([[2]]))
            self.assertEqual(replay.next_drafts(), [5, 6, 7])

    def test_metadata_backing_covers_last_m200_block(self):
        # A 64-request service uses physical slot 64 plus reserved slot 0.
        for dtype in (torch.int32, torch.int8):
            elem_size = torch.empty(0, dtype=dtype).element_size()
            backing = torch.empty(_serve.aligned_metadata_size(65, elem_size), dtype=dtype)
            view = backing[:65]
            block_elems = 32 // elem_size
            last_block_end = (64 // block_elems + 1) * block_elems
            self.assertGreaterEqual(view.untyped_storage().nbytes(), last_block_end * elem_size)
            self.assertEqual(view.numel(), 65)

    def test_verify_dispatch_for_full_windows(self):
        for k in (3, 7):
            with self.subTest(k=k):
                self._check_verify_dispatch((k + 1, k + 1), k, "mtp", (k + 1, 1, 1))

    def test_verify_dispatch_for_short_windows(self):
        for k in (3, 7):
            with self.subTest(k=k):
                self._check_verify_dispatch((k + 1, 1, 2), k, "mtp", (k + 1, 1, 1))

    def test_dflash_dispatch_for_full_and_short_windows(self):
        # K=4/5 keep physical width 6 and use the supported vector verify.
        for k, width, variant in ((1, 4, 2), (2, 4, 2), (3, 4, 2), (4, 6, 1), (5, 6, 1), (6, 8, 2), (7, 8, 2)):
            self.assertEqual(_cache.replayssm_sd_ring_len(16, k), 32)
            for lengths in ((k + 1, k + 1), (k + 1, 1, 2)):
                with self.subTest(k=k, lengths=lengths):
                    self._check_verify_dispatch(lengths, k, "dflash", (width, variant, 2))

    def test_mtp7_ring_keeps_32_entries_and_96_request_slots(self):
        config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=96))
        self.assertEqual(_cache.replayssm_sd_ring_len(16, 7), 32)
        self.assertEqual(_cache.replayssm_sd_ring_capacity(config), 96)

    def _check_verify_dispatch(self, lengths, k, method, dispatch):
        width, verify_variant, compact_variant = dispatch
        rows = len(lengths)
        tokens = sum(lengths)
        lengths = torch.tensor(lengths, dtype=torch.int32)
        pack = _cache.replayssm_sd_pack_indices(lengths, k + 1, width)
        v = torch.arange(tokens, dtype=torch.float16).view(1, tokens, 1, 1).expand(1, tokens, 16, 128)
        q = torch.ones(1, tokens, 8, 128, dtype=torch.float16)
        gates = torch.ones(1, tokens, 16)
        slots = torch.arange(1, rows + 1, dtype=torch.int32)
        ring_slots = slots.flip(0)
        zero = torch.zeros(rows, dtype=torch.int32)
        meta = SimpleNamespace(
            replayssm_sd_block_slots=slots,
            replayssm_sd_ring_slots=ring_slots,
            replayssm_sd_positions=zero,
            replayssm_sd_bases=zero,
            replayssm_sd_flush=zero,
            replayssm_sd_lengths=lengths,
            replayssm_sd_pack_indices=pack,
        )
        if tokens != rows * width:
            meta.replayssm_sd_token_gather = _serve.replay_token_gather(pack, rows, width)
        # Match the service's nonzero aligned checkpoint offset and make its
        # physical block IDs independent of the request-owned ring IDs.
        checkpoint = torch.empty(rows + 2, 16, 128, 128, dtype=torch.float16)[1:]
        layer = torch.nn.Module()
        layer.num_k_heads = 16
        layer.num_v_heads = 32
        layer.tp_size = 2
        layer.head_k_dim = layer.head_v_dim = 128
        layer.A_log = torch.empty(0)
        layer.kv_cache = None, checkpoint
        layer._replayssm_sd_d_ring = torch.empty(0)
        layer._replayssm_sd_k_ring = torch.empty(0)
        layer._replayssm_sd_g_ring = torch.empty(0)
        _serve.initialize_replayssm_sd_310p(
            layer,
            SimpleNamespace(
                additional_config={"replayssm_sd": True},
                speculative_config=SimpleNamespace(method=method, num_speculative_tokens=k),
                scheduler_config=SimpleNamespace(max_num_seqs=rows),
                compilation_config=SimpleNamespace(max_cudagraph_capture_size=0),
            ),
        )
        self.assertTrue(layer.use_replayssm_sd)
        ops = Mock()

        def fake_verify(*args):
            self.assertEqual(args[0].shape, (rows, width, 8, 128))
            self.assertEqual(args[3].dtype, torch.float32)
            self.assertEqual(args[4].dtype, torch.float16)
            self.assertIs(args[5], checkpoint)
            self.assertIs(args[9], slots)
            self.assertEqual(args[14].numel(), rows * 16 * 2)
            self.assertEqual(args[16:18], (64, 7))
            self.assertIs(args[18], ring_slots)
            self.assertIs(args[12], lengths)
            self.assertEqual(args[19], verify_variant)
            args[13].copy_(args[2] + 1)

        ops.verify_.side_effect = fake_verify
        with patch.object(torch.ops, "replayssm_sd_310p", ops):
            output = _serve.replayssm_sd_310p_decode(layer, q, q, v, gates, gates, meta)

        torch.testing.assert_close(output, v + 1)
        self.assertEqual([call[0] for call in ops.mock_calls], ["verify_", "compact_"])
        compact = ops.compact_.call_args.args
        self.assertIs(compact[0], checkpoint)
        self.assertIs(compact[4], slots)
        self.assertEqual(compact[8].numel(), rows * 16 * 4)
        self.assertEqual(compact[9:11], (32, 7))
        self.assertIs(compact[11], ring_slots)
        self.assertEqual(compact[12], compact_variant)


if __name__ == "__main__":
    unittest.main()

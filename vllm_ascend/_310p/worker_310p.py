#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
import gc

import psutil
import torch
import torch_npu
from vllm.logger import logger
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import MemorySnapshot, memory_profiling
from vllm.utils.torch_utils import set_random_seed  # noqa: E402
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups

from vllm_ascend import envs
from vllm_ascend._310p.dflash_kv_layout import dflash_shared_kv_layout
from vllm_ascend._310p.kv_cache_sharing import get_310p_dflash_private_page_bytes, get_310p_shared_cache_slots
from vllm_ascend._310p.model_runner_310p import NPUModelRunner310
from vllm_ascend._310p.ops.compute_wy import load_compute_wy_library
from vllm_ascend.ops.replayssm.cache import replayssm_sd_enabled
from vllm_ascend.utils import is_rc_device
from vllm_ascend.worker.worker import NPUWorker, init_workspace_manager


class NPUWorker310(NPUWorker):
    def _scale_kv_cache_memory_for_multi_group(self, available_memory: int, *, mode: str = "physical") -> int:
        kv_cache_spec = self.get_kv_cache_spec()
        layout_resolver = getattr(self.vllm_config.cache_config, "get_resolved_kv_cache_layout", None)
        if isinstance(kv_cache_spec, dict) and callable(layout_resolver):
            groups = get_kv_cache_groups(self.vllm_config, kv_cache_spec)
            if layout := dflash_shared_kv_layout(self.vllm_config, groups):
                planner_bytes = max(len(group.layer_names) * group.kv_cache_spec.page_size_bytes for group in groups)
                # v24 charged one logical page per slot against its half-budget.
                # The runner still allocates both backing families in either mode.
                divisor = planner_bytes if mode == "v24" else layout.bytes_per_pool_block
                num_blocks = available_memory // divisor
                backing_bytes = num_blocks * layout.bytes_per_pool_block
                logger.info(
                    "310P DFlash shared KV budget: physical_budget_bytes=%d physical_bytes_per_pool_block=%d "
                    "blocks_from_budget=%d mode=%s input_budget_bytes=%d planner_bytes_per_pool_block=%d "
                    "planned_backing_bytes=%d",
                    backing_bytes if mode == "v24" else available_memory,
                    layout.bytes_per_pool_block,
                    num_blocks,
                    mode,
                    available_memory,
                    planner_bytes,
                    backing_bytes,
                )
                return num_blocks * planner_bytes
            if mode == "v24":
                raise ValueError("310P DFlash v24 KV budget requires the shared KV layout")
            private_pages = get_310p_dflash_private_page_bytes(self.vllm_config, groups)
            if private_pages is not None:
                logical_group_bytes = [len(group.layer_names) * group.kv_cache_spec.page_size_bytes for group in groups]
                physical_bytes = sum(len(group.layer_names) * page for group, page in zip(groups, private_pages))
                # The upstream planner divides the reported budget by the
                # largest logical group. Translate the physical budget so its
                # block count fits the 310P runner's private layer buffers.
                num_blocks = available_memory // physical_bytes
                logger.info(
                    "310P DFlash KV budget: physical_budget_bytes=%d old_bytes_per_pool_block=%d "
                    "physical_bytes_per_pool_block=%d blocks_from_budget=%d old_blocks_from_budget=%d mode=%s",
                    available_memory,
                    sum(logical_group_bytes),
                    physical_bytes,
                    num_blocks,
                    available_memory // sum(logical_group_bytes),
                    mode,
                )
                return num_blocks * max(logical_group_bytes)
            share_slots = get_310p_shared_cache_slots(groups, layout_resolver()) if groups else {}
        else:
            share_slots = {}
        if mode != "physical":
            raise ValueError(f"310P DFlash KV budget mode {mode!r} requires a supported MRv1 DFlash cache layout")
        if share_slots:
            # Both 310P runners alias compatible Mamba groups at each slot,
            # matching the v0.28 shared_by allocation.
            logger.info_once("310P reuses uniform KV cache group slots without extra budget scaling.", scope="local")
            return available_memory
        return super()._scale_kv_cache_memory_for_multi_group(available_memory)

    def _create_model_runner(self):
        if self.use_v2_model_runner:
            from vllm_ascend._310p.worker.v2.model_runner import NPUModelRunner310V2

            model_runner = NPUModelRunner310V2(self.vllm_config, self.device)
            logger.info_once("Using NPUWorker310 and NPUModelRunner310V2.")
            return model_runner

        model_runner = NPUModelRunner310(self.vllm_config, self.device)
        logger.info_once("Using NPUWorker310 and NPUModelRunner310.")
        return model_runner

    def init_device(self):
        self.device = self._init_device()
        torch_npu.npu.set_compile_mode(jit_compile=False)
        if (compute_wy_library := load_compute_wy_library()) is not None:
            logger.info("310P AscendC compute WY enabled rank=%d library=%s", self.rank, compute_wy_library)

        init_workspace_manager(self.device, num_ubatches=1)
        self.model_runner = self._create_model_runner()

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        from vllm_ascend._310p.sharded_state_loader_310p import ShardedStateLoader310

        ShardedStateLoader310.save_model(
            self.model_runner.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

        ShardedStateLoader310.generate_quant_description(
            self.model_runner.model,
            path,
            self.vllm_config.quant_config,
        )

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how much
        memory can be used for KV cache without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculates the free memory that can be used for KV cache in
        bytes.
        """
        GiB = lambda b: b / GiB_bytes
        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with memory_profiling(
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            if replayssm_sd_enabled(self.vllm_config):
                self.model_runner.initialize_replayssm_sd_ring()
            self.model_runner.profile_run()
            free_memory, total_memory = torch.npu.mem_get_info()
            # The host memory or device memory for RC devices refers to the available portion of memory
            # which cannot be obtained via torch.npu.mem_get_info()
            if is_rc_device():
                free_memory = psutil.virtual_memory().available
            torch_memory = torch.npu.memory_reserved()
            non_torch_memory_before_empty_cache = total_memory - free_memory - torch_memory

        self.non_torch_memory = profile_result.non_torch_increase
        self.peak_activation_memory = profile_result.torch_peak_increase
        non_torch_memory_cleared_by_empty_cache = non_torch_memory_before_empty_cache - self.non_torch_memory

        free_gpu_memory = profile_result.after_profile.free_memory
        assert self.init_snapshot.free_memory > free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {GiB(self.init_snapshot.free_memory)} GiB, "
            f"current free memory {GiB(free_gpu_memory)} GiB. "
            "This happens when other processes sharing the same container "
            "release GPU memory while vLLM is profiling during initialization. "
            "To fix this, ensure consistent GPU memory allocation or "
            "isolate vLLM in its own container."
        )

        # Divide the available memory by 2, to reserved more memory for other operators workspace and other cache
        # This could avoid OOM with default gpu_memory_utilization
        # The 310P RC device shares the host memory and device memory.
        # Therefore, the space available for allocating KV cache and Mamba cache needs to be calculated
        # based on the already occupied space of the system memory.

        if is_rc_device():
            vm = psutil.virtual_memory()
            remaining_memory = self.requested_memory - (vm.total - vm.available)
            physical_free_bytes = vm.available
        else:
            remaining_memory = (
                self.requested_memory - profile_result.non_kv_cache_memory - non_torch_memory_cleared_by_empty_cache
            )
            physical_free_bytes = free_gpu_memory

        mode = "physical"
        input_budget = remaining_memory // 2
        spec_config = self.vllm_config.speculative_config
        if spec_config is not None and spec_config.method == "dflash" and not self.use_v2_model_runner:
            explicit_bytes = self.cache_config.kv_cache_memory_bytes
            if explicit_bytes is not None:
                if not 0 < explicit_bytes <= physical_free_bytes:
                    raise ValueError(
                        f"310P DFlash explicit KV budget {explicit_bytes} bytes must be positive and fit within "
                        f"the {physical_free_bytes} free bytes after profiling"
                    )
                mode = "explicit"
                input_budget = explicit_bytes
            elif envs.VLLM_ASCEND_310P_DFLASH_V24_KV_BUDGET:
                mode = "v24"
            logger.info(
                "310P DFlash KV memory after profiling: mode=%s requested_bytes=%d remaining_bytes=%d free_bytes=%d",
                mode,
                int(self.requested_memory),
                int(remaining_memory),
                physical_free_bytes,
            )

        self.available_kv_cache_memory_bytes = self._scale_kv_cache_memory_for_multi_group(int(input_budget), mode=mode)

        logger.debug(profile_result)
        logger.info_once(
            "Available KV cache memory for planner: %.2f GiB (budget mode %s)",
            GiB(self.available_kv_cache_memory_bytes),
            mode,
            scope="local",
        )
        return int(self.available_kv_cache_memory_bytes)

    def _warm_up_atb(self):
        # 310p device do not support torch_npu._npu_matmul_add_fp32 atb ops
        logger.info_once("Skip warm-up atb ops for 310P device.")

    def _init_device(self):
        device = torch.device(f"npu:{self.local_rank}")
        torch.npu.set_device(device)

        # This lazy import avoids torch_npu re-initialization in patch
        # Note that this should be imported after torch.npu.set_device
        # to avoid repeated set_device in extra processes

        gc.collect()
        torch.npu.empty_cache()

        # take current memory snapshot
        self.init_snapshot = MemorySnapshot(device=device)
        self.requested_memory = self.init_snapshot.total_memory * self.cache_config.gpu_memory_utilization
        if is_rc_device():
            self.init_snapshot.free_memory = psutil.virtual_memory().available
            logger.info_once("Root Complex (RC) mode: host and device memory are shared.")
        if self.init_snapshot.free_memory < self.requested_memory:
            GiB = lambda b: round(b / GiB_bytes, 2)
            raise ValueError(
                f"Free memory on device "
                f"({GiB(self.init_snapshot.free_memory)}/"
                f"{GiB(self.init_snapshot.total_memory)} GiB) on startup "
                f"is less than desired GPU memory utilization "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{GiB(self.requested_memory)} GiB). Decrease GPU memory "
                f"utilization or reduce GPU memory used by other processes."
            )

        if (
            self.parallel_config.data_parallel_size > 1
            and self.parallel_config.data_parallel_size_local > 0
            and self.parallel_config.distributed_executor_backend not in ["ray", "external_launcher"]
            and self.vllm_config.parallel_config.data_parallel_backend != "ray"
            and self.vllm_config.parallel_config.nnodes_within_dp == 1
        ):
            visible_device_count = torch.npu.device_count() if torch.npu.is_available() else 0
            assert self.parallel_config.local_world_size <= visible_device_count, (
                f"local_world_size ({self.parallel_config.local_world_size}) must "
                f"be less than or equal to the number of visible devices "
                f"({visible_device_count})."
            )

        # Initialize the distributed environment.
        self._init_worker_distributed_environment()
        # Set random seed.
        set_random_seed(self.model_config.seed)

        return device

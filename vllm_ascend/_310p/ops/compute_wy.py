# SPDX-License-Identifier: Apache-2.0
"""Load the opt-in AscendC WY preparation used by 310P chunk GDN."""

from pathlib import Path

import torch
from vllm.logger import logger

from vllm_ascend import envs


def load_compute_wy_library() -> Path | None:
    setting = envs.VLLM_ASCEND_310P_COMPUTE_WY_LIBRARY
    if not setting:
        return None
    library = Path(setting).expanduser()
    if not library.is_absolute() or not library.is_file():
        raise ValueError("VLLM_ASCEND_310P_COMPUTE_WY_LIBRARY must be an absolute path to the built .so")
    library = library.resolve()
    torch.ops.load_library(str(library))
    if not hasattr(torch.ops.replayssm_310p_compute_wy, "prepare"):
        raise RuntimeError(f"310P compute WY library did not register replayssm_310p_compute_wy.prepare: {library}")
    return library


def compute_wy_for_prefill(q, k, v, g, beta, chunk_size):
    """Return BHTD kernel inputs for supported shapes, otherwise defer to Torch."""
    if not envs.VLLM_ASCEND_310P_COMPUTE_WY_LIBRARY:
        return None
    if (
        q.device.type != "npu"
        or q.dtype != torch.float16
        or k.dtype != q.dtype
        or v.dtype != q.dtype
        or beta.dtype != q.dtype
        or g.dtype != torch.float32
        or chunk_size != 64
        or q.shape[0] > 32
        or v.shape[2] > 64
        or q.shape[-1] not in (64, 128)
        or v.shape[-1] not in (64, 128)
        or q.shape[2] <= 0
        or v.shape[2] <= 0
        or v.shape[2] % q.shape[2] != 0
        or q.shape[1] <= 0
        or q.shape[1] % chunk_size != 0
    ):
        logger.warning_once("310P compute WY is unavailable for this prefill shape or dtype. Using Torch WY.")
        return None
    return torch.ops.replayssm_310p_compute_wy.prepare(
        q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous(), chunk_size
    )

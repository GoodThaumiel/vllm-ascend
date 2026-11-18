# SPDX-License-Identifier: Apache-2.0
"""Select the grouped RGDR ABI loaded by the 310P worker."""

from functools import cache

import torch


@cache
def grouped_rgdr_launch_config() -> tuple[int, int | None]:
    op = getattr(torch.ops.rgdr_v310_cube_fp16, "run_", None)
    if op is None:
        raise RuntimeError("The 310P grouped RGDR library must register run_")
    arguments = {arg.name for arg in op.default._schema.arguments}
    if "row_offsets" not in arguments:
        raise RuntimeError("Rebuild the 310P grouped RGDR library with packed row support")
    if "probe" in arguments:
        # Stable arithmetic and two-head schedule validated in c98759c.
        return 8, 11
    # Compatibility with the original 0.30 MTP pilot library.
    return 7, None

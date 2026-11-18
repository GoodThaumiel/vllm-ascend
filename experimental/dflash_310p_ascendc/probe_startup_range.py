# SPDX-License-Identifier: Apache-2.0
"""Isolate an int32 Range used during MRv1 DFlash worker startup."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="Logical index among the visible NPUs.")
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--source", choices=("npu", "cpu"), default="npu")
    args = parser.parse_args()
    if args.length < 1:
        parser.error("--length must be positive")

    import torch
    import torch_npu

    device = torch.device("npu", args.device)
    print(f"torch={torch.__version__} torch_npu={torch_npu.__version__} device={device}", flush=True)
    torch.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)
    print("device_sync_begin", flush=True)
    torch.npu.synchronize(device)
    print("device_sync_end", flush=True)

    expected = torch.arange(args.length, dtype=torch.int32, device="cpu")
    print(f"range_begin source={args.source} length={args.length} dtype=int32", flush=True)
    if args.source == "npu":
        result = torch.arange(args.length, dtype=torch.int32, device=device)
    else:
        result = expected.to(device)
    torch.npu.synchronize(device)
    print("range_end", flush=True)

    print("readback_begin", flush=True)
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    print(f"PASSED source={args.source} length={args.length}", flush=True)


if __name__ == "__main__":
    main()

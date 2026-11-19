# 310P GDN prefill · AscendC compute WY

这个实验替换 310P MRv1 chunk GDN 的 WY 准备阶段，可用于 TP2 DFlash 服务。输入仍由 0.30 的 GDN prefill 做归一化、按请求填充到 64-token chunk，状态更新和输出继续由现有的 `fwd_h` 与 `fwd_o` 计算。可和 grouped RGDR 的 decode/verify 开关同时启用。

算子计算沿用 [v24 DFlash 分支 1989b04](https://github.com/HaonanWu1/vllm-ascend/tree/1989b04c04e6ec5cb8c2e394bc8ebd107422196a/csrc/moe/chunk_gated_delta_rule_compute_wy) 的 310P 实现。它支持 128 维 head，保留原有分块三角计算及数值敏感输入的 FP32 分支。公开 [#11941](https://github.com/vllm-project/vllm-ascend/pull/11941) 的当前版本限制为 64 维。[#13122](https://github.com/vllm-project/vllm-ascend/pull/13122) 支持 128 维，使用另一套低阶 Cube 实现。[#12607](https://github.com/vllm-project/vllm-ascend/pull/12607) 提交的是 A3 的整段 fused prefill。本目录只保留一个 310P AscendC kernel，调用改为直调，Matmul tiling 和 workspace 由独立 Torch binding 提供。

## 构建与单算子校验

在 310P3、CANN 9.2.1、当前服务使用的 Python/Torch/torch_npu 环境里执行。Torch 库位于 `$WY_BUILD/libreplayssm_310p_compute_wy.so`，CANN kernel 库位于 `$WY_BUILD/lib/`。Torch 库通过 ELF RUNPATH 中的 `$ORIGIN/lib` 找到 kernel。

```bash
cd experimental/chunk_gdr_310p_ascendc
export WY_BUILD="$PWD/build-c921"
cmake -S . -B "$WY_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DSOC_VERSION=Ascend310P3 -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.1 \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build "$WY_BUILD" --target replayssm_310p_compute_wy -j8
export VLLM_ASCEND_310P_COMPUTE_WY_LIBRARY="$WY_BUILD/libreplayssm_310p_compute_wy.so"
readelf -d "$VLLM_ASCEND_310P_COMPUTE_WY_LIBRARY" | grep -F '$ORIGIN/lib'
ldd "$VLLM_ASCEND_310P_COMPUTE_WY_LIBRARY" | grep libreplayssm_310p_compute_wy_kernel

cd ../..
python3 -m experimental.chunk_gdr_310p_ascendc.smoke --device 0
```

确认 `ldd` 将 kernel 解析到本次构建的 `$WY_BUILD/lib/`。Smoke 使用现有 vLLM-Ascend 的 `fwd_h/fwd_o` 扩展。检查 64/128 维的 Q/K 转排、W/U 和 FP32 累计门控，并用有尾块的两条 packed 请求比较整个 chunk GDN 的输出和 checkpoint。

用 `--stage wy` 单测 WY，参考值和比较都在 CPU。用 `--stage chunk` 比较 AscendC WY 与 CPU Torch WY，两个分支都调用实际服务的 chunk GDN 入口和同一套 NPU `fwd_h/fwd_o`，最后在 CPU 比较输出和 checkpoint。脚本分别同步输入填充、WY、`fwd_h` 和 `fwd_o`。若卡住，保留最后一条 `[compute WY smoke]` 和它之后出现的设备日志。

## 服务验证

设置 `VLLM_USE_V2_MODEL_RUNNER=0`，将上述库路径 export 到启动 `vllm serve` 的 shell。重启后核对两个 rank 的 `310P AscendC compute WY enabled` 日志，并在 prefill profiling 中查找 `replayssm_310p_compute_wy`。当形状或 dtype 不适用时，该 worker 会记录一次回退日志。清除库路径并重启服务即可复跑 Torch WY 基线。

先跑 C1 和混合长度请求，检查输出及 CANN 错误，再用 [DFlash 清单 §9.2](../dflash_310p_ascendc/LEVEL2.md#92-vllm-serve-双图与-2k400) 的 TP2、2k/400、C48 命令做开关对照。保持 K、RGDR 开关、显存参数和 capture sizes 一致，记录 prefill 时间、TTFT、实际 running、吞吐及两个 rank 的 kernel 时间。若可用 KV 容量变化，附上每轮的 cache blocks 和 prefill 调度批次 token 数。

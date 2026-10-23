# L3.2：SD verify 框架打包与原生算子 roofline 对照

L3.1 独立算子由用户在 `869b961`、CANN 9.2.0、Ascend910B4 上回传 TP4/TP8 × B1/32/64 × 8/20/40 核共 18 组 `perf` 全通过。40 核下 TP4 B32、T4、history 7/12 的图回放 p50 分别为 986.15/1302.61 µs；TP8 为 525.16/684.45 µs。最大回传误差约 1.22e-4。这个结果接受独立非 flush verify 的正确性和性能基线，不接受完整 MTP serving。

本轮把同一份 AscendC kernel 和 C++ launcher 纳入 `vllm_ascend_C`，注册 `_C_ascend.replayssm_sd_verify_` 与 Meta；独立 lab 构建继续共用源码。`vllm_ascend.ops.replayssm.ascendc.replayssm_sd_verify` 是框架 Python 入口。**没有**接入 GDN speculative dispatch 或修改 scheduler/cache allocator；SD flush、accepted-token cursor commit/reset 未完成，`vllm serve --use-replayssm` 的 MTP 仍应由现有保护拒绝。

## F1：框架重新编译和冒烟

在同一容器拉取本提交后，沿用已验证的源码安装流程重编译扩展；仅拉取 Python 文件不足以注册新 op。常见入口：

```bash
python3 -m pip install --no-build-isolation --no-deps -v -e .
python3 experimental/replayssm_sd_ascendc/run.py --backend framework \
  --stage smoke --tp 4 --batch 2 --cores 40 --device 0 \
  --output /tmp/replayssm-sd-framework-smoke-tp4.json
python3 experimental/replayssm_sd_ascendc/run.py --backend framework \
  --stage graph --tp 4 --batch 2 --cores 40 --device 0 \
  --output /tmp/replayssm-sd-framework-graph-tp4.json
```

- [ ] 两个 JSON 的 `status=PASSED`，`library` 指向安装的 `vllm_ascend_C`，而非独立 lab `.so`。
- [ ] `torch.ops._C_ascend.replayssm_sd_verify_` 存在；无 missing symbol、RPATH、Meta、NPUGraph capture 错误。
- [ ] checkpoint 不变，候选 d/k/g 和输出与 golden 一致，NPU 无残留异常。
- [ ] TP4 过关后同样跑 TP8；失败时回传首个 JSON、构建日志、设备日志，本级停止。

## F2：先小范围对照，再抓完整 profiling

脚本直接调用当前 `vllm_ascend.ops.gdn` 的原生 `recurrent_gated_delta_rule`（带 `actual_seq_lengths`、每候选独立 state slot 和 `num_accepted_tokens`），以及框架内的 Replay SD verify。默认 `--accepted 1`；可单独用 `--lengths 4 --accepted 4` 检查另一种上轮接受数。两侧共享同一批 BF16 q/k/v、FP32 g、按 BF16 量化后的 beta；原生初始 BF16 state 从 Replay checkpoint 和已提交的 FP16/FP32 ring 重建。先比**第一次输出**，再分别进入标记的测量区间，每条路径默认连续调用 100 次。区间内没有输入准备、状态重置或 CPU 读回。原生状态会随重复调用继续变化；这不构成连续 100 次服务请求的端到端语义，只有第一次输出作精度验收。

先在空闲卡做 2 次小冒烟：

```bash
python3 scripts/replayssm/profile_sd_operator_compare.py \
  --tp 4 --device 0 --batches 1 --context-lengths 32768 \
  --lengths 4 --positions 7 --repetitions 2 --warmup 0 \
  --output /tmp/replayssm-sd-roofline-smoke-tp4.json
```

首次输出误差与两侧调用通过后，用现有 profiling 工具包裹整个进程，按 `REPLAYSSM_SD/<case>/native` 与 `/replay` 区间过滤。建议先 TP4，再 TP8；两次使用不同报告路径：

```bash
python3 scripts/replayssm/profile_sd_operator_compare.py \
  --tp 4 --device 0 --batches 1,16,32,64 --context-lengths 32768 \
  --lengths 1,4 --positions 0,7,12 --ring 32 --tile-v 32 --cores 40 \
  --repetitions 100 --warmup 2 \
  --output /tmp/replayssm-sd-roofline-tp4.json \
  2>&1 | tee /tmp/replayssm-sd-roofline-tp4.log
```

- [ ] 每个 case 的 `output_max_abs_error` 低于 0.05，Replay status 全 0；任何失败停在该 case，回传 JSON 和日志。
- [ ] 分别汇总原生/Replayssm kernel 的设备 duration、真实 HBM/UB bytes、Vector/Scalar/Cube 利用率、核数和 FLOPs；JSON 中的读写 byte 下界仅作建模参考，不代替 profiler counters。
- [ ] 给出 T1/T4、history 0/7/12、B1/16/32/64 的同场景对照；重点看 B32/64、T4、history 12。
- [ ] `context-lengths` 只是标签：单次 recurrent 算子没有 4k/8k/32k 输入维度。长上下文造成的 KV、FIA、调度和排队影响以后用 `vllm serve` 测。
- [ ] JSON 会记录原生每候选 state slot 分配与 Replay page-backed ring 分配；这仅是单层状态 backing 对照，不是完整服务显存节省。

本级只验证插件打包和 recurrent 边界的性能/精度。若 B32/64、T4 仍明显偏慢，先利用 profiler 确认重复 dot/Scalar 同步、checkpoint matvec 与 DMA 占比，再决定共享点积表或 Cube 投影，不直接改变已通过的算子数学路径。

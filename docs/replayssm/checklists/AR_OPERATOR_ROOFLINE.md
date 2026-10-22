# Replay AR / native GDN 单算子 profiling 清单

该脚本在**同一个 vLLM Ascend 0.30 构建和 CANN 9.2.0 环境**中，对照两条真实调度入口：

- 原生：`vllm_ascend.ops.gdn.recurrent_gated_delta_rule`，即关闭 `--use-replayssm` 后 GDN decode 使用的 FLA NPU 算子。
- Replay：`vllm_ascend.ops.replayssm.ascendc.replayssm_ar_step`，即框架 torch binding。

输入是一个 TP rank 的 Qwen GDN 头部几何。默认 TP4、L=16、`tile_v=32`、1510 个 slot（来自上机显存报告）。状态采用相同的页步长布局：TP4 每页 524288 bytes，前 15360 bytes 是 conv state 区域，后面依次为 checkpoint、D、K、G。原生 state 由同一 Replay checkpoint + 已提交的历史 D/K/G 在 CPU 上重建，再量化为 BF16。两条路径使用相同的 q/k/v/g、相同的 BF16 量化 beta 值和相同的 active slot。脚本比较**首次调用输出**，之后只做重复调用与 profiling。测量区间没有缓存复制或重置操作。

## 重要口径

`4k/8k/32k` 是请求的上下文长度标签，并参与随机输入 seed；**单 token GDN recurrent 算子没有上下文长度参数**。Native 的 `actual_seq_lengths` 表示本次 decode 的 token 数，这里每个请求都是 1，不能填 4096/8192/32768。`mamba-cache-mode none` 下，固定 slot 数、batch、ring position 和 dtype 时，GDN 单算子的运算量与上下文长度无关。因而相同 batch/position 的三个长度点理论上应重合；若差距大，应检查系统噪声、缓存驻留或 profiler 分组。长上下文导致的 attention KV、FIA、调度变化应在 `vllm serve` 整网 profiling 中测量。

默认每个 batch×context×position 场景，Native 与 Replay **各调用 100 次**；`--warmup 0` 保证整脚本没有额外算子 warmup。`position=15` 执行 flush，0/8 不 flush。默认共 3×3×3=27 场景、两种算子共 5400 次测量调用。每个区间打印 `PROFILE_BEGIN/END` 并带 `torch.profiler.record_function` 标签；结果 JSON 记录每个场景的参数、页大小、实际启动 block 数和首次输出误差。Host 区间耗时包含调度，**不要用它代替设备 kernel 时延**。Roofline 的 FLOPs、HBM/UB 流量和持续时间以 NPU profiler 的实际 kernel counter 为准。

## 先做小范围冒烟

停止 `vllm serve`，确认目标卡空闲；使用此前验证过的 0.30.0+empty / 当前 vLLM Ascend 编译安装环境：

```bash
python scripts/replayssm/profile_ar_operator_compare.py \
  --tp 4 --device 0 --batches 1 --context-lengths 4096 --positions 0 \
  --repetitions 2 --cores 8 \
  --output /tmp/replayssm-ar-roofline-smoke.json
```

首次输出精度、Replay status、页布局均通过后，运行完整矩阵，并用你现有的 profiler 工具包裹**整个命令**：

```bash
python scripts/replayssm/profile_ar_operator_compare.py \
  --tp 4 --device 0 --batches 1,16,32 \
  --context-lengths 4096,8192,32768 --positions 0,8,15 \
  --capacity 16 --num-slots 1510 --tile-v 32 --cores 8 \
  --repetitions 100 --warmup 0 \
  --output /tmp/replayssm-ar-roofline-tp4.json \
  2>&1 | tee /tmp/replayssm-ar-roofline-tp4.log
```

TP8 改为 `--tp 8`，使用空闲的 TP8 rank 所在卡，输出文件另命名。改变 Replay 核数时只改 `--cores` 和输出文件；Native 算子不受该参数影响。BS1/TP8/`tile_v=32` 只有 24 个任务，传 32/40 仍只会启动 24 个 Replay block。

- [ ] 报告 `status=PASSED`，所有场景首次输出误差低于默认 `max_abs_error=0.05`，Replay status 全 0。
- [ ] 每个 `PROFILE_BEGIN ... native/replay` 到对应 `PROFILE_END` 之间恰好 100 次调用；按标签或 kernel 名区分两条路径。
- [ ] 报告的 TP4 state stride/页大小与服务日志一致（524288 bytes/page）；TP8 页大小另行核对。
- [ ] 分别给出 position 0/8/15、BS1/16/32 的设备 kernel duration、Vector/Scalar/MTE2 计数、实际核数；不要把 host 时间放上 roofline。
- [ ] 若三个 context 长度的单算子点不重合，回传 profiler trace 与环境占用；不要先推断 GDN 有上下文长度复杂度。
- [ ] 回传 JSON、脚本日志、profile 数据、源码 commit、torch-npu/CANN/FLA NPU 版本和 `npu-smi` 状态。

脚本没有运行模型前处理、48 层、Attention、TP 通信或 MTP。此处结论只能用于定位单算子瓶颈；整网收益仍以同构配置的 `vllm serve` 结果为准。

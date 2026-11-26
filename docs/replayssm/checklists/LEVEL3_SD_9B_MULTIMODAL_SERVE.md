# Qwen3.5-9B 多模态服务吞吐验证

在 Ascend910B4、CANN 9.2.0、vLLM `0.30.0+empty` 使用同一套源码版 vLLM Ascend。
使用带视觉编码器和 MTP 权重的 Qwen3.5-9B，BF16 模型、conv cache 和 SSM checkpoint，
PP1、MTP3。TP1 和 TP2 分开建立 native 基线。输入用真实多模态请求，模型侧 prompt
约 4k token，输出约 400 token。记录服务统计的实际长度和图片或视频规模。

| 路径 | `--additional-config` 中的 ReplaySSM 字段 | GDN 投机路径 |
|---|---|---|
| Native MTP3 | `"replayssm_sd":false` | Native recurrent GDR |
| ReplaySSM V13 | `"replayssm_sd":true,"replayssm_sd_verify_variant":13` | V13/tile128 + Compact v1/tile32，独立请求 ring |
| ReplaySSM V20 | `"replayssm_sd":true,"replayssm_sd_verify_variant":20` | V20/tile128 + Compact v1/tile32，独立请求 ring |

V13 是现有服务路径。V20 已编入主扩展，使用 8 KiB checkpoint 分块和批量 gate 写回，
本轮用于验证多模态服务吞吐。9B TP1 单层 B32/B64 的 V20 串行图为 809.0/1537.5 µs，
实验库 mixed AIC+AIV 串行图为 773.6/1464.2 µs，native 为 485.4/954.4 µs。
这些数据来自 pos15/T4、mixed flush 的单卡算子实验。V13 和 V20 的多模态服务排名待此轮测量。
mixed kernel、AIV 历史预取和 verify+compact 融合另有独立算子实验。

## Atlas A3 单 die

本轮 A3 目标为 `Ascend910_9362`。每个进程通过 `ASCEND_RT_VISIBLE_DEVICES` 看到一个逻辑
设备，使用 TP1。先用 `npu-smi info -m` 确认物理卡、die 与逻辑 ID 的对应，再用运行时
确认设备名及可用显存。CANN 编译选择 `SOC_VERSION=ascend910_9362`。Blue2 上已用
CANN 9.2.1、torch-npu 2.10.0.post7、vLLM `0.30.0+empty` 源码构建主扩展。

服务前先跑 [9B 算子检查](LEVEL3_SD_MTP_GEOMETRY.md)，核对 V13/V20 的图回放以及
Compact v1、MTP3 cursor。服务用下节 TP1 参数，先把 `MAX_SEQS` 设为 4，
`BATCH_TOKENS` 设为 4096，验证一条文本和一条真实图文请求。看到 ReplaySD
启用、图捕获完成及 MTP draft 计数增长后，再测 4k/400 和高并发。
报告随附 SoC、CANN、运行时包版本、主扩展 hash、KV blocks 和 TPOT。A3 使用自己的
同镜像 native MTP3 对照。

## 启动及冒烟

从仓库根目录执行。主扩展须包含 V20 launcher，已在该环境运行过 V20 算子测试的主扩展可复用。
更新 Python 源码后重启服务。以下命令选单个 TP 和单个路径，逐项重启。
将服务参数中的输入预算按实际图片或视频设置。起始的 `MAX_LEN=8192` 覆盖约 4k 的
**模型侧总 prompt** 加 400 输出，并为视觉 token 波动留余量。若 4k 指纯文本，先查看
加入视觉 token 后的实际长度。保留视觉编码器，按相同配置启用多模态处理。

```bash
set -euo pipefail
: "${REPLAYSSM_9B_MODEL:?set path to Qwen3.5-9B}"
TP=${TP:-1}
MODE=${MODE:-v20}
MAX_SEQS=${MAX_SEQS:-64}
MAX_LEN=${MAX_LEN:-8192}
BATCH_TOKENS=${BATCH_TOKENS:-8192}
MEM_UTIL=${MEM_UTIL:-0.85}
R=${R:-/tmp/replayssm-9b-mm}
mkdir -p "$R"
case "$MODE" in
  native) EXTRA='{"replayssm_sd":false}' ;;
  v13) EXTRA='{"replayssm_sd":true,"replayssm_sd_verify_variant":13}' ;;
  v20) EXTRA='{"replayssm_sd":true,"replayssm_sd_verify_variant":20}' ;;
  *) echo "MODE must be native, v13 or v20" >&2; exit 2 ;;
esac
export VLLM_ASCEND_REPLAYSSM_MEMORY_PROBE=1
vllm serve "$REPLAYSSM_9B_MODEL" \
  --host 127.0.0.1 --port 18083 --served-model-name qwen35-9b \
  --tensor-parallel-size "$TP" --pipeline-parallel-size 1 \
  --dtype bfloat16 --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype bfloat16 \
  --mamba-cache-mode none --no-enable-prefix-caching --enable-chunked-prefill \
  --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$BATCH_TOKENS" \
  --gpu-memory-utilization "$MEM_UTIL" --no-async-scheduling --seed 42 \
  --no-enforce-eager --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
  --additional-config "$EXTRA" 2>&1 | tee "$R/tp$TP-$MODE-s$MAX_SEQS.log"
```

先跑一条短文本和一条真实多模态请求，确认 `/health`、图 capture、HTTP 结果和 MTP
draft/accepted 计数均正常。Replay 日志应显示选择的 verify 版本、Compact v1 及独立 ring。
TP2 检查两 rank，TP1 检查单 rank。通过后用服务器上的真实压测工具运行多模态持续负载。
同 TP 各路径使用相同媒体、模板、采样、长度预算、warmup、压测节奏、CANN 软件栈和卡组。
先跑 native/V20，再加入 V13 以区分 kernel 更换带来的差异。

## 并发与验收

- **相同负载对照**。在相同的请求到达率和服务配置下对比 TTFT、TPOT、端到端 p99 及吞吐。
  如果两边实际 running 数差异大，再选双方都能稳定达到的相同 running 区间对比 TPOT。
- **饱和吞吐对照**。客户端持续供给请求，服务端用有限的 `max_num_seqs`。
  如果 running 率先达到上限且 KV 仍有余量，提升上限并重启。可从 64 起测，按实测容量
  尝试 96/128。若出现持续抢占、显存紧张或尾时延跃升，使用时延预算内吞吐最高的一档。
  Replay 按 `max_num_seqs` 预分配 ring，9B 每个请求槽的理论 payload 约 TP1 9.09 MiB/卡、
  TP2 4.55 MiB/卡，另有保留槽和其他缓存开销。每一档都记录实际 ring bytes 和 KV blocks。
- **比较单位**。主指标为稳定窗口内实际生成 token/s/卡，辅以成功 req/s/卡及满足既有
  TTFT、TPOT SLA 的 goodput/卡。TP2 的分母是 2。若用两张卡做资源对照，还可运行两个
  并行的 TP1 实例，在相同总流量下合并分子再除以 2。
- **解释收益**。采集 `/metrics` 的 running/waiting、KV 使用和 MTP draft/accepted 增量，
  报告 p50/p95/p99 TTFT、TPOT、请求端到端时延、错误和抢占数。带上输出长度实测分布。
  如果有效并发升高但 token/s/卡未增加，用相同 running 区间的 TPOT、接受率和 prefill
  占比判断瓶颈。若做 profiler，单独重放同一饱和场景，分别记录 vision encoder、prefill
  及 decode。将每条结果的启动参数、commit、扩展 hash、JSON 和 metrics 保存到独立文件。
- **验收**。相同设备和输入下，Replay 在时延预算内的 token/s/卡或 goodput/卡超过 native，
  饱和窗口无持续抢占、错误或吞吐回落。若当前没有正式时延预算，先提交吞吐及 TTFT/TPOT
  尾分位随负载的对照数据，由业务侧选工作点。显存节省和 running 峰值作为解释指标。

decode 主导时可以用 `吞吐比 ≈ 平均有效 decode 并发比 ÷ TPOT 比` 辅助解释。
例如有效 decode 并发增加 20%、TPOT 增加 10%，估算 decode 吞吐增加约 9%。
验收用实际服务吞吐，包含视觉编码、prefill 和排队的影响。

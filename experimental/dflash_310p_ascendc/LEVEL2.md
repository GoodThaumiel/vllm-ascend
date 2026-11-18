# 310P DFlash MRv1 · L2A / L2B 服务验证

硬件为 Ascend310P3，两张空闲卡，CANN 9.2.1，vLLM / vLLM-Ascend 0.30.0。复用通过 LEVEL1 的 `experimental/dflash_310p_ascendc/build/libdflash_310p_torch.so`。本轮 Python-only，已有该库时无需重新编译 AscendC 或 vLLM-Ascend。

使用现有 35B-A3B W8A8 target 及对应版本的 DFlash 权重。确认 3.5 或 3.6 配对、记录两个 checkpoint 的配置和 revision。首轮只测文本、TP2/DP1/PP1、FP16、greedy、K=7、原生 RGDR。关闭 async scheduling、prefix caching、grouped RGDR 和 ReplaySSM。

## 1. 启动前

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
export ASCEND_RT_VISIBLE_DEVICES=4,5  # 换成实际空闲的两张卡
unset VLLM_ASCEND_310P_GROUPED_RGDR_LIBRARY
unset VLLM_ASCEND_REPLAYSSM_310_TRACE_CONFIG
export TARGET=/path/to/35B-A3B-w8a8
export DRAFT=/path/to/matching-DFlash
export RESULTS=/data/l00972401/dflash-level2
mkdir -p "$RESULTS"
git rev-parse HEAD
git status --short
python3 -m unittest \
  experimental.dflash_310p_ascendc.test_reference \
  tests.ut.replayssm.test_dflash_310p_mask \
  tests.ut.replayssm.test_dflash_310p_service_contract
```

验证结果目录内放一份代表性短 prompt `short.txt`，和至少 20 条不同长度的 `prompts.jsonl`。后者每行使用 `{"prompt":"..."}`，混合长短请求，最长约 2k token，用于 C10 补入。保留相同文件用于 eager/graph 对照。

## 2. L2A · eager C1

```bash
SPEC=$(python3 -c 'import json,os; print(json.dumps(dict(method="dflash", model=os.environ["DRAFT"], num_speculative_tokens=7, draft_tensor_parallel_size=2, draft_sample_method="greedy")))')
COMMON=(
  "$TARGET" --host 127.0.0.1 --port 18083 --served-model-name dflash310
  --tensor-parallel-size 2 --dtype float16 --quantization ascend
  --mamba-cache-dtype float16 --mamba-ssm-cache-dtype float16
  --max-model-len 4608 --max-num-batched-tokens 4608 --max-num-seqs 10
  --gpu-memory-utilization 0.85 --no-enable-prefix-caching --no-async-scheduling
  --speculative-config "$SPEC"
)
vllm serve "${COMMON[@]}" --enforce-eager 2>&1 | tee "$RESULTS/eager.log"
```

另一个终端，确认 `/health` 返回 200，两个 rank 均打印 `310P DFlash MRv1 draft KV groups`。应看到全注意力和滑窗层分配到不同物理组。跑短 prompt 和 2k prompt，验证接受 token 计数随请求增加。

```bash
python3 scripts/replayssm/validate_sd_serve.py --model dflash310 \
  --prompt-file "$RESULTS/short.txt" --max-tokens 128 --logprobs 5 \
  --concurrency 1 --repetitions 2 --output "$RESULTS/eager-c1.json"
python3 scripts/replayssm/validate_sd_serve.py --model dflash310 \
  --prompts-jsonl "$RESULTS/prompts.jsonl" --max-tokens 400 \
  --concurrency 10 --repetitions 2 --output "$RESULTS/eager-c10.json"
```

## 3. L2B · 双图 C1 / C10 补入

停止 eager 服务后，在相同设备启动。以下配置覆盖 1–10 个活跃请求。可以按需只配置部分 `cudagraph_capture_sizes`，以 token 数为单位。K=7 时每个可用值是 8 的倍数。只有与实际均匀 decode token 数完全匹配的配置才同时回放 target 和 draft 图，其他批次走 eager。配置稀疏时，实际回放覆盖率取决于运行时批大小。

```bash
vllm serve "${COMMON[@]}" \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[8,16,24,32,40,48,56,64,72,80]}' \
  2>&1 | tee "$RESULTS/graph.log"
```

```bash
python3 scripts/replayssm/validate_sd_serve.py --model dflash310 \
  --prompt-file "$RESULTS/short.txt" --max-tokens 128 --logprobs 5 \
  --concurrency 1 --repetitions 2 --compare "$RESULTS/eager-c1.json" \
  --output "$RESULTS/graph-c1.json"
python3 scripts/replayssm/validate_sd_serve.py --model dflash310 \
  --prompts-jsonl "$RESULTS/prompts.jsonl" --max-tokens 400 \
  --concurrency 10 --repetitions 2 --output "$RESULTS/graph-c10.json"
grep -E '310P DFlash (MRv1 draft KV groups|graph replay|graph ready)' "$RESULTS/graph.log"
```

C1 要求每个 TP rank 都出现 `component=target` 和 `component=draft` 的 8-token **replay**。C10 要求 20/20 完成、出现多请求 descriptor 的双图 replay，并且补入后无 QBM/L0C、507015、越界或非法 metadata。出现分叉时保留同一 prompt 序列的首个不匹配位置和 top-5，作为后续固定历史定位的输入。原始服务压测 JSON 与日志用于本轮验收，性能基线另测。

回传两份服务日志、四份 JSON、首个错误（含 rank、descriptor 和 CANN 日志），以及 `git status --short`。若服务启动或 C1 已失败，可直接回传该项日志。

## 4. C48 首轮挂起定位

先用 C48 配置确认服务启动，`/health` 返回 200。保留 `310P DFlash KV buckets` 日志中的 `block_tokens` 和 `page_bytes`，用来核对组间页大小与原有 token 布局。若两个 rank 停在启动阶段的 KV 分组，直接回传 `determine_available_memory` 的首个堆栈和对应 rank，暂不发送压测请求。该阶段早于图捕获。

保持两个 checkpoint、请求文件、卡对和其他服务参数一致。以下对照只比较是否完成及首个停滞阶段。

1. 复现原配置 `--max-num-seqs 48 --max-num-batched-tokens 8192` 和 C48。启动前设置 `VLLM_ASCEND_310P_DFLASH_STAGE_TRACE=8`，记录最早 8 个调度步的两个 TP rank 日志。若原图配置需要缩小，可先测 `[8,32,64,128,256,384]`，并写清实际配置。
2. 保持 C48 / 8192，切换到 `--enforce-eager`。若两个模式都卡在相同 prefill 阶段，再保持 C48 和 eager，将批 token 上限降到 4096，对比第一批的 `rows`、`tokens` 和 `max_query`。若改图配置或降低 token 上限后通过，保留失败配置作为对照。
3. 若 eager 稳定复现且边界日志只指向整个 forward，用相同 eager 配置加 `ASCEND_LAUNCH_BLOCKING=1` 单跑定位。该设置会改变算子提交方式和时延，定位结果单独记录。
4. 若两 rank 停在 draft prepare 的回读附近，单卡隔离 `prepare_`。使用同一个 LEVEL1 库，分别运行下列命令。每组覆盖 8192 个 context token 及 8191 个有效 token 的尾块场景。完整跑通只排除这些输入的独立复现。

```bash
for rows in 10 48; do
  python3 -m experimental.dflash_310p_ascendc.run --stage eager --device 0 \
    --k 7 --prefill-rows "$rows" --output "$RESULTS/prepare-prefill-${rows}.json"
done
```

提取两 rank 的边界日志，保留完整原始日志和卡占用变化。

```bash
grep -E '310P DFlash (stage|KV buckets|exact capture|graph replay)|507015|HCCL|QuantBatchMatmul|L0C' "$RESULTS/c48.log"
npu-smi info
```

- `target_input_end` 后无 `target_forward_begin` 时，看两 rank 的 metadata / 位置准备与 prefill 路径。
- `target_forward_begin` 的 `mode` 为 `NONE`，且此前读到的是 prefill 时，若没有 `target_forward_enqueued`，优先定位 target prefill 内的算子和 TP 通信。
- `draft_prepare_readback_begin` 或 `draft_prior_stream_wait_begin` 后无对应结束时，记录前面最后完成的边界。这些等待会暴露更早的异步算子失败。
- 只有 trace 先出现 `mode` 为 `FULL`，才把 ACLGraph replay 列为当前卡点。日志里的 `enqueued` 表示主机已返回，设备工作可能仍在执行。

## 5. Worker 初始化时 `Range_int32` 超时

如果日志在 `MinTokensLogitsProcessor` 初始化报 507014，并且 CANN 指出的故障内核是 `Range_int32_double_high_performance__kernel0`，先隔离 Range。这里还在构造 worker 的 model runner，早于 KV 定容、图捕获和请求处理。固定 `K=7`、48 sequences、8192 batched tokens、最大图尺寸 384 时，上游 MRv1 DFlash 初始化有两次 int32 `arange`，长度分别为 `8192`（基类索引）和 `8577`（context + query positions + 1）。

先用 `npu-smi info` 核对物理卡映射和占用。在确认空闲的卡上，每条命令使用独立进程，设置 `CARD` 为对应的可见卡编号。探针只用 PyTorch/torch_npu，无需启动服务或重编译库。

```bash
ASCEND_RT_VISIBLE_DEVICES="$CARD" python3 -m experimental.dflash_310p_ascendc.probe_startup_range \
  --device 0 --length 8192
ASCEND_RT_VISIBLE_DEVICES="$CARD" python3 -m experimental.dflash_310p_ascendc.probe_startup_range \
  --device 0 --length 8577
```

若某个长度失败，在健康空闲卡上用相同长度加 `--source cpu` 做拷贝对照，并在另一张空闲卡上复测同一 NPU 用例。保留两次 `device_sync`、`range_begin/end`、`readback_begin` 边界和首个 CANN 故障日志。独立 Range 与 CPU 拷贝都通过后，用同一启动参数在一对空闲卡上启动服务。若仍报同一内核，只在下一次启动诊断设置 `ASCEND_LAUNCH_BLOCKING=1`，获取实际提交 Range 的 Python 调用栈。服务启动通过后，去掉该环境变量，继续 §4 的 C48 定位。

## 6. C48 KV 计费与容量对照

Python-only，沿用当前已验证的 `.so`。用同一对卡、权重、请求文件、输入/输出长度和显存利用率，比较修复前后的 TP2/K7 服务。保留 C48 的 `max-num-seqs=48`、`max-num-batched-tokens=8192` 和 capture sizes。服务达到 `/health` 200 后先用相同 prompt 跑 C1，再跑 C48，记录峰值 running/waiting、抢占次数和吞吐。

```bash
grep -E '310P DFlash (KV buckets|KV budget|KV allocation|MRv1 draft KV groups)|Available KV cache memory|Maximum concurrency|KV cache usage|[Pp]reempt' \
  "$RESULTS/c48.log"
npu-smi info
```

`KV budget` 给出每个池 block 的旧计费字节数和当前物理字节数，以及同预算下的两种 block 数。`KV allocation` 给出实际 `pool_blocks` 和各组的 `block_tokens`、`planner_page_bytes`、`backing_page_bytes`、`speculative_blocks`。按 gid 与 `MRv1 draft KV groups` 对照即可单独计算 draft 全注意力和滑窗的 padding 差额。记录两个 rank 的这些日志，以及服务启动后的每卡显存。`NPU KV cache usage` 表示调度器共享 block ID 的占用率，可以结合 running 数判断容量。HBM 分配以卡上读数和各组 backing bytes 为准。

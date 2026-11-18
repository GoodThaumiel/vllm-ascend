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

## 7. v24 式共享 backing 对照

本轮只改 Python。保持 TP2、K7、FP16 state、原生 RGDR、相同权重、设备、图尺寸和显存预算。使用同一份 2048-token 输入和 400-token 输出的请求集。先关闭外置 `dflash_kv_fix`，用 `VLLM_ASCEND_310P_DFLASH_SHARED_KV=0` 跑私有 backing 对照，再设 `=1` 跑共享版。该变量在启动 `vllm serve` 前设置，默认值为 `1`。

1. 共享版启动后核对两个 rank 都打印 `310P DFlash shared KV budget` 和 `310P DFlash shared KV allocation`，且 draft 仍有原来的滑窗和全注意力组。日志列出物理 bytes/block、实际 `pool_blocks`、各组逻辑 block 大小，以及 2048+400+7 token 的并发估计。如果出现 `310P DFlash KV allocation` 而无 `shared KV allocation`，回传分组和配置，表示该布局未选中共享路径。
2. 先跑 eager C1，再跑 graph C1/C10。用同一 prompt 对照私有 backing，记录首分叉和 logprobs。C10 使用长短混合及持续补入，覆盖 block ID 在不同请求和 group 间回收，检查双 rank target/draft replay、无非法 slot、无 507014/507015。若换布局后出现稳定分叉，保留相同输入、两种布局的首个分叉及对应 group 日志。
3. 跑 C48 的 2k/400，记录峰值和持续 running、waiting、preemption、吞吐，以及每 rank 服务稳定后的 HBM。`estimated_concurrency` 按完整 2k/400 序列和一个保留 ID 计算，调度分块及请求长度变化会影响实测值。

以 30 个 GDN 层、10 个 target attention 层、5 个 draft SWA 层和 1 个 draft full 层为例，假设 target 每 rank 一个 256 维 KV head、draft 每 rank 四个 128 维 KV head，max-model-len 4096。共享版使用 6 个 GDN backing 槽、5 对 K/V 槽。Target/draft 调度页分别为 768/384 token，底层仍按 64/128 token 的 kernel 页访问。即使每个请求需要的全局 ID 从约 36 增到 62，ID 的物理成本也从约 25.2 MiB 降到 7.0 MiB。如果旧版持续 7 条请求对应 253–288 个总 block，且两次 `physical_budget_bytes` 相同，新版预计持续 14–16 条。实际模型以日志中的池大小和每请求 ID 数为准。

## 8. v24 自动预算与显式物理预算

本轮是 Python-only。用 §7 已验证的 TP2、K7、2k/400、共享 KV、无 prefix cache 配置，分别以以下方式启动，每轮先结束旧实例。外置 v24 插件关闭。三轮保持相同的权重、卡对、`max-num-batched-tokens`、capture sizes 和 `gpu-memory-utilization`。

| 对照 | 启动方式 | 池大小 |
|---|---|---|
| 0.30 默认 | `VLLM_ASCEND_310P_DFLASH_V24_KV_BUDGET=0` | `profile 后剩余 / 2 / 物理 bytes per block` |
| v24 预算算法 | `VLLM_ASCEND_310P_DFLASH_V24_KV_BUDGET=1` | `profile 后剩余 / 2 / planner 逻辑 bytes per block` |
| 显式物理预算 | 加 `--kv-cache-memory-bytes <每卡物理字节>` | `显式预算 / 物理 bytes per block` |

显式值优先于环境变量。该值按每 rank 计，profile 仍执行，输入不得超过 profile 后实测空闲内存。先复用默认轮的 `planned_backing_bytes` 校验显式路径的 block 数，再结合实际 workspace/graph 余量选需要对比的预算。v24 模式沿用当前 0.30 分组，按 v24 的逻辑计费方式定容。不同分组会产生不同的倍数。

每轮先检查 `/health` 和 C1 双图，再跑 C10 补入、C48 2k/400。回传两 rank 的 `310P DFlash KV memory after profiling`、`shared KV budget`、`shared KV allocation`、图内存日志、启动后每卡显存、实际持续 running 和抢占次数。缓存分配早于图捕获。若扩大预算后捕获或补入 OOM，记录首个失败阶段，改用较低的显式物理预算验证容量曲线。测 52 的上限时，需把 `max-num-seqs` 和施压并发都设在 52 以上。

## 9. K=1–7 与 8 核 grouped RGDR

本级使用 vLLM/vLLM-Ascend 0.30.0、310P3 TP2、CANN 9.2.1、torch_npu 2.10.0.post7 和同一对 35B W8A8 target/drafter 权重。每次启动固定一个 K。保持 greedy、`temperature=0`、FP16 state、`--mamba-cache-mode none`，并关闭 prefix cache、async scheduling、ReplaySSM 和外置 0.23 monkey patch。DFlash prepare 沿用 LEVEL1 `.so`。Grouped RGDR 使用 `ascend310p-rgdr-patch` 提交 `c98759c` 的计算内核和 Torch binding，服务分派到 `rank1-gather16-interleave`，8 核。

### 9.1 构建与单算子回归

在仓库根目录运行。只构建 RGDR Torch 扩展，沿用当前安装的 vLLM-Ascend。

```bash
export RGDR_BUILD="$PWD/experimental/rgdr_v310_ablation/build-c921"
cmake -S experimental/rgdr_v310_ablation -B "$RGDR_BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend310P3 \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.1 \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build "$RGDR_BUILD" --target rgdr_v310_cube_fp16_torch -j8
export RGDR_SO="$RGDR_BUILD/librgdr_v310_cube_fp16_torch.so"
readelf -d "$RGDR_SO" | grep -E 'RUNPATH|RPATH'
```

确认 RUNPATH 包含 `$ORIGIN/lib`。用空闲单卡分别测 K=2/3/5/7，覆盖奇数 query 宽度与目标 K=5 容量场景。Grouped RGDR 对照都用相同的输入种子，并覆盖 ragged 窗口、非零 storage offset、保留槽和图重放。

```bash
for k in 2 3 5 7; do
  python3 -m experimental.dflash_310p_ascendc.probe_attention --stage graph --k "$k" --device 0 \
    --output "$RESULTS/attention-k${k}.json" || break
  common=(--mtp "$k" --batch 32 --cores 8 --ragged --device 0 --library "$RGDR_SO")
  python3 -m experimental.rgdr_v310_ablation.bench "${common[@]}" --variant cube-grouped \
    --write-golden "$RESULTS/grouped-k${k}.pt" --output "$RESULTS/grouped-k${k}.json" || break
  python3 -m experimental.rgdr_v310_ablation.bench "${common[@]}" --variant rank1-gather16-interleave \
    --golden "$RESULTS/grouped-k${k}.pt" --output "$RESULTS/interleave-k${k}.json" || break
done
```

### 9.2 `vllm serve` 双图与 2k/400

用 §2 的 `COMMON`，从中移除原有 `--speculative-config "$SPEC"`，并设置 `--max-model-len 4096 --max-num-batched-tokens 8192 --max-num-seqs 48 --mamba-cache-mode none --language-model-only`。每次服务启动前设置 K，下面以 K=5 为例。命令使用 §8 的 v24 自动预算复现上一轮约 34 条 running 的配置。若上一轮有显式预算，保留原值并在两条路径使用同一个预算。

```bash
export K=5
export VLLM_ASCEND_310P_DFLASH_V24_KV_BUDGET=1
SPEC=$(python3 -c 'import json,os; print(json.dumps(dict(method="dflash", model=os.environ["DRAFT"], num_speculative_tokens=int(os.environ["K"]), draft_tensor_parallel_size=2, draft_sample_method="greedy")))')
COMP=$(python3 -c 'import json,os; w=int(os.environ["K"])+1; print(json.dumps(dict(cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[w*n for n in (1,4,8,10,16,24,28,32,34,36,40,48)])))')
unset VLLM_ASCEND_310P_GROUPED_RGDR_LIBRARY
vllm serve "${COMMON[@]}" --speculative-config "$SPEC" --compilation-config "$COMP" \
  2>&1 | tee "$RESULTS/native-k${K}.log"
# 测完并停止原生服务后，再启动 grouped 路径。
VLLM_ASCEND_310P_GROUPED_RGDR_LIBRARY="$RGDR_SO" \
  vllm serve "${COMMON[@]}" --speculative-config "$SPEC" --compilation-config "$COMP" \
  2>&1 | tee "$RESULTS/grouped-k${K}.log"
```

使用 §2/§3 的 HTTP 验证工具。先以原生 RGDR 测 K=2、3、5、7 的 C1 和 C10 持续补入，再对相同 K 启用 grouped RGDR 重跑。K=1、4、6 至少跑双路径 C1，补齐宽度覆盖。最大 capture token 数应为 `max-num-seqs × (K+1)` 或更小。若当前负载的实际 running 批大小未捕获，实测会走 eager。做吞吐对照时两条路径使用相同的 capture sizes，并从日志核对命中率。


C48 使用上一轮的约 2048-token prompt 文件。在另一终端分别为两种服务设置 `ROUTE=native` 或 `ROUTE=grouped`，输出写到不同文件。

```bash
export K=5 ROUTE=native
python3 scripts/replayssm/validate_sd_serve.py --model dflash310 \
  --prompt-file "$RESULTS/2k.txt" --max-tokens 400 --concurrency 48 \
  --timeout 1200 --output "$RESULTS/${ROUTE}-k${K}-c48.json"
```

回传每 rank 的 `310P grouped FP16 RGDR enabled`、target/draft graph replay、首分叉与 logprobs、acceptance、实际 running、TPOT 和总吞吐。Grouped 版的 target verify 在 msprof 中应出现 `rgdr_v310_cube_probe_rank1_gather16_interleave` 或 `_h16`，blockDim=8。单 token 非投机分支仍可能出现原生 RGDR。优先测 K=5、7 的 C48 2k/400，核对 K=7 的约 34 条运行基线和 K=5 的 `estimated_concurrency`。K=2/4/6 若出现图错误，保留同 K eager 对照和首个 SplitFuse 或 GDN 错误。

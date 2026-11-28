# Qwen3.5-27B TP4 ReplaySSM SD 服务上机清单

在 `releases/v0.30.0` 使用源码编译的 vLLM Ascend、vLLM
`0.30.0+empty`、Qwen3.5-27B BF16、TP4、PP1 和 MTP3。为服务
分配四个空闲逻辑设备。用 `npu-smi info -m` 核对物理卡与逻辑
设备的映射。910C 单 die 验证使用 `ascend910_9362`、CANN 9.2.1
和 torch-npu 2.10.0.post7。此前 910B4 环境为 CANN 9.2.0 和
torch-npu 2.10.0.post4。记录本机实际版本，并为该 SoC 重新编译。

开启 `replayssm_sd` 后，TP4/MTP3 且全模型 GDN 有 16 个 key head、
48 个 value head、K/V head dim 均为 128 时，默认选紧凑 mixed
verify+compact。AIC 和 AIV 按 key head 整组轮转，折叠阈值为
L8，物理 ring 为 32，checkpoint BF16，D/K FP16，G FP32。
预期首轮无历史，之后最多折叠上轮接收的 4 条历史。

已测的是 910C 单 rank 的 27B 算子几何，固定 T4、历史 4、
全 batch flush 时，B32/B64 单层 graph p50 为 198/348 µs，
native 为 201/377 µs。四设备 TP4 服务的正确性、吞吐和时延
在本清单中验收。9B 的 L8/L20 长文本实验已观测到 token 分叉，
请保留本轮输出与 MTP 接受率供定位。[单 die 原始结果和条件](LEVEL3_SD_910C_KERNEL_SCALING.md#缩短待折叠历史)

## 1. 构建和安装

在源码 checkout 根目录执行。将变量设置成服务器上的实际路径和
SoC。命令构建主扩展与 mixed 动态库，保留其相对 RUNPATH。

```bash
set -euo pipefail
export CANN_ROOT=/usr/local/Ascend/cann-9.2.1
export SOC_VERSION=ascend910_9362
# 910B4 示例：CANN_ROOT=/usr/local/Ascend/cann-9.2.0；SOC_VERSION=Ascend910B4
source "$CANN_ROOT/set_env.sh"
python3 -m pip install --no-build-isolation --no-deps -e .

export MIXED_BUILD="$PWD/experimental/replayssm_cube_910b/mixed_verify/build-$SOC_VERSION"
cmake -S experimental/replayssm_cube_910b/mixed_verify -B "$MIXED_BUILD" \
  -DSOC_VERSION="$SOC_VERSION" -DASCEND_CANN_PACKAGE_PATH="$CANN_ROOT" \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build "$MIXED_BUILD" -j8
PKG_DIR=$(python3 -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("vllm_ascend").origin).parent)')
export MIXED_INSTALL="$PKG_DIR/_replayssm_mixed"
cmake --install "$MIXED_BUILD" --prefix "$MIXED_INSTALL"

git rev-parse HEAD
python3 -m pip show vllm vllm-ascend torch-npu
sha256sum "$MIXED_INSTALL/libreplayssm_sd_mix_torch.so" \
  "$MIXED_INSTALL/lib/libreplayssm_sd_mix_kernel.so"
python3 - "$MIXED_INSTALL/libreplayssm_sd_mix_torch.so" <<'PY'
import sys
import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op

assert enable_custom_op()
assert hasattr(torch.ops._C_ascend, "replayssm_sd_cursor_")
torch.ops.load_library(sys.argv[1])
assert hasattr(torch.ops.replayssm_mix_910b, "verify_")
print("ReplaySSM SD cursor and mixed verify loaded")
PY
```

若选择自定义安装位置，在 Replay 组的 `additional-config` 中传
`replayssm_sd_mixed_library` 绝对路径。保持相邻 `lib/` 里的
kernel 依赖。默认路径缺库会在启动时报错，便于确认本轮测试路径。

发布前已在 Blue2 的 CANN 9.2.1 环境构建、安装并加载该 mixed
库，确认 RUNPATH 为 `$ORIGIN/lib`。安装后的 kernel SHA-256
为 `a354452304e6eed7c6402f62aca9168f1b8f2b5ae7df000d120808ede8c91298`，
与此前 910C 连续窗口和服务对照使用的 kernel 一致。

## 2. 启动 `vllm serve`

对照组用同一源码、同一权重、同一组设备顺序重启。先运行
`MODE=native`，再运行 `MODE=mixed-l8`。如需隔离阈值变化，
加测 `MODE=mixed-l20`。如需复查旧 vector 路径，选 `vector-v20`。

```bash
set -euo pipefail
: "${REPLAYSSM_MODEL:?set local path to Qwen3.5-27B}"
: "${ASCEND_RT_VISIBLE_DEVICES:?set four reserved logical NPU devices}"
export ASCEND_RT_VISIBLE_DEVICES
MODE=${MODE:-mixed-l8}
PORT=${PORT:-18083}
R=${R:-/tmp/replayssm-27b-tp4}
MAX_SEQS=${MAX_SEQS:-64}
MAX_LEN=${MAX_LEN:-34816}
MNB=${MNB:-8192}
MEM_UTIL=${MEM_UTIL:-0.85}
mkdir -p "$R/$MODE"
case "$MODE" in
  native) EXTRA='{"replayssm_sd":false}' ;;
  mixed-l8) EXTRA='{"replayssm_sd":true}' ;;
  mixed-l20) EXTRA='{"replayssm_sd":true,"replayssm_sd_logical_limit":20}' ;;
  vector-v20) EXTRA='{"replayssm_sd":true,"replayssm_sd_verify_variant":20}' ;;
  *) echo "Unknown MODE: $MODE" >&2; exit 2 ;;
esac
export VLLM_CACHE_ROOT="$R/$MODE/cache"
export VLLM_ASCEND_REPLAYSSM_MEMORY_PROBE=1

vllm serve "$REPLAYSSM_MODEL" \
  --host 127.0.0.1 --port "$PORT" --served-model-name qwen35-27b \
  --tensor-parallel-size 4 --pipeline-parallel-size 1 \
  --dtype bfloat16 --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype bfloat16 \
  --mamba-cache-mode none --no-enable-prefix-caching --enable-chunked-prefill \
  --no-async-scheduling --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" \
  --max-num-batched-tokens "$MNB" --gpu-memory-utilization "$MEM_UTIL" --seed 42 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,16,32,64,128,256],"max_cudagraph_capture_size":256}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}' \
  --additional-config "$EXTRA" 2>&1 | tee "$R/$MODE/serve.log"
```

在 mixed-l8 日志确认 `logical limit 8`、`grouped mixed verify and
fused compact`、`independent request ring` 和 FULL 图 capture。
记录每个 rank 的 KV cache 容量、ring 容量与分配字节数。
Native 使用相同的 MTP3 和 BF16 cache。默认的 MTP3 proposer
为 eager，target decode 使用 FULL 图。

## 3. 冒烟和负载对照

另开终端，从同一个源码目录执行。准备一条短文本和固定长文本
JSONL，每行是 `{"prompt":"..."}`，按模型 tokenizer 将长文本
控制在约 32k token。两组使用同一文件。

```bash
set -euo pipefail
MODE=${MODE:-mixed-l8}
R=${R:-/tmp/replayssm-27b-tp4}
URL="http://127.0.0.1:${PORT:-18083}"
: "${REPLAYSSM_SHORT_PROMPT:?set a short prompt text file}"
: "${REPLAYSSM_LONG_PROMPTS:?set the 32k prompts JSONL file}"
curl -fsS "$URL/health"
python3 scripts/replayssm/validate_sd_serve.py --url "$URL" --model qwen35-27b \
  --prompt-file "$REPLAYSSM_SHORT_PROMPT" --concurrency 1 --repetitions 2 \
  --max-tokens 128 --logprobs 5 --output "$R/$MODE/smoke.json"
for c in 1 32 64; do
  python3 scripts/replayssm/validate_sd_serve.py --url "$URL" --model qwen35-27b \
    --prompts-jsonl "$REPLAYSSM_LONG_PROMPTS" --concurrency "$c" --repetitions 1 \
    --max-tokens 512 --timeout 1800 --output "$R/$MODE/long-c$c.json"
done
```

逐组核对 HTTP 成功数、实际输入输出 token 数、吞吐、请求 p50/p90、
running/waiting、抢占和 MTP 接受率。记录 decode 纯图 forward 的
p50/p90 时，按实际活跃行、实际候选 token 数和捕获图 shape 对齐。
原生与 Replay 的容量有差异时，分别汇报同活跃批的时延以及同客户端
负载的吞吐。压测并发可以补 B16/B18、B30/B32/B34 和 B62/B64，
并在图中区分无 flush、混合 flush 和全 flush 的 step。

若输出分叉，用短文本的 top logprobs 比较第一处分叉，保留两侧
token 和 top-1/top-2 间距。回传启动参数、commit、库 hash、
服务日志、JSON、metrics 及可用的 target forward trace。

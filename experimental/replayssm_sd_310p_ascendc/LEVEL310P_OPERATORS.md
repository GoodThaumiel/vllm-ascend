# L310P.1 ReplaySSM SD 独立算子上机清单

目标：Ascend310P3 × 2（每次先在其中一张卡做隔离测试），CANN 9.2.0、
torch_npu 2.10.0.post4、vLLM 0.30.0+empty、vllm-ascend v0.30.0 源码。
本级仅验证三个算子；不启用 `vllm serve` 的 ReplaySSM 分派。

## 1. 环境、构建、加载

在仓库根目录执行，先确认所见设备确为 `Ascend310P3`：

```bash
git rev-parse HEAD
git status --short
npu-smi info
python3 -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.get_device_name(0))'
python3 -m unittest experimental.replayssm_sd_310p_ascendc.tests.test_reference
cmake -S experimental/replayssm_sd_310p_ascendc -B experimental/replayssm_sd_310p_ascendc/build \
  -DSOC_VERSION=Ascend310P3 \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.0 \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build experimental/replayssm_sd_310p_ascendc/build -j 8
readelf -d experimental/replayssm_sd_310p_ascendc/build/libreplayssm_sd_310p_torch.so | grep -E 'RPATH|RUNPATH'
python3 -c 'import torch, torch_npu; torch.ops.load_library("experimental/replayssm_sd_310p_ascendc/build/libreplayssm_sd_310p_torch.so"); print(torch.ops.replayssm_sd_310p.verify_._schemas, torch.ops.replayssm_sd_310p.compact_._schemas, torch.ops.replayssm_sd_310p.cursor_._schemas)'
```

如果编译失败，立即停止，不运行旧 `.so`。回传第一条 CCE 错误、完整编译日志、
`CMakeCache.txt` 中的 SoC/CANN/Python 配置，以及构建目录下的文件清单。
优先区分编译器 API 不支持、UB 预算、link/RPATH、运行时设备错误。

## 2. 精度与状态生命周期

先运行小 batch，确认 eager 与图模式均通过。`run.py` 的输出 JSON 即回传件：

```bash
mkdir -p /tmp/replayssm-310p-l1
for mtp in 3 5; do
  python3 experimental/replayssm_sd_310p_ascendc/run.py \
    --mtp "$mtp" --batch 2 --stage smoke --device 0 \
    --output "/tmp/replayssm-310p-l1/mtp${mtp}-smoke.json"
  python3 experimental/replayssm_sd_310p_ascendc/run.py \
    --mtp "$mtp" --batch 1 --stage sequence --device 0 \
    --output "/tmp/replayssm-310p-l1/mtp${mtp}-sequence.json"
  python3 experimental/replayssm_sd_310p_ascendc/run.py \
    --mtp "$mtp" --batch 2 --stage graph --device 0 \
    --output "/tmp/replayssm-310p-l1/mtp${mtp}-graph.json"
done
```

`smoke` 覆盖 history 0/7/12/16、ring 跨尾、checkpoint 与请求 ring 槽号分离、混合 flush、padding、
verify→compact→cursor；`sequence` 覆盖 24 轮接受、折叠和 ring 周转；
`graph` 在捕获后更改设备元数据并重放。逐元素 int32 状态码必须精确一致；
FP16 输出和 checkpoint 使用 atol=rtol=2^-9、匹配率至少 99%、
最大绝对误差不超过 0.1 的混合容差，并记录实际最大误差。

## 3. 与 310P 原生算子对比

只有第 2 节全过后才运行性能矩阵。脚本直接调用
`torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310`，与 Replay 使用相同
Q/K/V/g/beta；native 的初始 state 先从同一份 checkpoint 和 history 重建。
计时区不含输入准备和状态重置，预热 20 次、图重放 100 次：

```bash
for mtp in 3 5; do
  for batch in 1 16 32 64; do
    python3 experimental/replayssm_sd_310p_ascendc/run.py \
      --mtp "$mtp" --batch "$batch" --stage perf --tile-v 32 \
      --cores 7 --iterations 100 --warmup 20 --device 0 \
      --output "/tmp/replayssm-310p-l1/mtp${mtp}-b${batch}-perf.json"
  done
done
```

每组记录 verify、compact、cursor、三个算子组合、native 的 graph replay
p50/p90/min，场景为 pos0/12/16 与 flush0/1。单算子时延和三个算子总时延
都要报告；此级只建立基线，不以快于 native 为通过条件。`tile_v=64` 仅在
tile32 构建、精度和图测试均通过后补测，compact 始终使用 tile32。

## 4. 回传

请回传：全部 JSON、首个失败 case 的完整 stderr/CCE 日志、
`git rev-parse HEAD`、`git status --short`、设备型号与包版本、
`readelf` RUNPATH 输出。若能同时抓 msprof，附上每个 kernel 的 task duration、
AIV vector/scalar/MTE2/MTE3；这部分用于下一轮优化，不影响本级验收。

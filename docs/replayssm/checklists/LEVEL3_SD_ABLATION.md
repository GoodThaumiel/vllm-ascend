# L3 SD verify Ascend910B4 消融上机清单

目标环境：CANN 9.2.0、torch_npu 2.10.0.post4、vLLM 0.30.0+empty、此分支源码编译的 vLLM Ascend、Ascend910B4。先对独立非 flush verify 做实验；不切换 `vllm serve` 的 MTP dispatch。

| variant | 相对前一版唯一新增项 | 主要判断 |
| --- | --- | --- |
| 0 | 已验证基线 | 同容器复现参考时延 |
| 1 | K=128 的 `Add + WholeReduceSum` | 归约时延、精度是否改善 |
| 2 | 连续批量搬运 ring gate 与候选 gate/beta | Scalar/MTE2 等待是否下降 |
| 3 | 每 value head 一任务，全部 V tile 共享点积表 | 高 batch 的重复归约是否下降；低 batch 核数可能不足 |
| 4 | variant 3 加两槽 checkpoint 预取 | MTE2 与 Vector 是否有有效重叠 |

`tile_v=32/64` 是与 variant 正交的第二个维度。按源码显式 `InitBuffer` 计算，五版在 tile32 分别申请 60,832 / 56,736 / 57,568 / 59,616 / 76,000 字节，在 tile64 分别申请 105,248 / 101,152 / 101,984 / 104,032 / 136,800 字节；这些不是编译后的 UB 占用。用户提供的 Vector UB 容量是 192 KiB。请保存 CANN 编译日志中每个核的实际 UB 用量、blockDim、失败核名。全部是 40 个 Vector 核可用的 AIV-only 算子；variant 3/4 的有效任务数为 `B×HV`，其余为 `B×HV×(128/tile_v)`。

## 1. 构建与正确性，按顺序执行

先在同一镜像里重编译主扩展。独立 lab 也能用于诊断；其 `build/lib` RPATH 已沿用验证过的配置。

```bash
python3 -m pip install --no-build-isolation --no-deps -v -e .
cmake -S experimental/replayssm_sd_ascendc -B experimental/replayssm_sd_ascendc/build \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.0 \
  -DSOC_VERSION=Ascend910B4 -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build experimental/replayssm_sd_ascendc/build --parallel 4
```

在一张空闲卡上先固定 TP4/B2/tile32，逐版做 `smoke` 和 `graph`，任何一版失败即停在该版并回传首个 JSON、编译日志和设备日志；确认卡健康后再继续。`graph` 覆盖位置 0/7/12/16、环绕、padding、变长 T=1..4。

```bash
for variant in 0 1 2 3 4; do
  for stage in smoke graph; do
    python3 experimental/replayssm_sd_ascendc/run.py --backend framework \
      --stage "$stage" --tp 4 --batch 2 --tile-v 32 --cores 40 \
      --variant "$variant" --device 0 \
      --output "/tmp/sd-ablate-v${variant}-${stage}.json" || exit 1
  done
done
```

以上全部 `PASSED` 后，tile64 对五版重复 `smoke`/`graph`，并对 TP8 重复。检查 `out/d_cache/k_cache/g_cache` 对 CPU dense golden、所有 status=0；图 capture 与 replay 均无异常。新归约可能改变 FP32 加法顺序，需记录最大绝对误差、相对 L2 和最差位置，不只看 PASS。

## 2. 小范围性能筛选

固定 TP4、tile32、40 cores、B=1/32/64，逐版运行图回放计时。每个 `perf` 报告包含 pos0/T1、pos7/T4、pos12/T4、pos16/T4；reset 在计时区外。先用这 15 组挑出候选，再对候选与 v0 做 tile64、TP8 扩展，避免无效大矩阵。

```bash
for batch in 1 32 64; do
  for variant in 0 1 2 3 4; do
    python3 experimental/replayssm_sd_ascendc/run.py --backend framework \
      --stage perf --tp 4 --batch "$batch" --tile-v 32 --cores 40 \
      --variant "$variant" --iterations 50 --device 0 \
      --output "/tmp/sd-ablate-tp4-b${batch}-v${variant}-tile32.json" || exit 1
  done
done
```

对比同一 `(TP,B,T,history,tile_v)` 的 p50/p90，重点是 B32/64、T4、history12/16。v3/v4 的 B1 有效核数少，允许它们输给 v1/v2；最终可按 batch 选路。任何精度失败、507014、board error 或残留进程，先停止后续测试。

## 3. 对原生路径抓 profiler

筛出最优 1–2 个变体后，使用原生同一 Python 入口 `vllm_ascend.ops.gdn.recurrent_gated_delta_rule` 做 100 次对照。下面以 v3 为例，分别跑 v0 和候选；`context-lengths` 只是标签，非本算子的维度。脚本报告的 host 总时间不能代替设备 kernel 时间；在 `REPLAYSSM_SD/.../native` 和 `/replay` 区间抓 profiler，再算设备 p50/p90 与 roofline。

```bash
python3 scripts/replayssm/profile_sd_operator_compare.py \
  --tp 4 --device 0 --batches 1,32,64 --context-lengths 32768 \
  --lengths 4 --positions 12 --ring 32 --tile-v 32 --cores 40 \
  --variant 3 --repetitions 100 --warmup 2 \
  --output /tmp/sd-ablate-native-v3.json
```

请一并回传：每版首个失败 case 或全 PASS JSON、算子设备时延、实际 blockDim/UB 用量、Vector/Scalar/MTE2/MTE3 占比与等待、HBM/L2 读写、可用的 UB bank conflict 指标、实际 `fla_npu` 后端（stable/ctypes）及原生二进制版本。bank conflict 和流水重叠目前是待检验假设。验收目标是 B32/64 的 SD verify 尽量接近 native；若仍显著慢，先看归约数和等待变化，再决定是否做跨 GQA head 的点积共享或 Cube 小矩阵路径。compact/cursor 与完整服务时延留在 verify 选型后一起评估。

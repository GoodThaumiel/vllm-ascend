# Ascend310P3 ReplaySSM SD 算子重设计

目标是降低 TP2、W8A8、MTP3/7 下的 Replay verify 和 compact 开销，
重点测 B32、B64 和 B128。沿用 CANN 9.2.1、FP16 checkpoint/D/K 和 FP32 G ring，
以及 16 条逻辑历史、32 个物理位置的独立请求 ring。

## RGDR 实测给出的依据

在 B32、相同输入的 msprof 对照中，`cube-grouped` 的 MTP3/7 p50
为 1.323/2.534 ms，原生 V310 为 1.957/3.859 ms，旧 Cube DEQ
为 5.783/11.402 ms。Grouped 的 Scalar 为 0.468/0.926 ms，
约为旧 Cube 的十分之一。这轮同时改变了 task 划分、state tile 和搬运，
各项收益需要独立消融才能归因。

可复用已验证的双 value head 分组、整窗载入、FP16 RHS 的 NZ→ZN
`LoadData2D`、L0B 复用、L0C→UB 单次搬运和连续写回。
现有 Replay verify 已经给一个 value head 复用 Q/K 于多个 V tile。
310P 的首要改动是跨两个共享 Q/K 的 value heads 复用，并将当前窗口的
所有 checkpoint 投影合成一次矩阵乘，减少 Scalar 发指令和逐 token 的等待。

## Verify 首版

每个 task 对应 `(request, key head)`，负责两个 value heads。
整窗读入 Q/K/V/g/beta，只读取一次行元数据和共享 K ring。
保持原有独立 checkpoint slot 和 ring slot，按 base 处理 ring 跨尾。
输入长度可以短于物理窗口，padding 行保持零输出和既有 status 行为。

把一轮实际候选的 K/Q 组织为列矩阵 `B=[K_window^T | Q_window^T]`，
补齐到 16 列。MTP3 有最多 8 列有效，MTP7 有最多 16 列有效。
两个 value heads 顺序复用同一份 L0B。每个 head 采用完整 V=128，
用 `LoadData2D → Mmad → L0C → UB` 算 `H_checkpoint B`。
FP16 输入、FP32 L0C 累加，结果以 FP32 搬出。

用 `α_i=exp(g_i)` 和 FP32 后缀积，把历史贡献加到投影上：

```text
w_i = product(α_j, j > i)
P = product(α_i) * (H_checkpoint B) + sum_i w_i d_i (k_i^T B)

for t in actual_window:
    P = exp(g_t) * P
    d_t = beta_t * (v_t - P[:, K_column(t)])
    P += d_t (k_t^T B)
    y_t = P[:, Q_column(t)] / sqrt(128)
```

候选之间的递推仍按 token 顺序。共享 key head 的 `k_i^T B` 和
`k_t^T B` 对两个 value heads 只算一次。首版保留 FP32 历史修正和
小矩阵更新，可先用批量 Vector 实现；把这些 dot 或历史修正另行交给
Cube 时，再评估 FP16 中间量的舍入。状态写回仅包括输出、候选 D/K/G
ring 和 status，verify 不写 checkpoint，也不生成每个候选的全量 state。
当前 token 的输出取出后，后续更新可跳过已消费的 K/Q 列。首次上机
同时记录短窗与满窗，因为补齐 16 列的固定成本在短窗中占比更大。

按 310P3 硬件矩阵估算，一个 head 的 FP16 checkpoint L0A 为
32 KiB，FP16 RHS L0B 为 4 KiB，FP32 L0C 结果为 8 KiB。
这些是矩阵载荷，实际 UB 还需计入队列、历史 K/D、两头共享缓冲、
bank 对齐和编译器分配。两头依次使用同一份 L0A/L0C workspace。
V=128 为首选，若 CCE 资源或上机时延不合适，用 V=64 做同口径对照。

### 与逐 token Cube 更新的比较

另一条可复用路径是先从 checkpoint+ring 恢复完整 state 到 UB，
再复用 RGDR grouped 的逐 token 单圈 Cube。它适合做易读的对照实现，
但每个候选都要把更新后的 state 重新打包送入 L0A，且需要处理 FP32
工作 state 到 FP16 Cube 输入的舍入。首版选择整窗投影，使每 head
只搬入一次 checkpoint，避免这条逐 token 的大矩阵往返。

## Compact 与 cursor

Compact 第一版仍独立 launch，沿用 `(request, key head)`，
先读一次 metadata。对于 `flush=0` 或 `pos=0` 的行只处理 status。
重路径按 ring 跨尾拆成至多两段，批量载入 K/D/G，再算 FP32 后缀积。
代数上等价的折叠为：

```text
H_commit = product(α_i) * H_checkpoint + sum_i w_i d_i k_i^T
```

先做 FP32 批量 Vector 折叠。随后测试以历史为短 K 轴的 Cube rank 更新。
FP16 D/K 可直接作为 Cube 输入，但不能把 FP32 后缀权重静默折进 FP16 D
或 K。先用 FP32 权重路径作为精度参照。Cube 候选如需在上 Cube 前
将带权 D 转成 FP16，单独记录并比较 checkpoint 和下一轮重建误差。
验证端应保留 FP32 Vector 折叠作为基线。按状态块整块写 status，
去掉当前每 V tile 的 32B status 读改写。

Cursor 保留当前语义和调用点。它按上一轮接受数决定本轮 pos/base/flush。
verify 写入本轮候选，compact 只折叠本轮开始时的已提交历史，
候选仍留在原 ring 位置。等独立 verify 和 compact 在 MTP3/7、
混合 flush、跨尾和多轮接续均通过后，再评估在同一 task 中复用
checkpoint 与 history 的 verify→compact 融合。

## 算子验收

本地用 60 个 CPU 用例比对了整窗投影公式和现有 recurrence reference。
覆盖 MTP3/7、短窗与满窗、5 个历史长度、3 个 ring base 和两种 flush。
FP64 投影转 FP16 后的输出和 D ring 逐元素一致。FP64 后缀积折叠与
现有 FP32 compact reference 的 checkpoint 最大绝对差为
`3.05e-5`。310P3 的 Cube 累加、UB 布局和服务级精度仍需单独验证。

1. 先以 CPU FP64 recurrence 对照整窗投影和后缀积公式，再验证设备路径。
   检查 `out`、D/K/G ring、status、checkpoint、下一轮重建 state。
   覆盖 pos 0/1/7/12/16、跨尾、短窗、padding、混合 flush、
   checkpoint/ring 槽号不同、拒绝、全接受及 slot 重用。
2. 用相同输入分别抓 V12 verify+compact+cursor、新 verify 配旧 compact、
   新 verify+新 compact，并保留原生 V310 和 RGDR `cube-grouped`。
   B32/B64/B128 × MTP3/7，先测固定长度和 flush，再测服务采样的分布。
   统计单次 task p50/p90、实际 rows/length/pos、flush 行数与
   `sum(flush*pos)`。分别看 Scalar、Vector、MTE、Cube 和可用 L2 指标。
3. 新版本需要在目标高并发和两档 MTP 下改善三算子合计，且通过多轮正确性。
   `cube-grouped` 是服务性能的额外参照。算子收益通过后，再做同预算
   和同有效并发的服务复测。

## 整窗 Cube verify 首版上机清单

首版实现独立算子 `replayssm_sd_310p_verify_cube`。使用
`--status-variant 2 --tile-v 64` 选择，64 指兼容 V12 的 status ABI。
实际 Cube V tile 为 128。每个 core 串行处理同一 status 块里的两对
value heads，每对复用一次 L0B。K 固定在 0–7 列，Q 固定在 8–15 列。
首版用批量 FP32 Vector 计算共享 dot，并按历史顺序在投影上更新，
历史衰减的 Scalar 系数暂留在 UB 读取。状态仍由 V12 compact 和 cursor
维护。本轮先测 verify 替换收益，服务配置继续使用 V12 verify。

在仓库根目录使用 CANN 9.2.1，配置一个新的 Release build 目录。

```bash
set -e
op=experimental/replayssm_sd_310p_ascendc
build="$op/build-cube-verify"
results=/tmp/replayssm-310p-cube-verify
mkdir -p "$results"
cmake -S "$op" -B "$build" -DCMAKE_BUILD_TYPE=Release \
  -DSOC_VERSION=Ascend310P3 -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.1 \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build "$build" -j 8
lib="$build/libreplayssm_sd_310p_torch.so"
readelf -d "$lib" | grep -E 'RUNPATH|RPATH'
python3 -m unittest experimental.replayssm_sd_310p_ascendc.tests.test_reference
for mtp in 3 7; do
  for stage in smoke sequence graph; do
    python3 "$op/run.py" --library "$lib" --mtp "$mtp" --batch 2 \
      --stage "$stage" --tile-v 64 --cores 7 --status-variant 2 \
      --output "$results/cube-mtp$mtp-$stage.json"
  done
done
```

精度通过后，先测 B32，再扩到 B64/B128。使用同一空闲设备，每轮
生成同一组 Q/K/V/g/beta、checkpoint、ring 和 metadata。
`status_variant=1` 是 V12，`2` 是整窗 Cube。可给下方命令加
`msprof --output=... --ai-core=on --aic-metrics=PipeUtilization` 前缀，
用 kernel 名和 timeline 区分独立 verify 与三算子组合的 launch。

```bash
for batch in 32 64 128; do
  for mtp in 3 7; do
    for variant in 1 2; do
      python3 "$op/run.py" --library "$lib" --mtp "$mtp" --batch "$batch" \
        --stage perf --tile-v 64 --cores 7 --status-variant "$variant" \
        --warmup 20 --iterations 100 \
        --output "$results/v$variant-mtp$mtp-b$batch.json"
    done
  done
done
```

回传首个编译或运行错误、精度 JSON 和同口径 p50/p90。
重点比较 `verify` 及 `combined`，同时记录 compact 和 cursor。
对比 pos0、pos12 两种 flush、pos16 flush。首次检查 FP32 L0C
搬出后的列布局、短窗零填充、ring 跨尾和 status 是否与 reference 一致。

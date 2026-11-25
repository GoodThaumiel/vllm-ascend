# 910B4 整窗 Cube verify、短 K compact 与 Native RGDR 对照

## 迁移判断

核对的远端分支名为 `replayssm-310`，tip 为 `d46aa506`。310P3 verify 将一个 key head 的窗口 K/Q 放入 128×16 面板，每个 value head 用一次 Mmad 投影完整 checkpoint，L0B 面板在两个 value head 间复用，FP32 结果交给 Vector 递推。compact variant 2 将 K/D 按环尾拆成至多两段，两个 value head 共读 K，在 FP32 UB 里用 Broadcast/Mul/Add 折叠。若现场还有独立 Cube compact 二进制，回传其 commit/symbol 可补一组直接对照。[310 verify 源码](https://github.com/GoodThaumiel/vllm-ascend/blob/d46aa506/experimental/replayssm_sd_310p_ascendc/kernels/verify_cube.cpp) · [310 compact 源码](https://github.com/GoodThaumiel/vllm-ascend/blob/d46aa506/experimental/replayssm_sd_310p_ascendc/kernels/compact_batch.cpp) · [310 交接记录](https://github.com/GoodThaumiel/vllm-ascend/blob/d46aa506/experimental/replayssm_sd_310p_ascendc/ascend310p-replayssm-sd-handoff.md)

在 910B4 TP4 上，Qwen3.5-27B 每 rank 为 HK4/HV12，一个 key head 有三个 value heads。把分组的三个 checkpoint 按 V 行拼成 [384,128]，共用 [128,16] RHS，得到每行每 key head 一次 GEMM。另一组独立 BMM 按每个 value head 的 [128,128] 投影作为对照。T4 有8列有效，T1仅2列有效。B1 只有4个 key head 组，后端可再按M切块，实际blockDim与尾波需看trace；B32已有128组。原V20每候选两次MatVec，T4可整窗合并，但 history correction、候选递推仍占时间。P0 B32/T4/p15 要从229.6降到187.8µs方能追平原生，约需省41.8µs，B64对应422.8→370.0，约需省52.8µs。此处仅是同轮旧verify单算子的目标差额。T4/p0原本已快于native，应关注任何回退。

compact 的原式为 H_new = exp(sum g) H + (D × suffix_weight)ᵀ K。p=16 的 rank 更新刚好适合短 K=16，p=1 只有1/16有效。flush 行多、历史长时更可能受益；真实收益取决于每轮各行的 flush 分布。310P3 批量 compact 的显式 UB 约216.5 KiB，超过910B4单AIV的192 KiB。后续手写版本需要切 V 或分时复用工作区。910B4 有20 AIC/40 AIV，310 的 L0C→UB 搬运、同核事件与两个 head 的调度不能按原样推算；优先验证 910B 可用的矩阵库出结果、Vector 消费结果的成本，若进入混合核实现再单独测试 Fixpipe、workspace 和跨核同步。[A2 Fixpipe 接口说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_store/Fixpipe_L0CToL1.md)

## 范围与精度

实验是可 capture 的 torch_npu 多算子 demo。使用框架 V20、compact v1 和 vllm-ascend 原生 recurrent_gated_delta_rule 同数据对比。torch.bmm 最终使用哪个算子/计算单元由当前二进制决定，必须从 trace 确认 Cube MAC、dtype 和转换。预打包 projection 含转回FP32的消费，fold_gemm 仅含矩阵乘；完整 demo 在计时内准备输入、修正、更新 ring 和状态。graph 包含多个 GM 中间结果，可量化当前实现成本，不能由它给单核融合速度定上限。V20 与 demo 的 checkpoint/ring 独立存储，数据槽是固定连续映射，demo 支持统一 pos/T/base，flush 行为固定前缀。compact mixed 已知 active 行数，真实服务需要在设备端识别动态 flush 行，须额外计费。

本轮 BF16 checkpoint、BF16 Q/K/V、FP16 D/K、FP32 G/beta（与 native 在 beta 边界对齐）。普通 BMM 的 BF16 投影输出转回 FP32，compact 的加权 D 会转 FP16 且 BMM 输出 FP16；后续折叠与缩放用 FP32。与低层 Mmad 直接保留 FP32 输出有舍入差异。CPU dense 参考、上机逐算子的 output/D/K/G/checkpoint 对比用于筛选可行性。未覆盖服务多轮分布、padding/槽位抢占；生产实现仍需验证这些场景。

## 执行

CANN 9.2.0、torch-npu 2.10.0.post4、vLLM 0.30.0+empty、已验证可调用 V20/compact v1 的扩展。此轮只用 Python，无需重编旧框架 .so。选空闲卡，所有性能 case 在同卡串行。每个 batch 做原生输出/V20、demo 输出及 ring/V20、compact checkpoint/v1 校验，再计时。任一失败保存批号、错误与之前完成的批次，停止该轮。reset 和黄金对照在 Event 外；reset 紧接计时会影响 cache 热度，同一模式保持一致。

```bash
set -euo pipefail
P=experimental/replayssm_cube_910b
R=/tmp/replayssm-cube910b-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$R"
git rev-parse HEAD > "$R/commit.txt"
git status --short > "$R/status.txt"
npu-smi info > "$R/device.txt"
python3 -m unittest tests.ut.replayssm.test_cube_910b
for t in 1 4; do
  for pos in 0 1 15 16; do
    python3 "$P/run.py" --device 0 --batches 1,2 --base 31 --pos "$pos" --length "$t" \
      --flush mixed --iterations 3 --output "$R/smoke-t$t-p$pos.json"
  done
done
for round in 1 2 3; do
  for scenario in '0 none' '1 full' '7 mixed' '15 full' '15 none'; do
    read -r pos flush <<< "$scenario"
    python3 "$P/run.py" --device 0 --length 4 --pos "$pos" --flush "$flush" \
      --iterations 100 --output "$R/r$round-p$pos-$flush.json"
  done
done
```

今后每个 verify/compact 版本统一比较 B=1,2,4,8,16,32,64,128；使用 `--batches all` 会逐整数扫 B1–128。小 batch、中间拐点和 B128 单独标注。追加业务 T1 或实测 (T,pos,flush) 分布时每种场景单独出图。每次回传含 actual native、Vector、candidate 和 verify+compact 同图对照的 JSON。cursor 一轮一次，按实际服务轨迹另计。建议以 full pair p50/p90、三轮漂移和服务实际 flush 比例决定是否进入手写混合核阶段。

## Profiling 与 roofline

对同一 B/pos/base/T/flush 分别运行下例，每个进程只计时一个 path，保留 P0 上机已验证的 msprof 配置。路径选择仍会运行正确性前置检查，按 MEASURE 与 graph replay 身份取样，排除预热、capture、reset、正确性和复制。对于多算子 graph 需聚合一次 replay 内**所有任务**的 FLOP 和 HBM 读写字节；耗时取相同口径的 graph Event 时间，单 kernel 的 task duration 另表回传。记录 AIC MAC、AIV Vector/Scalar、MTE、格式转换、核数/频率、内存层级及可获得的并行重叠信息。失败时保留首错以及是否有 CPU fallback。

```bash
for path in native v20 projection grouped_projection cube_verify compact_v1 fold_gemm cube_compact vector_pair cube_pair; do
  msprof --task-time=l0 --output="$R/prof-$path" python3 "$P/run.py" --device 0 \
    --path "$path" --batches 32 --pos 15 --flush full --iterations 100 --output "$R/prof-$path.json"
done
```

带宽、Cube 和 Vector ceiling 使用你此前在同型号、相应精度测得的 GB/s 和 TFLOP/s，或明确标成规格值。推荐对照之前已有的 910B roofline 校准记录。绘图依赖 matplotlib，可在 Mac 离线分析。下面可直接用 benchmark JSON 出延时曲线和建模 roofline，斜线为 HBM 带宽 × 算术强度，两条水平线分别给出 Cube 和 Vector 上限，混合路径落点需结合 profiler 解释。

```bash
python3 "$P/plot.py" "$R/r1-p15-full.json" --out "$R/p15-full.svg" \
  --bandwidth-gbs "$HBM_GBS" --cube-tflops "$CUBE_TFLOPS" --vector-tflops "$VECTOR_TFLOPS" \
  --roof-source "$CALIBRATION_RUN"
```

默认图标为 Model，用源码的主要计算量、逻辑 payload 和实测整图 p50。每点导出 CSV 便于检查。logical payload 不反映缓存命中、临时 GM 和格式转换，不能用模型点直接断言 memory/vector bound。图中零算术路径只有延时点。要绘制实测 roofline，在同一命令追加 `--counters path.csv`，列名为 `batch,path,flops,hbm_read_bytes,hbm_write_bytes`，每行对应该 JSON 的一个 (B,path)，用每 replay 均值、FMA=2 FLOPs、read/write 字节。只有实际指令计数可换算 FLOPs；pipe 时间占比和内存占用均不能代替计数。若工具仅给 task-time，则保留模型图并报告缺失计数。请附上校准来源、CSV单位/换算方法、原始msprof、环境和实际加载的.so身份。

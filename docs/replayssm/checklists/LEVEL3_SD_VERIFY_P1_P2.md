# Verify P1/P2：独立探针上机清单

本次提交只新增实验扩展，不改生产kernel/serve。平台沿用910B4/CANN9.2.0。
主场景27B TP4几何、MTP3、tile128、FP32 UB、BF16 checkpoint、FP16 D/K、FP32 G。
扩展位于`experimental/replayssm_sd_ascendc/probing`，只允许`--profiling-only`。

## 1. P0审核与实验口径

P0报告支持pos0→1的大跳变、pos1→15较缓增量；H1仍是inferred，不能写成已证实。
H2同样待A1检验。小MTE2增量不能排除DMA处于关键路径；pipe之和不能分解task时长。
84进程=28组合×before/v13/after，是一轮夹测，不是三次独立夹测；本次明确repeat三轮。
B32/T1/p0漂移4.59%不用于定量归因。Graph结果中B32/p0/T1 v20慢于v13，不能宣称全case更快。
Graph和旧compare输入不同，不能把15–30us差直接称launch overhead。MTE3下降不能仅凭组合实验定性为协同效应。

本次输入与P0不同：独立连续ring，checkpoint槽5..B+4、ring槽1..B，两个存储独立分配；
固定seed42，始终先生成16行history，再仅改变有效pos，避免跨pos时随机数消耗改变其他row的数据。
未使用cache元素填有限零值；状态机不推进，候选写区与有效history不相交。
必须在本轮probe baseline内做差，不能直接减P0的tied-index结果。

## 2. 编译（独立目录）

```bash
set -euo pipefail
P=experimental/replayssm_sd_ascendc/probing
RUN=/tmp/replayssm-p1p2-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN/compiler" "$RUN/environment" "$RUN/cases"
git rev-parse HEAD > "$RUN/environment/commit.txt"
git status --short > "$RUN/environment/dirty.txt"
npu-smi info > "$RUN/environment/device.txt"
python3 -m pip freeze > "$RUN/environment/packages.txt"
cmake -S "$P" -B "$P/build" -DSOC_VERSION=Ascend910B4 \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.0 \
  -DPython3_EXECUTABLE="$(command -v python3)" > "$RUN/compiler/configure.log" 2>&1
cmake --build "$P/build" -j8 > "$RUN/compiler/build.log" 2>&1
python3 -m unittest tests.ut.replayssm.test_verify_probing_fixture \
  tests.ut.replayssm.test_sd_v20_22_reference > "$RUN/compiler/cpu.log" 2>&1
```

构建失败停止，不加载旧.so。binding在build/，kernel依赖在build/lib/，RPATH为`$ORIGIN/lib`。
保存实际加载binding和kernel .so的hash、编译命令/对象/资源报告，若没有反汇编工具保留ELF给开发方。
MAC无法验证CCE编译和设备语义，本地C++检查仅使用接口stub做语法检查。

## 3. 模式

| mode | 定义 | 正确性要求与解释 |
| --- | --- | --- |
| baseline | 冻结V20快照 | B2与FP64 golden比较 |
| a1_control | V20 + history K sink | B2输出/cache仍对golden |
| a1_replace | 与control同sink，history dot改有限非零常数 | 非数值等价；需要核验dot确实删、后续correction仍在 |
| a2_control | V20 + D/dot/suffix sink，初始化dot padding | B2输出/cache仍对golden |
| a2_remove | 与control同sink，旁路Q/K history correction | 非数值等价；前序dot/D/suffix不能被DCE |
| local | V20只执行(row0,head0,tile0)，blockDim=1 | B2该head输出对golden；不是B1的12task |
| u1_chunk4k | 同一32KiB BF16 state分8次加载/Cast | 同一个MatVec消费全部state，再sink128；包含消费成本 |
| u1_chunk8k | 同一state分4次加载/Cast | input queue均为8KiB，隔离分块次数而非资源规划 |
| u2_history_tree | 热UB history correction的16行树 | 每轮清零dest+correction+sink；初始D装载也包含于task总时长 |

sink每task4096个FP32，分task独占；A1输出p×128 K元素；A2每candidate输出有效dot/suffix及p×128 D。
匹配control的sink写回开销可能很大，必须与baseline分别报告，不能把差值当纯阶段时长。
A1常数替代可能引发下游常量传播，若生成指令也改变后续区域则结论仅为组合删除的净效应。
源码使用同一outQueue转存sink，没有新增UB scratch；实际UB/栈与溢出以编译报告为准。
U2当前只有依赖式重复样本，独立buffer吞吐、L2片段和更多task缩放留待结果驱动。

## 4. 安全与数值门槛

```bash
DEVICE=0
for mode in baseline a1_control a1_replace a2_control a2_remove local u1_chunk4k u1_chunk8k u2_history_tree; do
  for pos in 0 1 16; do
    for length in 1 4; do
      python3 "$P/run.py" --profiling-only --mode "$mode" --batch 2 \
        --pos "$pos" --length "$length" --base 31 --device "$DEVICE" --iterations 2 \
        --output "$RUN/cases/safe-${mode}-p${pos}-t${length}.json" \
        > "$RUN/cases/safe-${mode}-p${pos}-t${length}.log" 2>&1
    done
  done
done
for mode in baseline a1_control a1_replace a2_control a2_remove; do
  python3 "$P/run.py" --profiling-only --mode "$mode" --batch 2 --pos 7 --padding \
    --device "$DEVICE" --iterations 2 --output "$RUN/cases/padding-${mode}.json" \
    > "$RUN/cases/padding-${mode}.log" 2>&1
done
```

runner检查sink两端guard、有限值和完整grid模式的status/out；B2 baseline/control对golden，local比较单head。
这不是完整内存访问证明；另检查编译/同步、NPU错误，出现首个问题停该模式。
不要并发跑性能测量，不reset其他人的卡；用执行方已有超时管理每个子进程。

## 5. P1 profiler：真正三轮夹测

每个组合**独立进程和symbol**，外层使用P0已验证的msprof设置。以下是应用参数模板；
不加profiler裸跑只输出Graph Event计时，不会产生pipe数据。
B32/64 × pos0/1/7/15 × T4/base0，每组合三轮。
每轮执行baseline → a2_control → a2_remove → baseline，再baseline → a1_control → a1_replace → baseline；
第二轮交换control/probe顺序，保留两端baseline。每次输出文件名含round、order、mode、B、pos。

```bash
python3 "$P/run.py" --profiling-only --mode a2_remove --batch 32 --pos 7 \
  --length 4 --base 0 --cores 40 --device "$DEVICE" --iterations 100 \
  --output "$RUN/cases/r1-a2-remove-b32-p7.json" > "$RUN/cases/r1-a2-remove-b32-p7.log" 2>&1
```

预期每symbol104次=安全1+warmup2+capture1+measure100，工具重放若有变化必须解释。
按单symbol/PID/device/stream实际顺序与MEASURE标记审计，不能照搬旧脚本103条规则。
结果记录每调用ID与阶段、task/aiv/vec/scalar/mte2/mte3/cycles、Graph样本、基线漂移。
本轮count实际核验后再裁掉4条；不能只见104就视为审计通过。
pos0为负对照；收益低于漂移和轮间波动时判inconclusive。

编译证据是归因门槛：A1两个history dot消失、spec dot/MatVec仍在；A2 Broadcast/Mul/tree消失，
history dot、D DMA/Cast与sink仍在。比较资源和sink增量。若只拿到代码体积，不足以证明目标指令删除。

## 6. P2局部/API样本

```bash
for mode in local u1_chunk4k u1_chunk8k u2_history_tree; do
  for pos in 0 1 7 15; do
    for repeats in 1 16 64; do
      python3 "$P/run.py" --profiling-only --mode "$mode" --batch 2 --pos "$pos" \
        --cores 1 --inner-repeats "$repeats" --iterations 100 --device "$DEVICE" \
        --output "$RUN/cases/${mode}-p${pos}-r${repeats}.json" \
        > "$RUN/cases/${mode}-p${pos}-r${repeats}.log" 2>&1
    done
  done
done
```

同样外包profiler，按R分别保存。U1与pos无关，先pos0一组，其他pos仅按需复核；
U2补pos2/4/8/16，用相同命令。用task关于R的斜率/截距分析，R增加不保证编译器保留循环，必须查生成指令。
local只执行一task；完整baseline可用`--cores 1/4/8/40 --batch 1/8/32/64`恢复多task调度，
但两者仅同一snapshot代码，任务数不同不能线性外推。完整grid禁止inner-repeats>1。

## 7. 回传

至少A1、A2、L1、U1四包；U2作为补充。每包包括实际命令、输入shape/stride/hash、
source/so hash、编译资源/指令差异、原始CSV/选择映射、Graph原始样本、三轮漂移、首个错误。
报告分别写observations和interpretation。api_facts实测字段在收到数据前继续null。
本轮无提速保证；probe不能进入serve，只有正确优化候选才恢复完整数值和服务回归。

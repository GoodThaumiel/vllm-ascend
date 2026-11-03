# Verify probing 第一轮上机清单（P0先行）

[方案](../VERIFY_PROBING_DESIGN.md) / [源码地图](../../replayssm_verify_probe_map.md) /
[结果包协议](../../../experimental/replayssm_sd_ascendc/probing/README.md)。

本次交付是方案与测试清单，**没有新增A1/A2等探针代码**。测试方现在执行第1–5节，
回传后再实现和运行第6节。不要等待不存在的probe入口，也不要自行把v15–v19当作删除式探针。
主基线v20，v13为原始对照，Native仅整体参考。以下从仓库根目录执行。

## 1. 环境与构建身份

使用已有CANN9.2.0容器、910B4空闲卡，不变更驱动/版本规避策略。
TP4是单卡张量几何模拟，不是启动4卡服务；无需模型权重。
由执行者确认DEVICE对应的物理卡可用，不操作其他人的卡。

```bash
set -euo pipefail
DEVICE=0
RUN=/tmp/replayssm-verify-probing-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN/environment" "$RUN/compiler" "$RUN/cases"
git rev-parse HEAD > "$RUN/environment/commit.txt"
git status --short > "$RUN/environment/status.txt"
git diff > "$RUN/environment/tracked.diff"
python3 -m pip freeze > "$RUN/environment/packages.txt"
npu-smi info > "$RUN/environment/device-before.txt"
msprof --help > "$RUN/environment/msprof-help.txt" 2>&1
```

另外保存CANN安装版本文件、驱动/固件、实际可见设备映射及卡温度/频率（工具能提供则保存，否则写不可得）。
不要导出全套环境变量或凭据。记录其他进程/卡负载，性能测试禁止与其他任务并行。
本提交仅文档；已经从5b0a2b0重建且支持v20的.so可复用，但必须写出其构建commit和hash。
不确定.so来源时重建并保存完整日志：

```bash
python3 -m pip install --no-build-isolation --no-deps -v -e . > "$RUN/compiler/build.log" 2>&1
python3 -m unittest tests.ut.replayssm.test_sd_v20_22_reference \
  tests.ut.replayssm.test_sd_v15_19_reference \
  experimental.replayssm_sd_ascendc.tests.test_reference > "$RUN/compiler/cpu-tests.log" 2>&1
python3 tests/ut/replayssm/test_framework_meta.py > "$RUN/compiler/meta-tests.log" 2>&1
```

从实际import路径确认.so，保存绝对路径和SHA256，禁止仅对build目录的任意旧.so计算hash。
保留编译器命令、kernel对象和资源报告；若本次复用.so且产物不在，注明缺失。

## 2. 原版安全与图回归（先通过再计时）

```bash
for variant in 13 20; do
  for stage in smoke graph; do
    python3 experimental/replayssm_sd_ascendc/run.py \
      --backend framework --tp 4 --geometry 27b --mtp-size 3 --batch 2 \
      --tile-v 128 --cores 40 --variant "$variant" --stage "$stage" --device "$DEVICE" \
      --output "$RUN/cases/v${variant}-${stage}.json" \
      > "$RUN/cases/v${variant}-${stage}.log" 2>&1
  done
  python3 scripts/replayssm/check_sd_independent_ring.py \
    --tp 4 --geometry 27b --mtp-size 3 --variant "$variant" --device "$DEVICE" \
    --output "$RUN/cases/v${variant}-independent.json" \
    > "$RUN/cases/v${variant}-independent.log" 2>&1
done
```

任何编译失败、status非零、精度超差、NPU fault或hang停止本轮，回传首个失败。
原版kernel必须过golden，不能使用“probe允许不正确”豁免。

## 3. 单case profiler 与103次调用核验

先只跑v20/B32/T4/pos7/base0。测试方使用之前已验证的真机msprof调用方式，
设置与历史L3.13相同（`--task-time=l0`及当时实际pipe采集配置）。保存完整外层命令和工具版本。
不同MindStudio版本参数存在差异，本清单不引入未确认的片上timeline选项。
下面的`python3 ...`是待profiler包装的应用命令；**裸跑不会产生task/pipe CSV**。

```bash
python3 scripts/replayssm/profile_sd_operator_compare.py \
  --tp 4 --batches 32 --context-lengths 32768 --lengths 4 --accepted 1 \
  --positions 7 --base 0 --ring 32 --tile-v 128 --cores 40 --variant 20 \
  --repetitions 100 --warmup 2 --seed -32 --device "$DEVICE" \
  --output "$RUN/cases/audit-v20-b32-p7-t4.json" \
  > "$RUN/cases/audit-v20-b32-p7-t4.log" 2>&1
```

负seed参数是有意的：脚本内部实际seed=`seed+B*1000+pos*10+T`；上例得到32042。
通式传入`42-pos*10-T`，这样同batch跨pos/T保持相同原始tensor。
这是合成输入实验，context=32768仅标签，不增加算子输入shape或有效history。

验收：精确匹配Replay V20 symbol、device、PID和stream，按执行顺序确认
1次check + 2次warmup + 100次测量=103条；Native另分symbol处理。
用PROFILE_BEGIN/END对应的record_function/task关联验证最后100确属测量段。
若工具没有关联信息，只能在数量/顺序与源码一致且没有额外工具重放时采用索引排除前三条，
并标记`selection_method=ordered_single_case`及局限。数量不符不得任意裁剪。
输出`selected_calls.csv`保留原文件名、原行号、task/stream ID、选择/排除原因和单位。

确认此审计通过之后才展开矩阵，否则先回传审计材料，不收集大量不可用的混合统计。

### 最小矩阵与运行顺序

每个case独立进程/输出目录，不能拼到单个应用里再按耗时排序切分。
同一批数据至少3轮：每case顺序v20-before、v13、v20-after。
这不是A1/A2实验，只是比较可信度与pos曲线的先行测量。

| case集 | B | T | pos | base | 轮数 |
| --- | --- | --- | --- | --- | --- |
| 首轮发现集 | 32,64 | 4 | 0,1,7,15 | 0 | 3 |
| 发现集稳定后补点 | 32,64 | 4 | 2,4,8,12 | 0 | 3 |
| 候选长度控制 | 32,64 | 1 | 0,1,7,15 | 0 | 3 |
| 环尾控制（第二因素） | 32,64 | 4 | 15 | 16,31 | 3 |

应用命令模板（每次将B/P/T/V/RUN_ID替换为具体值，并外包同一profiler配置）：

```bash
B=32; P=15; T=4; V=20; RUN_ID=r1-before-b32-p15-t4
python3 scripts/replayssm/profile_sd_operator_compare.py \
  --tp 4 --batches "$B" --context-lengths 32768 --lengths "$T" --accepted 1 \
  --positions "$P" --base 0 --ring 32 --tile-v 128 --cores 40 --variant "$V" \
  --repetitions 100 --warmup 2 --seed "$((42-P*10-T))" --device "$DEVICE" \
  --output "$RUN/cases/${RUN_ID}.json" > "$RUN/cases/${RUN_ID}.log" 2>&1
```

该脚本未传独立ring_slots，记录`layout=tied_index_page_strided_proxy`。
第2节独立ring功能通过不等于此处性能已代表独立分配；正式删除式实验runner须另行实现独立ring。
base0时history不跨尾，pos0→1跳变不会混入额外环尾DMA。
不把CPU wall计时、task_us、aiv_us混为一个指标。

## 4. Graph延迟和已有API参考

Graph Event计时单独运行，不能从第3节的host_range_seconds推导。
现有run.py输出固定case集合（含pos0/7/12/15/16），不支持任意pos1/2/4/8；
这些点本轮先有profiler值，Graph栏填null。不要编造不支持的过滤参数。
至少三轮，v20-before/v13/v20-after；下例是一个单次测量：

```bash
python3 experimental/replayssm_sd_ascendc/run.py \
  --backend framework --tp 4 --geometry 27b --mtp-size 3 --batch 32 \
  --tile-v 128 --cores 40 --variant 20 --stage perf --iterations 100 --device "$DEVICE" \
  --output "$RUN/cases/graph-r1-before-v20-b32.json" \
  > "$RUN/cases/graph-r1-before-v20-b32.log" 2>&1
```

同参补B64、variant13和各轮，逐case比较；run.py与compare的数据生成口径不完全相同，
两组结果各自在自身对照内比较，不能直接用graph-task做差解释launch成本。

可运行已有metadata API参考（它不是新的U1/U2）：

```bash
for B in 32 64; do
  python3 scripts/replayssm/benchmark_sd_metadata.py \
    --batch "$B" --heads 12 --cores 40 --iterations 100 --mode both --device "$DEVICE" \
    --output "$RUN/cases/metadata-b${B}.json" > "$RUN/cases/metadata-b${B}.log" 2>&1
done
```

其计时包含相同的对齐sink写回，不含pack，不能称纯32B load延迟。
若需要profiler，按separate/packed分别抓，不混合两模式聚合。

## 5. 回传与P0验收

每个case分别报告task_us、aiv_us、vec/scalar/mte2/mte3、cycles、Graph p50/p90/min（不可得填null）。
保留每轮分布，不把B32/B64/T/pos混在一起取中位数；不同pipe不能相加。

| 必需产物 | 检查点 |
| --- | --- |
| 环境/身份 | commit、dirty patch、.so路径/hash、构建日志/选项或复用来源、版本、卡状态 |
| 实际命令 | 每case含外层profiler完整命令、seed、输入shape/stride、base、布局说明 |
| 原始与选择数据 | 原CSV/JSON/log、测量100条的映射和排除理由 |
| 编译证据 | kernel对象/IR/反汇编、资源报告；不可得时具体说明 |
| 基线稳定性 | 前后漂移、三轮配对差值、IQR；>3%漂移建议复跑 |
| 初步事实 | pos0→1、pos1→7、pos7→15的task与pipe差，不直接声明根因 |

结果包按README组织，原始大文件外存附hash。报告只写P0完成项，
A1/A2、局部采样、U1/U2和模型校准保持pending。结束后确认本轮无残留进程/设备错误。

## 6. P1/P2后续执行约定（当前不要运行）

待专用实验kernel/runner提交后，以本轮P0同设备/口径进行：

1. 安全：B2，pos0/1/16，T1/4，padding和独立ring跨尾；status/canary/finiteness及同步正常。
2. A2-control/A2，再A1-control/A1：B32/64、T4、pos0/1/7/15，3轮baseline-control-probe-baseline；
   编译核验目标删除且上游sink依赖保留，否则标为DCE污染，不能做单阶段归因。
3. L1：真正一个task一个core，pos0/1/7/15，R1/16/64，再按设计逐步恢复task/core数。
4. U1及选定U2：区分依赖链/吞吐、热UB/GM、重复数；同口径empty/sink control。
5. A3/A4仅由A1/A2残差触发。必须有新manifest和新symbol，不临时改生产kernel。

此阶段才要求至少4个实验包（两消融、一局部、一API）与两类有证据的成本来源。

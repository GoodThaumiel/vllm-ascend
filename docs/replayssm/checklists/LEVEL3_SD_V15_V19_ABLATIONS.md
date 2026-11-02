# L3.11 SD verify v13 派生消融：Qwen3.5-27B TP4 / MTP3

本轮只新增独立单算子实验，`vllm serve` 继续使用 v13 verify 和 v1 compact；v14 是单独的 verify+compact 融合算子，不是本轮 variant。目标：910B4、CANN 9.2.0、torch_npu 2.10.0.post4、vLLM 0.30.0+empty、vllm-ascend 同一源码构建。所有新变体使用 `--backend framework --tile-v 128`、FP32 UB、BF16 checkpoint、FP16 D/K、FP32 G、cores=40。

| variant | 与 v13 唯一主要差别 | 需要关注 |
| --- | --- | --- |
| 13 | 原有基线 | 同容器与同设备每轮重测 |
| 15 | history coefficient 保留广播；逐 history row 用 `MulAddDst` 累积，取代 weighted tree | 归约顺序和 FP32 舍入变化；未移除 broadcast，保留其他 UB 分配以隔离计算 |
| 16 | checkpoint 输入缓冲 4 KiB → 8 KiB，8 次 → 4 次读 | 分配估计 188,448 B，须检查 CCE 临时 UB |
| 17 | speculative dot 保持在 UB，逐行加权累积，无固定 8 行树归约 | 改变累加顺序及小行广播，可能比 v13 和已有 v8 慢 |
| 18 | Q/K MatVec 按 16 行分块交错，复用无活跃值的 weighted-history 工作区 | 每任务从一次大归约变多次小归约，可能性能回退；精度和 UB 生命周期 |
| 19 | 候选 gate 聚合成一次写，跨 ring 尾最多两次 | 只改变 G 写回；metadata 仍是原 ABI 的 5 次读取 |

变体独立基于 v13，不累加。按源码静态 `InitBuffer` 估算：v13/v15/v18/v19 184,352 B；v16 188,448 B；v17 184,736 B。192 KiB=196,608 B；CCE 的实际分配和临时空间为准。16 KiB 输入缓冲在保留 32 value head 容量时预计需要 196,640 B，超过 192 KiB 32 B，所以本次不加入会影响全扩展构建的 16 KiB 变体。

## 1. 编译和本地参考

拉取后重编 910B 主扩展，检查 5 个 `replayssm_sd_verify_v15...v19` 均进入编译，保存完整构建日志、各 kernel 的 CCE UB 报告和 `.so` 路径/hash。任一编译失败停止，勿复用旧 `.so`。

```bash
python3 -m pip install --no-build-isolation --no-deps -v -e . 2>&1 | tee /tmp/l311-build.log
python3 -m unittest tests.ut.replayssm.test_sd_v15_19_reference experimental.replayssm_sd_ascendc.tests.test_reference
python3 tests/ut/replayssm/test_framework_meta.py
```

上面的 CPU 测试仅覆盖数学和 ring 索引，不能确认 AscendC 编译、指令精度或异步同步正确性。

## 2. 正确性、图回放与独立 ring

先单卡 TP4 几何、B2、每个 variant 分别跑 smoke 和 graph；图重放会改变 position/length/base。覆盖 pos=0/7/12/15/16、length=1/2/3/4、padding 和 ring 跨尾；比较 FP64 dense golden 的 output、D/K/G、checkpoint 及 status。接着跑独立 ring 探针，同时比较 tied/detached 和 compact 后 checkpoint。

```bash
for variant in 13 15 16 17 18 19; do
  for stage in smoke graph; do
    python3 experimental/replayssm_sd_ascendc/run.py \
      --backend framework --tp 4 --geometry 27b --mtp-size 3 \
      --batch 2 --tile-v 128 --cores 40 --variant "$variant" --stage "$stage" \
      --output "/tmp/l311-v${variant}-${stage}.json" || exit 1
  done
  python3 scripts/replayssm/check_sd_independent_ring.py \
    --tp 4 --geometry 27b --mtp-size 3 --variant "$variant" \
    --output "/tmp/l311-v${variant}-detached.json" || exit 1
done
```

任何非零 status、精度超差、图 capture/replay 失败或 NPU error，保存首个失败结果并停止该变体。先不要因 15/17 的计算顺序变化放宽现有 `atol=0.01, rtol=0.03` 门槛；报告各输出误差，必要时进行逐项定位。V19 特别检查跨尾前后未写入的 ring 槽和其他 head 是否保持不变。

## 3. 真实性能对比

精度通过后，相同机器、同一张空闲卡、同一容器串行测 v13/15/16/17/18/19，先 B32 再 B64。现有 perf 脚本会在每次计时前恢复 cache；恢复不计入 Event 时间，但图回放仍含设备运行环境开销。JSON 含 pos=0/7/15 与 length=1/4，以及 pos12/16 的 T4 对照。

```bash
for batch in 32 64; do
  for variant in 13 15 16 17 18 19; do
    python3 experimental/replayssm_sd_ascendc/run.py \
      --backend framework --tp 4 --geometry 27b --mtp-size 3 \
      --batch "$batch" --tile-v 128 --cores 40 --variant "$variant" \
      --stage perf --iterations 100 \
      --output "/tmp/l311-v${variant}-b${batch}-perf.json" || exit 1
  done
done
```

在相同 case、base、dtype、batch 和 warmup 下用 msprof 分别抓取任务名与 `task_duration/aiv_time/aiv_vec/aiv_scalar/aiv_mte2/aiv_mte3`；这些时间可能重叠，不相加推断关键路径。下面脚本每个 case 分别调用原生和实验 verify 100 次，打印 `PROFILE_BEGIN/END`；它的 host 总时间不是 graph replay latency。`--base 16`、pos15 使候选跨 ring 尾；kernel 统计按 `PROFILE_BEGIN/END` 区分。

```bash
for variant in 13 15 16 17 18 19; do
  python3 scripts/replayssm/profile_sd_operator_compare.py \
    --tp 4 --batches 32,64 --context-lengths 32768 --lengths 1,4 \
    --positions 0,7,15 --base 16 --tile-v 128 --cores 40 \
    --variant "$variant" --repetitions 100 \
    --output "/tmp/l311-profile-v${variant}.json" || exit 1
done
```

在上述命令外套用本机已验证的 `msprof` 启动方式。native 检查和第一次运行可能占用 profiler 范围；只汇总命名正确且位于 100 次区间的任务。verify-only 性能不能直接代表 verify+compact 或服务端到端性能。

若需要 simulator 对照，使用同一 driver 输入，分别编译入口 `replayssm_sd_verify_tile128_bulk_qkv` 和 `replayssm_sd_verify_v15_history_fma` 等新入口；保留相同有效 slot、非零数据、pos15/T4 和 ring base，检查 status 后采集。按 PC 映射热点及 trace 重叠，不能把不同指令/pipe 的 cycles 求和当作壁钟时间。

回传 commit、dirty 状态、build log/UB 占用、`.so` hash、每阶段 JSON、设备任务摘要、首个失败 traceback 与设备健康状态。此轮不根据模拟结果或 graph p50 单独替换服务默认值。

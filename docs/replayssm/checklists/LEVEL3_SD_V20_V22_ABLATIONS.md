# L3.12 SD verify V20–V22：27B / TP4 / MTP3

环境：Ascend910B4、CANN 9.2.0、torch_npu 2.10.0.post4、vLLM 0.30.0+empty；source-built vllm-ascend。单卡运行 TP4 几何：Q/K heads=4、value heads=12、K/V=128、logical history ≤16、ring=32，BF16 checkpoint / FP16 D,K / FP32 G / FP32 UB，tile128、cores40。

这轮是独立算子实验，`vllm serve` 仍使用 v13 verify + v1 compact。不能假设各 variant 的收益相加；真实耗时以同一 workload 的实测为准。

| variant | 实验 | 主要边界 |
| --- | --- | --- |
| 13 | 原基线 | 每个 request/value head 一个任务 |
| 16 | 8 KiB checkpoint | 对比 V20 |
| 19 | 聚合 G 写入 | 对比 V20 |
| 20 | V16 + V19 | 两种优化的相互作用；相同 workload 对比 |
| 21 | V13 + 一次读取 int32[8] packed metadata | 独立实验的 binding 新增可选 `packed_metadata`；真实 cursor 元数据在设备侧更新，仍需把打包生产与 graph 生命周期接入框架后才能用在服务 |
| 22 | V13 + `(request,key_head)` 分组 | 仅 27B TP4；每任务顺序处理 3 个 value head，仅共享原始 Q/K、history K 和原始 dot；checkpoint、D、G、suffix、beta、输出分别处理 |

V22 tasks 从 B×12 降为 B×4：B1/B2/B8 至多使用 4/8/32 个 vector core，B16 及以上最多 40 核。V22 有独立函数，不改原版任务循环；不同编译条件仍需重测基线。

V21 的 `packed_metadata` 为设备上连续 int32[B,8]，列 0–4 依次为 checkpoint slot、request ring slot、position、base、length，其余为 padding。生产者必须在每次 replay 前保证五列与本轮 metadata 一致；设备端 kernel 以 packed 值为准，不会将其与单独输入同步比较。它是 verify operator schema 的**可选新参数**，属于框架 binding 的侵入式扩展；现有调用可省略但必须重新编译扩展。V21 的测试打包采用 5 次设备侧列复制，是便于测量的参考生产方式，未接入 cursor，也不代表最终生产方案。V22 分组重用 UB 和控制路径是高风险算子改动；本轮不自动启用。

静态 UB 显式分配估算：V13/V21/V22 约 184,352 B，V20 约 188,448 B（192 KiB=196,608 B）。V22 的标量数组、编译器临时内存或溢出可能增开销，实际 CCE 资源报告为准。

## 1. 构建与 CPU 门槛

在 910B4 构建扩展，保存日志、`.so` hash、三个 kernel 和 metadata probe 的 CCE 资源报告。任一编译失败先停止，不加载旧 `.so`。

```bash
python3 -m pip install --no-build-isolation --no-deps -v -e . > /tmp/l312-build.log 2>&1
python3 -m unittest tests.ut.replayssm.test_sd_v20_22_reference \
  tests.ut.replayssm.test_sd_v15_19_reference experimental.replayssm_sd_ascendc.tests.test_reference
python3 tests/ut/replayssm/test_framework_meta.py
```

CPU 参考检查不同 value head 的 gate/state、混合长度、padding、跨尾、分组覆盖、独立 ring 槽和 metadata 动态变化；不验证 AscendC 编译器、同步或设备表现。

## 2. 先做 V21 的 metadata 微基准

两模式使用同一组 metadata、B×12 个任务和完全一致的每任务 32B 对齐输出，比较五次小读取与一次 32B 读取。graph capture 后改变 metadata，再验证输出；报告值**不包含 pack 成本**。输出写回、kernel 固定成本与 graph Event 开销均包含在两组读数内，因此差值不是纯单次 DMA 硬件延迟。

```bash
for batch in 1 2 8 16 32 64; do
  python3 scripts/replayssm/benchmark_sd_metadata.py \
    --batch "$batch" --heads 12 --cores 40 --iterations 100 --mode both \
    --output "/tmp/l312-meta-b${batch}.json" || exit 1
done
```

如需分别抓两模式的 profiler，在同一个 batch 下分别使用 `--mode separate` / `--mode packed`，按 `PROFILE_BEGIN/END` 与 `replayssm_sd_metadata_probe` 识别任务。请统计固定 batch/case 的 task duration、AIV、Scalar、MTE2/MTE3；避免混合多个 batch 排序后再反推标签。

## 3. verify 正确性 / graph / 独立 ring

先在 TP4/B2 运行。每个 variant 同样覆盖 pos=0/7/12/15/16、T=1/2/3/4、padding、不同 head 的随机 gate 与 state、history/candidate 环形跨尾，并对 output、D/K/G、checkpoint、status 与 FP64 dense golden 比较。V21 的 graph 校验额外在 replay 前清零 packed buffer，再验证图内 5 列 pack + verify 的结果，避免旧 metadata 假通过。

```bash
for variant in 13 16 19 20 21 22; do
  for stage in smoke graph; do
    python3 experimental/replayssm_sd_ascendc/run.py \
      --backend framework --tp 4 --geometry 27b --mtp-size 3 --batch 2 \
      --tile-v 128 --cores 40 --variant "$variant" --stage "$stage" \
      --output "/tmp/l312-v${variant}-${stage}.json" || exit 1
  done
  python3 scripts/replayssm/check_sd_independent_ring.py \
    --tp 4 --geometry 27b --mtp-size 3 --variant "$variant" \
    --output "/tmp/l312-v${variant}-detached.json" || exit 1
done
```

任一非零 status、精度超差、graph 失败、NPU error 均停止对应 variant，保留首个错误。特别检查 V22 的三个 value heads 是否分别使用正确 G/beta/D/state，padding head 的所有输出/status 是否覆盖，以及共享 K 写回跨尾是否正确。`check_sd_independent_ring.py` 比较 tied 与 detached 后再运行 compact，用于检查槽号隔离。

## 4. graph 性能：逐 workload 比较

正确性通过后，在同设备、同容器、空闲卡串行测。先 B32/B64，再测 B1/B2/B8/B16；每个 case 同 seed、重置在 Event 计时区外。关注 `pos0_len1`、`pos15_len1`、`pos15_len4`，以及输出中的其他 cases：

```bash
for batch in 32 64 1 2 8 16; do
  for variant in 13 16 19 20 21 22; do
    python3 experimental/replayssm_sd_ascendc/run.py \
      --backend framework --tp 4 --geometry 27b --mtp-size 3 --batch "$batch" \
      --tile-v 128 --cores 40 --variant "$variant" --stage perf --iterations 100 \
      --output "/tmp/l312-v${variant}-b${batch}-perf.json" || exit 1
  done
done
```

每个 JSON 的 `graph_replay_us` 是 verify-only。V21 另外输出 `graph_pack_plus_verify_us`，其中计入五列设备 copy 与一次 verify 的同图执行；这个单层口径不代表把 pack 一次摊到所有 GDN 层，也不代表实际框架已实现一次 pack。记录 p50/p90 和小 batch 的有效核数；若 V22 慢，需要按任务量与单任务长度分析，不以比值判断根因。

## 5. 固定 workload 的 profiler

在已验证的真机 profiler 外层启动下面命令；一次只指定一个 batch、position 和 length，将结果以 `variant/batch/position/length` 标记。示例只给 B32/pos15/T4；分别补 B32/pos0/T1、B32/pos15/T1、B64/pos15/T4：

```bash
for variant in 13 16 19 20 21 22; do
  python3 scripts/replayssm/profile_sd_operator_compare.py \
    --tp 4 --batches 32 --context-lengths 32768 --lengths 4 --positions 15 \
    --base 16 --tile-v 128 --cores 40 --variant "$variant" --repetitions 100 \
    --output "/tmp/l312-profile-v${variant}-b32-p15-t4.json" || exit 1
done
```

V21 此脚本预先生成 packed metadata，profiler 核函数数据不包含打包；打包成本另见第 4 节。核对 `PROFILE_BEGIN/END` 的 native 和 replay 任务，保留每 case 的 task duration、AIV、Vector、Scalar、MTE2、MTE3，以及采集模式/预热/卡状态。`aiv_*` 单元时间可重叠；每种优化的因果解释还需结合相同 case 的变化，不把混合 case 的中位数当作某个固定场景。

回传 commit、dirty 状态、构建日志和 CCE 资源、`.so` hash、各阶段 JSON、固定 case profiler 输出及任何首个失败 traceback。V20 只有在可重复优于候选、正确性与服务验证通过后才考虑成为新默认；V21/V22 不进行自动服务接入。

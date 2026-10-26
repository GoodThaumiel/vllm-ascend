# L3 SD verify v13：以 v11 为基线的 tile128 验证清单

环境：Ascend910B4、CANN 9.2.0、torch_npu 2.10.0.post4、vLLM 0.30.0+empty、源码编译的 vLLM Ascend。v11/tile64 已通过 TP4/TP8 精度与图回放，TP4/B64/pos12/T4 的 `task_duration` 为 518.90 µs、约 native 的 1.396×，作为本轮基线。v12/tile128 已编译通过，但带 history 的 smoke case 稳定出现 `d_cache` 超差，因此不得作为性能基线。

v12 的 history 树归约前三层给 FP32 高维 `Add` 传入 `mask=tileV=128`，而[官方 API 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00257.html)规定该 mask 最大为 64。v12 仅修正最后一级的 8-bit stride 溢出；此前三层的 mask 仍越界。这是确定的代码错误，也符合“pos0 通过、带 history 失败”的现象；现有报告尚不能排除其 `product_`/history 临时区分时复用存在第二个问题。

v13 从 v11 扩展 tile128，保留候选 Q/K/V 多行搬运；history 树归约前三级各以两个 64 元素的合法 mask 覆盖整行，末级以 count API 处理 128 元素。history 的 `expandedCoeff_`、`weightedHistory_` 继续独立分配，不与 `product_` 别名。`in_` 缩至 4 KiB，checkpoint 分八段读入。按源码显式申请，v13 为 **183,392 B**，距 192 KiB UB 上限有 **13,216 B**，仍需 CCE 编译器确认临时空间。每个任务的 V tile 数从 v11/tile64 的 2 降至 1；额外 checkpoint 搬运及归约指令可能抵消收益，不能预设性能。

1. 拉取本次 commit，在同一 9.2.0 容器重编主扩展。确认 `replayssm_sd_verify_tile128_bulk_qkv` 编译通过，保存 CCE 实际 UB 占用、首条错误/警告和构建日志。若报 UB 不足或编译失败，在此停止；不要用旧扩展测试 v13。

```bash
python3 -m pip install --no-build-isolation --no-deps -v -e .
```

2. **只做正确性**：TP4/B2/tile128/40 cores，先 smoke，后 graph。smoke 覆盖 pos0/T1、pos7/T2/3/4、pos12/T4、pos16/T4、环绕/非环绕与 padding。以现有 `run.py` 的 FP64 golden 和 `atol=0.01, rtol=0.03` 判断，检查 `out/d_cache/k_cache/g_cache`、status、checkpoint 不变，并与已通过的 v11/tile64 同 case 比较。任一失败立即停下，回传失败 JSON、报错索引及值；不要进入 perf。

```bash
for stage in smoke graph; do
  python3 experimental/replayssm_sd_ascendc/run.py --backend framework \
    --stage "$stage" --tp 4 --batch 2 --tile-v 128 --cores 40 \
    --variant 13 --device 0 \
    --output "/tmp/sd-v13-tp4-${stage}.json" || exit 1
done
```

3. 精度与图回放均通过后，再在相同容器比较 v11/tile64 与 v13/tile128，B32/B64、pos7/12/16、T4 的 p50/p90。reset 保持在计时区外。记录每版的 CCE UB 占用和有效核心数；性能回退也需回传，不切换默认 variant。

```bash
for batch in 32 64; do
  for variant in 11 13; do
    if [ "$variant" -eq 13 ]; then tile=128; else tile=64; fi
    python3 experimental/replayssm_sd_ascendc/run.py --backend framework \
      --stage perf --tp 4 --batch "$batch" --tile-v "$tile" --cores 40 \
      --variant "$variant" --iterations 50 --device 0 \
      --output "/tmp/sd-v13-b${batch}-v${variant}.json" || exit 1
  done
done
```

4. 若 v13 具有正收益，再用同一 `msprof` 比较 native、v11/tile64、v13/tile128 的 B64/pos12/T4（100 次计时）。回传 `task_duration`、AIV、Vector、Scalar、MTE2/MTE3 绝对时间，并特别看 checkpoint 八段搬运与 history 归约多指令的代价。TP4 全通过后补 TP8 smoke/graph，再决定是否扩展 TP8 性能测试。

```bash
for variant in 11 13; do
  if [ "$variant" -eq 13 ]; then tile=128; else tile=64; fi
  python3 scripts/replayssm/profile_sd_operator_compare.py \
    --tp 4 --device 0 --batches 64 --context-lengths 32768 \
    --lengths 4 --positions 12 --ring 32 --tile-v "$tile" --cores 40 \
    --variant "$variant" --repetitions 100 --warmup 2 \
    --output "/tmp/sd-v13-profile-v${variant}.json" || exit 1
done
```

请附源码 commit、`git status --short`、构建日志/UB 资源报告、所有 JSON 和 profiler 摘要。若 v13 仍只在带 history 的 case 失败，下轮先隔离树归约与缓存写入的首个发散中间值，再修改算术；不要仅凭 `d_cache` 的最终误差推断同步或别名原因。

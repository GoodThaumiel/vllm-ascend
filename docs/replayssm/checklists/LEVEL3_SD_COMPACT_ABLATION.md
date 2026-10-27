# L3.4：SD compact 元数据与状态写入消融

环境：Ascend910B4、CANN 9.2.0、torch_npu 2.10.0.post4、vLLM 0.30.0+empty、源码编译的 vllm-ascend 0.30。checkpoint/Mamba state 为 BF16，d/k ring 为 FP16，g ring 为 FP32。本次只改独立 compact 算子；`vllm serve` 的 SD/MTP dispatch 仍未接线。

## 三种实现

| variant | 映射 | 非 flush 路径 | flush 路径 |
|---|---|---|---|
| 0 | `(row, value head, V tile)` | 每 tile 四次元数据读和一次 4B status 写；L3.3 对照 | 原版 |
| 1 | `(row, value head)` | 每 head 读一次 flush；仅 head 0 读其余元数据、一次连续写整行 status | 每 head 读一次元数据并顺序处理各 V tile |
| 2 | `(row, value head, V tile)` | 每 tile 读一次 flush；仅首 tile 读其余元数据、一次连续写整行 status | 保留原版 tile 级并行与计算，仅把 flush 读提前 |

`variant=0` 仍是默认值。所有版本在图中按设备端逐请求的 flush 标志分支，调用数量不变；不使用 host 同步或按多数请求跳过。variant 1/2 的整行 status 写要求每行 status 为 32 字节整数倍，TP4/TP8 的 tile16/32 几何满足这一条件。它们只改变任务映射、元数据和 status 搬运，checkpoint 折叠公式、ring 读写范围及 cursor 语义不变。

## 上机顺序

1. 拉取提交并在原 CANN 9.2.0 容器重新构建主扩展。确认 `replayssm_sd_compact` 编译、加载、`RUNPATH`、Meta schema 和 `variant` 参数正常；保存首条 CCE 错误及 UB 报告。若编译失败，在此停止，不运行旧 `.so`。
2. 先运行 CPU 参考，再依次运行 TP4 B2 smoke、TP4 B2 graph、TP8 B2 graph。对每个 variant 都检查 `pos0`、`pos13/base31`、`pos16/base31`、`flush=0`、同 batch 混合 flush 和 padding。核对 checkpoint、d/k/g ring 和每个 status 元素；图捕获后修改 flush/position，结果必须跟着变化。
3. 精度与图模式通过后，测 TP4/TP8 × B1/B32/B64 × tile16/32 × variant0/1/2，先固定 cores40，再对最优候选补 cores20/8。每个 case 至少 50 次 graph replay，记录 p50/p90/min；重置缓存不计入计时。重点比较 `pos13_base30_flush0` 与 `pos13/16_base31_flush1`，不要把 pos0 当作典型非 flush 路径。
4. 对 B32、B64、TP4、tile32、cores40 的三种 variant 抓 msprof，分别记录 kernel task duration、AIV Scalar/Vector/MTE2/MTE3 和 GM bytes。各引擎可能重叠，不可相加解释总时延。随后用同一环境的 lifecycle 小范围脚本量测 `verify+compact+cursor`；`cursor` 一次/batch step，compact 一次/GDN 层/speculation iteration。

```bash
python3 -m pip install --no-build-isolation --no-deps -v -e .
python3 -m unittest experimental.replayssm_sd_compact_ascendc.tests.test_reference
python3 tests/ut/replayssm/test_framework_meta.py

for variant in 0 1 2; do
  python3 experimental/replayssm_sd_compact_ascendc/run.py \
    --backend framework --stage smoke --tp 4 --batch 2 --tile-v 32 \
    --cores 40 --variant "$variant" --device 0 \
    --output "/tmp/replayssm-compact-v${variant}-tp4-b2-smoke.json"
  python3 experimental/replayssm_sd_compact_ascendc/run.py \
    --backend framework --stage graph --tp 4 --batch 2 --tile-v 32 \
    --cores 40 --variant "$variant" --device 0 \
    --output "/tmp/replayssm-compact-v${variant}-tp4-b2-graph.json"
  python3 experimental/replayssm_sd_compact_ascendc/run.py \
    --backend framework --stage graph --tp 8 --batch 2 --tile-v 32 \
    --cores 40 --variant "$variant" --device 0 \
    --output "/tmp/replayssm-compact-v${variant}-tp8-b2-graph.json"
done

for tp in 4 8; do
  for batch in 1 32 64; do
    for variant in 0 1 2; do
      python3 experimental/replayssm_sd_compact_ascendc/run.py \
        --backend framework --stage perf --tp "$tp" --batch "$batch" \
        --tile-v 32 --cores 40 --variant "$variant" --iterations 50 --device 0 \
        --output "/tmp/replayssm-compact-v${variant}-tp${tp}-b${batch}-perf.json"
    done
  done
done
```

如 variant 1 在 flush 重路径退化，优先比较 variant 2；如非 flush 的 Task Duration 没下降，回传三版 kernel 的首个 msprof 对照，下一轮再考虑 status 合并位置或全 batch 无 flush 的受控 bypass。任何精度、图模式或设备健康失败，都停止对应后续性能测试，回传 source commit、`git status --short`、构建日志、首个失败 JSON/traceback、NPU/board 健康状态。

## Verify 与 compact fusion 的下一阶段设计判断

**原则上可行，但本次不实现。** 当前顺序必须是上一轮 accepted-token cursor 更新 → 本轮 verify → 本轮 compact；compact 只折叠旧 `[0, position)` 的 d/k/g 到 checkpoint，verify 的新候选写在 `[position, position+T)`，两段不能混淆。融合时同一 `(row, value head, V tile)` 必须先完成该 tile 的所有候选输出和新 d/k/g 写入，再用旧历史更新 checkpoint；不能让 checkpoint 的新值参与本轮 verify。共享 key head 的 candidate K 写入及 ring 环绕也要证明没有地址别名和跨 task 读写竞争。

v11 的 verify 已有 `(row, value head)` 分组路径，较容易在所有 V tile 完成后插入 compact；原始 tile 路径也可按 tile 融合。但 verify 的 UB buffer 已接近 tile128 的容量边界，不能直接叠加 compact 的 state/product buffer。需做 buffer 生命周期复用、分支资源报告和 tile64/128 分别编译。融合收益首先是省掉每层一次 compact launch 和重复 checkpoint/元数据搬运，不代表 flush 的历史折叠算术可消失。混合 flush、padding、图 replay、TP4/TP8、B1/32/64、不同接受数和跨尾场景全过后，再用完整 `vllm serve` 确认端到端收益。

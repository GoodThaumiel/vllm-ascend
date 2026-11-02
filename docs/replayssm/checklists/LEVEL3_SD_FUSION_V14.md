# L3.10 SD verify + compact 串行融合消融（27B / MTP3）

目标环境：Ascend910B4、CANN 9.2.0、torch_npu 2.10.0.post4、vLLM `0.30.0+empty`、本分支源码编译的 vllm-ascend。只测 Qwen3.5-27B 几何，TP4/TP8、本地 QK/V heads=4/12 或 2/6、MTP3、物理 ring=32、BF16 checkpoint、FP16 D/K ring、FP32 G ring。**本轮不切换 `vllm serve` 默认 dispatch**：服务仍用 verify v13/tile128 + compact v1/tile32；新 `replayssm_sd_verify_compact_fused_` 只在独立脚本中调用。

## 算法与资源边界

融合核沿用 v13 的候选输出、D/K/G 写入及 tile128 映射；全部候选写完后，若该请求 `flush=1` 且 `position>0`，在同一 `(request, value head)` 任务内按旧 `[0, position)` 顺序执行 `S = exp(g_i) * S + d_i ⊗ k_i`，最后将 BF16 checkpoint 写回。`flush=0` 不更新 checkpoint。新候选的环形位置从 `position` 开始，旧 history 与新候选在 32 槽 ring 中不重叠。仍分别输出 verify 与 compact status；只有 head0 连续写整行 compact status，避免各 head 的小块写入相互覆盖。

内核重用 verify 已加载的 FP32 checkpoint、history K/D/gate 和 product UB；没有第二份 state/product/history buffer。为分块写回 BF16 checkpoint，输出 queue 从 512B 增到 4096B；本实验的候选暂存仅需 12 个本地 value heads。按源码显式 `InitBuffer` 相加约 **186,976B**，距 192KiB UB 上限约 **9,632B**。CCE 可能还需要额外临时空间，能否编译以上机报告为准。这里是串行融合，不做 double buffer 或算术重排；flush 的折叠计算仍然存在。

## 1. 构建与本地门槛

拉取本次提交，在 CANN 9.2.0 容器重新源码编译主扩展。确认新 AscendC kernel `replayssm_sd_verify_compact_fused` 的编译、launch header、`.so` 加载和 Meta schema；保存 CCE 首条错误及 UB 资源报告。**编译失败即停，不用旧 `.so` 跑脚本。**

```bash
python3 -m pip install --no-build-isolation --no-deps -v -e .
python3 -m unittest tests.ut.replayssm.test_sd_fusion_reference
python3 tests/ut/replayssm/test_framework_meta.py
```

## 2. 精度与图回放

先 TP4/B2、后 TP8/B2。每项对同一输入分别执行 `v13 verify → v1 compact` 和融合核，并与 CPU 参考比对：输出、D/K/G ring、checkpoint、两组 status；覆盖 pos0、pos7、pos12、pos16、base31 跨尾、全 flush、全非 flush、混合 flush。graph 阶段在 capture 后恢复缓存并重放。`ring-layout=detached` 模拟当前服务中 checkpoint block id 与请求 ring id 分离；`tied` 作定位对照。

```bash
for tp in 4 8; do
  for layout in tied detached; do
    python3 scripts/replayssm/run_sd_fusion_ablation.py \
      --tp "$tp" --batch 2 --ring-layout "$layout" --stage smoke \
      --cores 40 --device 0 \
      --output "/tmp/l310-fusion-${tp}-${layout}-smoke.json" || exit 1
    python3 scripts/replayssm/run_sd_fusion_ablation.py \
      --tp "$tp" --batch 2 --ring-layout "$layout" --stage graph \
      --cores 40 --device 0 \
      --output "/tmp/l310-fusion-${tp}-${layout}-graph.json" || exit 1
  done
done
```

任何编译错误、非零 status、数值超差、NPU error 或图捕获失败，停在该组合并回传首个失败 JSON、日志及 NPU 健康状态。特别检查 flush 后 checkpoint 是否与串行路径匹配、非 flush checkpoint 是否保持不变，以及新候选尾部没有被折叠。

## 3. 时延消融

精度全过后，在相同容器与相同设备上测 graph replay p50/p90；计时区外恢复缓存。优先 TP4/B32、B64，随后 TP8/B32、B64。脚本内的 `serial` 是两个 kernel 的同一张图，`fused` 是一个 kernel 的图；两者都含相同输入布局和 status 写入。不要把 eager Event 数字与 graph replay 直接比较。

```bash
for tp in 4 8; do
  for batch in 32 64; do
    python3 scripts/replayssm/run_sd_fusion_ablation.py \
      --tp "$tp" --batch "$batch" --ring-layout detached --stage perf \
      --cores 40 --iterations 100 --device 0 \
      --output "/tmp/l310-fusion-tp${tp}-b${batch}-perf.json" || exit 1
  done
done
```

用 `msprof` 对 TP4/B32 与 B64、pos7/12/16、flush=0/1 记录三个任务名的 task duration 与 AIV Scalar/Vector/MTE2/MTE3，核对融合核是否减少重复 checkpoint/history 搬运、是否因 4KiB 输出 queue 或 8 段 checkpoint 写回带来回退。统计真实服务的 flush 请求行比例后，按实际分布加权比较串行与融合成本；不要假定每 16 轮恰好 flush 一次。

本轮只据独立 kernel/graph 结果决定是否继续接入服务。若融合在非 flush 明显胜出而 flush 回退，先定位折叠段与 checkpoint 写回；若 UB 编译失败，优先缩小输出 queue 或复用已结束生命周期的候选暂存，不改原有 v13+v1 对照。

回传 source commit、`git status --short`、构建日志与 UB 报告、全部 JSON、msprof 摘要、首个失败 traceback 和 NPU 健康状态。

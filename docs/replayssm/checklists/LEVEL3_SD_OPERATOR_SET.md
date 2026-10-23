# L3.3：SD 单算子补齐与小范围对照

目标环境仍是 vLLM `0.30.0+empty`、本分支 vllm-ascend 0.30、CANN 9.2.0、torch_npu 2.10.0.post4、Ascend910B4，TP4/TP8。保持 BF16 checkpoint/Mamba state；Replay d/k ring 为 FP16、g ring 为 FP32。当前提交**只补单算子和框架 op 注册**，不开放 MTP 的 `vllm serve` dispatch。每一步失败即停止后续依赖测试并回传首个失败日志/JSON。

## 1. 独立算子构建与精度

先按 [compact 独立清单](../../../experimental/replayssm_sd_compact_ascendc/README.md) 和 [cursor 独立清单](../../../experimental/replayssm_sd_cursor_ascendc/README.md) 各自构建，在 CANN 9.2.0/910B4 上确认 `readelf -d` 显示包含 `$ORIGIN/lib` 的 RUNPATH。运行两者的 CPU 测试、TP4 smoke→graph、TP8 graph；再运行 B1/B32/B64 的 graph perf，记录 p50/p90、核数及错误码。`compact` 重点覆盖旧 history=16、本轮 T4、base31 环形跨尾和候选不被折叠；`cursor` 重点覆盖 accepted=0/1/4、flush 后 base 移动、reset、padding、拒绝尾部。

本轮还把 SD verify 的允许窗口从 `position+T≤16` 扩为 `position≤16` 且 `position+T≤20`、不超过物理 ring。L3.1 的 18 组性能报告**未覆盖**新边界。请用框架 op 或独立 verify runner 补测 `position=16,T=4,ring=32,base=31`，并核对输出、d/k/g 和状态码。

## 2. 主扩展重新构建与注册

```bash
python3 -m pip install --no-build-isolation --no-deps -v -e .
python3 -m unittest tests.ut.replayssm.test_framework_meta
python3 experimental/replayssm_sd_ascendc/run.py --backend framework \
  --stage graph --tp 4 --batch 2 --cores 40 --device 0 \
  --output /tmp/replayssm-sd-framework-graph-tp4.json
python3 experimental/replayssm_sd_compact_ascendc/run.py --backend framework \
  --stage graph --tp 4 --batch 2 --cores 40 --device 0 \
  --output /tmp/replayssm-sd-compact-framework-graph-tp4.json
python3 experimental/replayssm_sd_cursor_ascendc/run.py --backend framework \
  --stage graph --batch 32 --device 0 \
  --output /tmp/replayssm-sd-cursor-framework-graph-b32.json
```

- [ ] 构建日志包含三个 AscendC kernel：`replayssm_sd_verify`、`replayssm_sd_compact`、`replayssm_sd_cursor`；无缺失 launch header 或运行时找不到 `.so`。
- [ ] `_C_ascend` 三个 schema、Meta 和 NPU op 都可加载；独立和主扩展的结果一致。
- [ ] 对 compact/cursor 在主扩展路径分别完成最小调用及图捕获；TP4 全过后抽查 TP8。如独立图通过而主扩展失败，回传主扩展构建日志和第一个错误栈。

## 3. 小范围 native/Replayssm 对照

独立 op 全过后，先运行 2 次的小冒烟，再扩展到 20 次。脚本每次把 native state、Replay checkpoint/ring 和 cursor 恢复到同一初始状态；恢复拷贝在计时区间之外。Replay 每次执行 `verify→compact→cursor`，非 flush 也执行一次跳过主体的 compact launch。首轮会比较 native/Replay 输出、flush 后 BF16 checkpoint、cursor 新位置与基址。`cursor` 在一个 batch step 只调用一次，**不能按 48 个 GDN 层重复计费**。

```bash
python3 scripts/replayssm/profile_sd_lifecycle_compare.py \
  --tp 4 --device 0 --batches 1 --positions 0 16 --length 4 \
  --accepted 1 --repetitions 2 --warmup 0 \
  --output /tmp/replayssm-sd-lifecycle-smoke-tp4.json

python3 scripts/replayssm/profile_sd_lifecycle_compare.py \
  --tp 4 --device 0 --batches 1 16 32 --positions 0 12 16 --length 4 \
  --accepted 1 --cores 40 --repetitions 20 \
  --output /tmp/replayssm-sd-lifecycle-tp4.json
```

TP4 通过后再将 `--tp` 改为 8，使用新的报告路径；另用 `--accepted 4` 检查全接收。请按 `REPLAYSSM_SD_LIFECYCLE/<case>/<op>` 标签抓 profiler，分别给出 native recurrent、Replay verify、compact、cursor 的设备时延及 HBM/UB 流量。报告同时给出 `verify+compact` 单层和 cursor 一次/batch 的合计，分别看 `position=0/12/16`；不要把脚本的解析性 48 层求和当作实测模型时延。

脚本目前是 **eager NPU Event** 小范围计时，可能包含 host dispatch 空隙；L3.1 数据是 graph replay，不能直接并列比较。独立 runner 的 graph perf 用来确认每个新算子图回放行为；完整 `vllm serve` 的吞吐、显存/waiting 与 MTP 精度要等 packed metadata、cache allocator 和 dispatch 接线后再测。

回传：源码 commit、`git status --short`、构建日志、所有 JSON、首个失败 traceback、NPU 健康状态，以及 TP4/TP8 对照 profiler 摘要。若 cursor B32/64 明显占比，下一轮优先将单核逐标量 UB 搬运改为批量/分块方案。

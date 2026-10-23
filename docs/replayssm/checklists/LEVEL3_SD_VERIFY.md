# L3.1：ReplaySSM SD 独立非 flush verify 上机验证

本提交只新增 `experimental/replayssm_sd_ascendc/`，没有修改 serving dispatch、native MTP 或 AR 算子。上机先按 [独立目录说明](../../../experimental/replayssm_sd_ascendc/README.md) 在 CANN 9.2.0 容器编译。源码为 vLLM `0.30.0+empty`、vLLM Ascend `releases/v0.30.0`，Qwen3.6-27B 的 TP4/TP8 本地 head 几何，BF16 checkpoint、FP16 d/k 和 FP32 g。

先只跑 TP4、batch 2、40 vector core；每个命令单独执行，失败时停止本级：

```bash
timeout 600 python3 experimental/replayssm_sd_ascendc/run.py \
  --stage smoke --tp 4 --batch 2 --cores 40 --device 0 \
  --output /tmp/replayssm-sd-l31-smoke-tp4.json
timeout 600 python3 experimental/replayssm_sd_ascendc/run.py \
  --stage graph --tp 4 --batch 2 --cores 40 --device 0 \
  --output /tmp/replayssm-sd-l31-graph-tp4.json
```

- [ ] 两个 JSON 的 `status` 均为 `PASSED`，加载的 `.so` hash、源码 commit、包版本和 CANN 路径已保存。
- [ ] `T=1`、`T=4`、环形跨尾、`position=12` 非 flush 边界、混合长度及 padding 行均与 CPU golden 通过。
- [ ] checkpoint 在所有验证 case 中不变；仅候选 d/k/g 区域被写入；毒化的历史尾部未读。
- [ ] NPUGraph capture/replay 通过，图重放时更换 position/base/length 仍正确。
- [ ] TP4 跑通后，以相同步骤复验 TP8；任一失败回传首个失败 JSON、build log 和设备日志，不继续跑性能矩阵。

正确性通过后，使用目录 README 的 `perf` 矩阵比较 TP4/TP8、batch 1/32/64、core 8/20/40，重点回传 `T=4`、history 7/12 与 `T=1` 的图内 p50/p90。当前脚本的计时代表 **一个独立 verify 核**，没有 flush、cursor commit/reset、MTP drafter、48 层模型或 HTTP 服务；它不能给出最终 SD 加速比。若 40 core 继续明显优于 8 core，下一轮 tiling 以 40 为默认；若 `T=4` 延迟随 batch 恶化，优先并行化小矩阵点积/三角修正，而不是优化 AR1。

本级通过后才开始 L3.2 flush 与 L3.3 accepted-token cursor；SD serving 接入留到两者各自验证通过以后。构建/运行依赖用户的 910B4 服务器，开发机只能完成 CPU 数学与 Python 静态检查。

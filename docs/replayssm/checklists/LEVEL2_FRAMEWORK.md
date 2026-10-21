# L2.2：框架打包与 torch binding 验证

## 本轮变更与门槛

用户已确认 `3ff971b` 加服务器 CMake 修复后的独立六组测试通过。
随后回传的 B=2、tile_v=32、cores=8 精确报告确认 TP4/TP8 的 graph 正确性通过；
匹配空历史 flush Event p50 分别为 Replay 125.61/140.73µs、Native 204.85/224.67µs。
这些是独立 namespace 的 eager 调用 Event，不替代本清单的框架构建与48层图内计时。
本轮接受这份反馈，开始打包验证，不将其扩展为整网 Replay 或 SD 验收。
项目仍停在 Level 2，`next_level_allowed=false`。

- 已验证 kernel 移到 `csrc/kernels/replayssm_ar.cpp`，计算源码保持逐字一致。
- host launcher 移到 `csrc/replayssm_ar.cpp`，独立与框架构建共用，不复制两套计算实现。
- 910B 构建注册 `_C_ascend.replayssm_ar_step_`，带完整 mutation schema 和 Meta 实现。
- Python 入口为 `vllm_ascend.ops.replayssm.ascendc.replayssm_ar_step`。
- 按用户修复保留 CANN 的 `build/lib` 布局，独立扩展 RPATH 增加 `$ORIGIN/lib`。
- 本轮改动构建与算子注册，侵入性低；未修改模型 dispatch、scheduler、KV allocator、MTP。

## F1：重新构建并加载框架算子（本轮先做）

沿用已通过的服务器 CANN 9.2.0 环境、构建依赖、SoC 和版本号策略，不升级任何基础组件。
必须重新编译 C++/AscendC 并安装，单纯 `git pull` 或 editable Python 生效不能证明扩展更新。
可沿用已有成功的完整构建命令；常见 editable 构建入口如下：

```bash
python3 -m pip install --no-build-isolation --no-deps -v -e .
python3 experimental/replayssm_ar_ascendc/run.py \
  --backend framework --stage smoke --tp 4 --batch 1 --capacity 16 --cores 8 \
  --devices 0 --output /tmp/replayssm-framework-f1-tp4-r1
python3 experimental/replayssm_ar_ascendc/run.py \
  --backend framework --stage smoke --tp 8 --batch 1 --capacity 16 --cores 8 \
  --devices 1 --output /tmp/replayssm-framework-f1-tp8-r1
```

- [ ] 不需要独立 lab `.so`；确实通过已安装的 `vllm_ascend_C` 调用新算子。
- [ ] `replayssm_ar_step_` 存在，未出现 missing symbol、RPATH、Meta 或 dispatch 错误。
- [ ] 报告中的 native 扩展路径/hash 与重编译产物一致。
- [ ] 输出、缓存、checkpoint 与原 CPU golden 对齐；设备无异常。
- [ ] 回传构建日志和 F1 两个 JSON/log 目录；确认后再做 F2。

如果原构建配置关闭了 `COMPILE_CUSTOM_KERNELS`，需按原已验证方式开启；不要改变框架版本来规避。
构建框架会重新编译原生算子，必须确认原生服务仍能加载，再开始后续功能接入。

## F2：同一算子在框架 namespace 下回归

F1 确认后，依次以 `--backend framework` 重跑原清单中的 `replay`、`flush`、`lifecycle`、`graph_perf`。
TP4/TP8各至少一次。参数沿用原已通过的 capacity、tile 和 cores，避免同时改变算法或 tiling。
独立构建仍可用于对照；新的独立和框架 namespace 共用同一源码。

- [ ] 历史重建、flush、padding、slot 复用、非法 metadata 与原结果一致。
- [ ] NPUGraph replay 修改输入/cursor/flush 后结果正确。
- [ ] FakeTensor/Meta 调用只推导，不分配设备数据或执行 launcher。
- [ ] 原生 `vllm serve + HTTP` 一级 smoke 无回归；当前仍走原生 GDN，不宣称已启用 Replay。

## F3：确认图内时延与批量规模

F2 确认后，分别以 TP4/TP8、B=1/8/16运行；可在空闲卡上并行。
先保持已验证 cores/tile，不做性能调参。以下是一个 TP4/B=8 的完整命令：

```bash
python3 experimental/replayssm_ar_ascendc/run.py \
  --backend framework --stage graph_perf --tp 4 --batch 8 --capacity 16 \
  --cores 8 --tile-v 32 --devices 0 --iterations 50 --stack-layers 48 \
  --native-baseline --timeout 600 \
  --output /tmp/replayssm-framework-f3-tp4-b8-r1
```

- [ ] 保存 eager Event、单算子 graph replay、48层独立状态 graph replay 三类指标。
- [ ] 48层 benchmark 所有 layer 精度通过；没有重用同一份状态而低估工作集。
- [ ] 分别报告 B=1/8/16 的 p50/p90，不能将 B=2 的数字线性外推。
- [ ] 图内48层总耗时与 [调用预算](../CALL_BUDGET.md) 对比，再决定是否优化。

F1–F3完成后再进入 target AR 的 dispatch/缓存生命周期接入。
之后才是 Replay `vllm serve + HTTP`，再后续是 MTP/SD 的多 token verify 和提交回滚。

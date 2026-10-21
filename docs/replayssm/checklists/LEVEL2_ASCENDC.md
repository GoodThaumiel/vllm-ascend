# L2.1 独立 AscendC 上机验证清单

用户已回报 `3ff971b` 加 CMake/RPATH 修复后独立六组通过。此清单保留为验收记录；
本轮新的工作入口是 [L2.2 框架打包清单](LEVEL2_FRAMEWORK.md)。

本轮只接受独立算子验证。环境沿用 CANN 9.2.0、torch_npu 2.10.0.post4、vLLM `0.30.0+empty`、
源码编译 vLLM Ascend 0.30.0；BF16 SSM state 与 Mamba cache 基线不变。
第一轮只跑 **A1 smoke**。每段反馈结果、修正并确认通过后才运行下一段。
这是第二级内部五段，不改变整个项目的五级验收门槛。

## A0：独立构建

- [ ] 记录 `git rev-parse HEAD`、`git status --short`、CANN/torch/torch_npu 版本。
- [ ] 沿用现有 9.2.0 环境配置，确认 SoC 编译目标名称；这里默认 Ascend910B4。
- [ ] 运行下面的构建，不重新安装 vLLM 或 vLLM Ascend。
- [ ] 回传 configure/build 日志和 `build/CMakeCache.txt`。

```bash
set -o pipefail
cmake -S experimental/replayssm_ar_ascendc \
  -B experimental/replayssm_ar_ascendc/build \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.0 \
  -DSOC_VERSION=Ascend910B4 \
  -DPython3_EXECUTABLE="$(command -v python3)" \
  2>&1 | tee /tmp/replayssm-ascendc-configure.log
cmake --build experimental/replayssm_ar_ascendc/build --parallel 4 \
  2>&1 | tee /tmp/replayssm-ascendc-build.log
```

产物：`build/libreplayssm_ar_kernel.so` 与 `build/libreplayssm_ar_torch.so`。
如果编译失败，停在 A0；不通过修改版本号或切回 Triton 来规避新的错误。
现有基线版本规避策略继续按 [BASELINE](../BASELINE.md) 执行。

## A1：单核、空历史 smoke（本轮运行到这里）

```bash
python3 experimental/replayssm_ar_ascendc/run.py \
  --stage smoke --tp 4 --batch 1 --capacity 16 --tile-v 32 --cores 1 \
  --devices 0 --output /tmp/replayssm-ascendc-a1-tp4-r1
python3 experimental/replayssm_ar_ascendc/run.py \
  --stage smoke --tp 8 --batch 1 --capacity 16 --tile-v 32 --cores 1 \
  --devices 1 --output /tmp/replayssm-ascendc-a1-tp8-r1
```

两条命令可在不同卡上并行；首次不建议占满八卡。

- [ ] `.so` 成功加载，算子在 `torch.ops.replayssm_lab.ar_step_` 注册。
- [ ] 同步正常返回，没有 507014、trap 或非法地址。
- [ ] BF16 输出与 FP64 CPU golden 对齐；输出 relative L2 error ≤2%。
- [ ] FP16 d/k 和 FP32 g 写入当前位置；checkpoint、其他位置和 slot 原值不变。
- [ ] `status` 所有 task 均为 0，未使用的缓存尾部 NaN 未污染输出。
- [ ] 回传两个 output 目录全部 JSON/log，以及测试后卡健康与进程残留情况。

脚本默认 180s host 超时。如果存在新的硬件异常，确认设备已恢复健康后才换卡/重跑。
父进程结束子进程只用于隔离测试，不代替设备复位。

## A2：历史重放与多核（A1 确认后）

```bash
python3 experimental/replayssm_ar_ascendc/run.py \
  --stage replay --tp 4 --batch 2 --capacity 16 --cores 8 --devices 0,1 \
  --output /tmp/replayssm-ascendc-a2-tp4-l16-r1
python3 experimental/replayssm_ar_ascendc/run.py \
  --stage replay --tp 8 --batch 2 --capacity 64 --tile-v 64 --cores 8 --devices 2,3 \
  --output /tmp/replayssm-ascendc-a2-tp8-l64-r1
```

- [ ] 覆盖 position 0/1/7/L÷2/L−1，FP16 历史恢复结果对齐。
- [ ] 覆盖 L=16/64、Vtile=32/64，确认不同核之间无竞争或死锁。
- [ ] 回传所有 case 的 out/cache 误差、UB 预算及首个失败 case。

## A3：flush 和 padding（A2 确认后）

```bash
python3 experimental/replayssm_ar_ascendc/run.py \
  --stage flush --tp 4 --batch 2 --capacity 16 --cores 8 --devices 0 \
  --output /tmp/replayssm-ascendc-a3-tp4-r1
python3 experimental/replayssm_ar_ascendc/run.py \
  --stage flush --tp 8 --batch 2 --capacity 1 --cores 8 --devices 1 \
  --output /tmp/replayssm-ascendc-a3-tp8-l1-r1
```

- [ ] 无历史/半窗口/满窗口 flush 后 BF16 checkpoint 对齐，历史缓存保持原值。
- [ ] slot 0 输出严格为零，checkpoint/历史不写入；L=1 同样覆盖。
- [ ] 未使用 slot、padding、历史尾部无越界写。

## A4：多步生命周期与非法 metadata（A3 确认后）

```bash
python3 experimental/replayssm_ar_ascendc/run.py \
  --stage lifecycle --tp 4 --batch 2 --capacity 16 --cores 8 --devices 0 \
  --output /tmp/replayssm-ascendc-a4-tp4-r1
python3 experimental/replayssm_ar_ascendc/run.py \
  --stage lifecycle --tp 8 --batch 2 --capacity 16 --cores 8 --devices 1 \
  --output /tmp/replayssm-ascendc-a4-tp8-r1
```

- [ ] 连续 `2L+3` 步，覆盖提前 flush、满窗口 flush、重置 cursor 和 row 重排。
- [ ] 模拟新请求复用 slot，只清零 checkpoint/cursor，旧历史不清空，确认不误读旧请求状态。
- [ ] 每一轮设备保留自己的缓存，不用 CPU golden 覆盖设备状态掩盖累计误差。
- [ ] 非法 slot/cursor/flush 返回约定状态码，没有缓存写入或设备异常。
- [ ] 所有正 slot 必须互异；测试工具拒绝重复 slot。

## A5：图模式与性能（A4 确认后）

```bash
python3 experimental/replayssm_ar_ascendc/run.py \
  --stage graph_perf --tp 4 --batch 2 --capacity 16 --cores 8 --devices 0 \
  --native-baseline --iterations 50 \
  --output /tmp/replayssm-ascendc-a5-tp4-r1
python3 experimental/replayssm_ar_ascendc/run.py \
  --stage graph_perf --tp 8 --batch 2 --capacity 16 --cores 8 --devices 1 \
  --native-baseline --iterations 50 \
  --output /tmp/replayssm-ascendc-a5-tp8-r1
```

- [ ] 非默认 stream 正常；NPUGraph 捕获后更换输入、cursor 和 flush，输出随新输入变化且正确。
- [ ] 相同静态地址下反复切换空/半/满历史和 flush，checkpoint/d/k/g 正确。
- [ ] 记录 Event p50/p90/min，状态恢复与编译/首次运行不计入稳态计时。
- [ ] 原生 fla_npu 与 Replay 空历史 flush 的输入、BF16 state、输出语义一致并均通过精度检查。
- [ ] 性能目前只收集，无加速比通过门槛；预处理、scheduler、通信与 HTTP 不在算子计时内。
- [ ] 本阶段通过后用户确认，再规划 `csrc` torch binding 集成和 `vllm serve + HTTP` 验证。

以上 TP4/TP8 是本地 head 几何；分布式 TP 通信及 MTP 验证属于后续框架接入阶段。
这一轮不会将项目 `next_level_allowed` 改为 true。

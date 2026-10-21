# ReplaySSM AR AscendC standalone lab

本目录是 L2.1 独立验证工程。构建、加载、测试都不依赖 vLLM 的 Python import、安装流程或服务启动。
目标环境为 Ascend910B4、CANN 9.2.0、torch/torch_npu 2.10；vLLM `0.30.0+empty` 与源码编译的
vLLM Ascend 0.30.0 基线不变。独立六组测试已由用户确认通过；当前已接入 csrc 构建与 binding，待重编译验证，尚未接入 serving。

## 本次迭代边界

- 独立 CMake 构建 AscendC `.so` 和一个 C++ torch binding `.so`。
- Python 使用 `torch.ops.load_library` 注册 `torch.ops.replayssm_lab.ar_step_`。
- kernel 仿照 fla_npu 的队列、`[Vtile,K]` 状态分片、向量乘和 K 归约组织；
  构建与 Python 调用测试的分离方式参考 GVirt-xlite。
- 原生 AR 的单步递推增加 Replay 历史恢复和按 flush 写回；历史恢复采用 FP32 向量外积累加。
- 框架打包阶段修改根 CMake 和 torch binding 注册；没有修改 vLLM、模型 dispatch、scheduler 或 cache allocator。
- 没有把未验证的 Triton kernel 注册成 fallback；Triton 路径保留作历史诊断。
- 独立测试已获用户硬件通过反馈；迁移后的框架构建/加载仍需另行验证，不能用 CPU 测试代替。

## 目录与参考

```text
replayssm_ar_ascendc/
  CMakeLists.txt           # AscendC kernel + torch library 独立构建
  csrc/torch_binding.cpp  # 独立 namespace；共用根 csrc host/schema/Meta
  ../../csrc/kernels/replayssm_ar.cpp # 共用且保持已验证内容的 kernel
  ../../csrc/replayssm_ar.cpp # 共用 host launcher
  lab.py                  # FP64 CPU golden、poison 数据、完整写入边界检查
  run.py                  # 隔离子进程、分阶段上机、事件计时和 JSON 报告
  tests/test_reference.py # 可在 CPU 上运行的 golden/契约/测试工具验证
```

参考来源：

- `flashserve/flash-linear-attention-npu`，commit
  `47d7c97ce439228e0ead6fab4dad8b0440b663a8`，
  `fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/`。
- 用户提供的 `GVirt-master/xlite/CMakeLists.txt`、`csrc/_C.cpp`、
  `csrc/kernels/recurrent_gated_delta_rule.h`、`tests/kernels/recurrent_gated_delta_rule.py`。
- 用户提供的原论文集成 `ReplaySSM-main/vllm/third_party/flash_linear_attention/ops/fused_recurrent_replayssm.py`。

这是独立编写的窄接口实现，没有复制 fla_npu 或 GVirt 的 kernel、宏库、host tiling 和运行时。
它复用了原生实现的计算组织思路与标准 AscendC API；不是已经验证过的原生算子的直接拷贝。

## 接口与数值语义

| 输入/输出 | shape | dtype |
|---|---|---|
| q, k，已归一化 | `[B,H,128]` | BF16 |
| v, out | `[B,HV,128]` | BF16 |
| g，log decay；beta，已经 sigmoid | `[B,HV]` | FP32 |
| checkpoint，原位读写 | `[slots,HV,128,128]`，末两维为 V/K | BF16 |
| d_cache，原位读写 | `[slots,HV,L,128]` | FP16 |
| k_cache，原位读写 | `[slots,H,L,128]` | FP16 |
| g_cache，原位读写 | `[slots,HV,L]` | FP32 |
| slots, positions, flush | `[B]` | INT32 |
| status，原位输出 | `[B*HV*(128/tile_v)]` | INT32 |

所有张量必须连续、在同一卡上，任何写入张量不得与其他参数共享 storage。
`HV % H == 0`，`1 <= L <= 64`。同一 launch 的正 slot 必须互异；slot 0 为 padding。
CPU 测试工具检查 slot 唯一性，binding 不从设备读回 metadata；调用方负责该前提。
flush/cursor 由设备张量控制，不在 host 上用 `.item()` 决策。

数学流程是：

```text
S = BF16_checkpoint promoted to FP32
for each committed history entry i:
    S = exp(g_i) * S + outer(FP32(d_i), FP32(k_i))
S_decay = exp(g) * S
d = beta * (v - S_decay @ k)
S_new = S_decay + outer(d, k)
out = BF16(S_new @ (q * scale))
```

非 flush 只追加 FP16 d/k 和 FP32 g；flush 只写回 BF16 checkpoint，历史原值不变。
调用方在 flush 后重置 position。逻辑位置 `L-1` 必须 flush；可提前 flush。
本 kernel 不修改位置，允许图 replay 时替换 metadata。padding 输出全零，缓存保持原值。
status 每个 task 独立写：0 成功/合法 padding，1 slot 越界，2 cursor 越界，
3 flush 非法/满窗口未 flush，4 padding 携带历史。非法行输出零，跳过所有缓存读写。

与之前 Triton 候选的区别：q/k norm、g/beta 预处理移到算子外；q/k 按原生路径存为 BF16；
不再将 decay-scaled d 中间量额外量化为 FP16，也不使用 `tl.dot`。
历史恢复与论文公式在实数上等价，但累加顺序和舍入不同，不能声称与 CUDA 内核逐 bit 相同。
CPU golden 用 FP64 闭式求和，避免照抄设备的顺序外积循环；独立 CPU 测试比较两种算法。

## 多核与 UB

task 为 `(row, value_head, Vtile)`；按 `task = blockIdx + n*blockNum` 分配，无跨核同步。
每个 task 持有完整 K=128、V=16/32/64 的 FP32 状态 tile，历史一次批量搬入。
同一 key head 的 k_cache 只由第一个 value head、Vtile 0 写；g_cache 只由 Vtile 0 写。
只读 `[0,position)` 历史，当前追加位置与其他 task 读取的历史无重叠。

| buffer | 分配字节，分别向上取整到 32B |
|---|---|
| 共用输入 queue | `2*K*max(L,Vtile)` |
| 共用输出 queue | `2*Vtile*K` |
| FP32 state / product | 各 `4*Vtile*K` |
| FP32 history k / d | `4*L*K` / `4*L*Vtile` |
| history g | `4*64` |
| q/k/v/delta/output/scalar scratch | `4*(2*K+3*Vtile+8)` |
| reduction scratch | 4096 |

L=64 时，Vtile=32 合计 104096B，Vtile=64 合计 153632B，均低于 192KiB。
第一轮固定 `tile_v=32, cores=1` 缩小执行范围；通过后再调 tile 与核数。
这是正确性起点，不是最终性能 tiling。事件计时才用于判断后续优化优先级。
预算是显式申请的 UB，剩余空间保留给 CANN 高阶 API/编译器；实机编译资源报告仍需确认。

## 独立编译与本地测试

在服务器已激活的 CANN 9.2.0 环境运行，SoC 名必须以服务器可用的编译目标为准：

```bash
cmake -S experimental/replayssm_ar_ascendc \
  -B experimental/replayssm_ar_ascendc/build \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.0 \
  -DSOC_VERSION=Ascend910B4 \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build experimental/replayssm_ar_ascendc/build --parallel 4
```

独立模式不需要安装或重新编译 vLLM Ascend；框架模式必须重编译安装扩展。
独立输出 `.so` 保留在 build 目录，加载时使用当前 torch_npu stream。
需要 CMake >=3.20、C++17 编译器、CANN 编译工具、已安装且配套的 torch/torch_npu。
CPU 开发机可运行以下测试；其不加载 `.so`、torch_npu 或 vLLM：

```bash
python3 -m unittest discover -s experimental/replayssm_ar_ascendc/tests -v
```

## 上机验证顺序

历史清单见 [独立 AscendC 检查单](../../docs/replayssm/checklists/LEVEL2_ASCENDC.md)，
当前下一步见 [框架打包清单](../../docs/replayssm/checklists/LEVEL2_FRAMEWORK.md)。
默认只运行 smoke，阶段之间不自动升级；使用新的 output 目录保留每轮证据。
先反馈 smoke 和编译结果，修正并确认通过后再进入 replay 阶段。

```bash
python3 experimental/replayssm_ar_ascendc/run.py \
  --stage smoke --tp 4 --batch 1 --capacity 16 --cores 1 --devices 0 \
  --output /tmp/replayssm-ascendc-smoke-tp4-r1
```

每张卡单独子进程，`--devices 0,1` 可并行测不同卡。这里 TP 仅指本地 head 几何，不是 HCCL TP 测试。
父进程默认 180s 超时，结束 host 进程不代表复位硬件；如再次触发设备异常，先确认卡健康再继续。
报告包含首个失败 case、已完成 case、源码 commit/dirty status、扩展 hash、CMake 参数、版本、UB 预算。
JSON 的 `next_level_allowed` 始终 false，整级接受需要用户明确确认。

graph_perf 阶段验证当前 stream、NPUGraph capture、改变输入/cursor/flush 后 replay，
然后用 NPU Event 记录微秒级延迟。每次测量前恢复状态，恢复操作不在计时区间内。
`--native-baseline` 额外测已安装 fla_npu 普通 AR 与空历史 flush Replay 的同输入对照；
历史重放路径仅报告原始延迟，不自动推导端到端加速比。测量不包含 norm/gating 预处理、HTTP 和调度。
当前 Replay 计时包含诊断 status 写入，原生算子无该写入；最终性能优化时需单独评估这项开销。

## 验证后合入路径

kernel/host launcher 已迁入根 `csrc/`，独立工程引用同一份源码。框架入口使用
`--backend framework`，通过现有扩展 `_C_ascend.replayssm_ar_step_` 调用，
Python 侧由 vllm_ascend 调用注册算子。之后再补模型 dispatch/cache 生命周期和 `vllm serve + HTTP` 验证。
该阶段仍属于 L2；独立算子测试不能替代后续 TP 通信、MTP 或服务验证。

`--stack-layers 48` 在 graph_perf 阶段增加48套独立状态的图内调用计时；仅用于调用预算，
不模拟完整模型。`tests/meta/` 可只构建实际共享 schema/Meta 供 CPU 编译和 FakeTensor 测试。

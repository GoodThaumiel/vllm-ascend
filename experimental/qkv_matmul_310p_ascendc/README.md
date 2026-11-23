# Ascend310P3 QKV FP16 MatMul

独立工程，用于复现 0.30.0 prefill/mixed batch 的 QKV 投影并和原生路径做单算子对比。算子计算 `Y = X @ W.T`，无 bias，固定 `K=2048`、`N=6144`，支持 `1 <= M <= 4096`。

| 张量 | 逻辑形状 | 物理形状 |
| --- | --- | --- |
| X | `[M, 2048]` | NZ `[128, ceil(M / 16), 16, 16]` |
| W | `[6144, 2048]` | NZ `[128, 384, 16, 16]` |
| Y | `[M, 6144]` | ND `[M, 6144]` |

Torch binding 可接收带原生 FRACTAL_NZ descriptor 的矩阵，也可接收按上述字节顺序打包的连续 4D tensor。后者接受 torch_npu 为普通 4D 张量使用的 ND 或 NCHW 基础格式标签。动态尾块按 16 行向上对齐读入和计算，只写回逻辑有效行。

每个 Cube 核负责一组不相交的 128 列，权重 panel 在 L1 中复用。每个输出 tile 以 256×128×128 的 M/N/K 分块运行 16 次低阶 `Mmad`，首个 K 块清空 L0C，其余块累加。默认测试 FP32 L0C 累加，按 NZ 条带搬到 UB，再用 M200 已验证的 `CAST_NONE` 转成 FP16，随后按条带写出 ND。满 tile 时 FP32 路径使用 128 KiB L0C 和 192 KiB UB。初版 FP16 L0C 可用 `--accum-dtype fp16` 单独计时。写回方式参考 [Ascend C 耦合架构示例](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00007.html)。

## 上机清单

从 `vllm-ascend` 根目录执行，使用已加载 CANN 9.2.1 和 torch_npu 的 310P3 容器。只编译实验目录。

```bash
OP=experimental/qkv_matmul_310p_ascendc
cmake -S "$OP" -B "$OP/build-c921" \
  -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend310P3 \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.1 \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build "$OP/build-c921" --parallel
readelf -d "$OP/build-c921/libqkv_matmul_310p_torch.so" | grep -E 'NEEDED|RUNPATH'
ldd "$OP/build-c921/libqkv_matmul_310p_torch.so" | grep qkv_matmul_310p_kernel
```

先验证布局和精度。首段每行设置两个分属不同 K 块的非零元素，用精确输出核对 K 累加、NZ 方向、尾块和输出前后的哨兵行，分别覆盖 native NZ descriptor 和手工打包输入，同时记录两种输入的格式标签。第二段用相同随机输入比较 CPU FP32 reference、原生输出和自研输出。

```bash
python3 -m experimental.qkv_matmul_310p_ascendc.bench \
  --device 0 --stage check --output /tmp/qkv310p-check.json
```

精度通过后分别抓 profile。两边的权重都预先转为 NZ。原生默认从 ND 激活调用 `F.linear`，与当前分支的 unquantized linear 入口一致。脚本记录 `F.linear` 返回的格式（2 为 ND，29 为 NZ）。跨镜像对比时，分别记录 device MatMul 的输出格式和返回格式，并检查前后的 `TransData` 是否转换了激活或结果。自研 `custom` 在计时前转好激活 NZ，`custom-from-nd` 则把激活转换算入时延。若要给两边传完全相同的 NZ tensors，可另跑 `--impl both --native-input-format nz`。只有测原生的 `--impl native` 可在旧镜像直接用同一脚本，无需自研库。[TorchNPU 矩阵乘分派源码](https://github.com/Ascend/op-plugin/blob/master/op_plugin/ops/opapi/MmKernelNpuOpApi.cpp)

310P3 上用 `--no-timing` 跳过 NPU Event 计时，避免 `aclrtEventElapsedTime` 返回 `107006`。每组先提交 1 次输出检查和 10 次预热，再提交 100 次正式调用。JSON 记录配置、格式及调用次数，设备时延从 msprof 的逐次记录计算。

脚本默认用 `--jit-compile off` 对齐 310P 服务 worker 的 `set_compile_mode(jit_compile=False)`。若需复现旧脚本的分派，用 `--jit-compile inherit` 保留容器默认值。定位分派时在同一容器分别用两个新进程测 `--impl native --rows 4001`，保持其他参数一致，并核对输入格式、`OP Type` 和前后 `TransData`。默认的 ND 激活与 NZ 权重适合验证 `MatMulWeightNz`，改成 NZ 激活可能走另一条分派。

```bash
ROWS=4001
ACCUM=fp32
for impl in native custom custom-from-nd; do
  msprof --output="/tmp/qkv310p-prof-$ROWS-$impl-$ACCUM" \
    --application="python3 -m experimental.qkv_matmul_310p_ascendc.bench \
      --device 0 --stage perf --impl $impl --rows $ROWS --accum-dtype $ACCUM \
      --no-timing --warmup 10 --repeats 100 --output /tmp/qkv310p-$ROWS-$impl-$ACCUM.json"
done
```

可将 `ROWS` 改为 4096 检查对齐整块，也可用 `ACCUM=fp16` 重测原 FP16 自研路径作为性能对照。每个 profile 只放一个 M，便于在自定义 launch 缺少 shape 标注时归属样本。回传 perf JSON 和对应的逐次算子记录，重点比较 native MatMul、`qkv_matmul_310p_fp32_accum` 的设备耗时，以及从 ND 激活出发时前置 `TransData` 的开销。统计 p50/p90 时排除每条路径的前 11 次 MatMul。

## 服务级 prefill JIT 实验

在 **310P MRv1、FULL_DECODE_ONLY** 的原服务命令前加 `VLLM_ASCEND_310P_PREFILL_JIT=1`，或在启动前执行以下命令。沿用原来的模型、投机和图尺寸配置。此实验只改 Python，沿用现有自定义算子库。

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ASCEND_310P_PREFILL_JIT=1
```

仅在 target forward 同时满足有活跃请求尚在算 prompt、当前批次走图外、当前处于真实请求执行时打开 JIT。包含 prefill 的混合批也会命中。退出 target forward 时恢复 TorchNPU 模式，draft 和 decode 图继续使用原有路径。作用域覆盖该 forward 中所有 TorchNPU 算子。切入和恢复各有一次设备同步，按完整 prefill forward 和 TTFT 验收。

1. 用同一服务配置分别跑开关 `0` 和 `1`。启动后先做 C1，确认能完整生成，并记录第一次和预热后的 TTFT。
2. 用 C10 持续补入覆盖纯 prefill、混合批和纯 decode。确认出现 `310P target prefill experiment uses TorchNPU jit_compile=True`，并按原 [DFlash 双图清单](../dflash_310p_ascendc/LEVEL2.md) 核对两个 TP rank 的 target/draft replay。
3. 同口径抓一轮 `max_num_batched_tokens=4096` 的 prefill。核对目标投影、相关 `TransData` 和整个 forward 的时延，单列首次遇到 shape 的编译开销。回传前后端到端 TTFT、设备 trace 和首个错误栈。

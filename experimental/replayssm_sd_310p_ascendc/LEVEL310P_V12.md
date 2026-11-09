# ReplaySSM SD 310P V12：状态块消融与请求 ring demo

基线：`replayssm-310` V11 `c24e3dffb`，Ascend310P3、CANN 9.2.1、
torch_npu 2.10.0.post7。V11 的 18 组多核精度与 11 组性能测试已通过。
本轮仍是算子与资源生命周期 demo，**尚未打开 vllm serve 的 SD 分派**。

## 已确认的 310P 性能判断

- B32/MTP3/tile32/pos0：native V310 task p50 1954 µs，Replay verify4
  2820 µs，compact 26 µs；graph wall-clock combined/native 为 1.49。
- 核数 1→7 时 Replay combined 约 6.46 倍加速，优先保留 7 核。
- compact 已不是主要差距；910B 上优先优化 compact 的结论不直接迁移。
- native 的 scalar/vec 时间高度重叠，Replay 的 MTE2 绝对时间为
  383 µs（native 56 µs）。单元时长可重叠，不能把这些差值相加
  当成关键路径。本轮单独消融 32B status 读改写，判断它的贡献。
- V11 的 msprof `Block Num=8` 与设备资料中的 7 个 Vector Core 不是
  同一个指标，暂不据此认定 native 实际使用了 8 个 AIV。
- tile64 在 V11 只通过正确性，尚无性能结论；本轮补同口径对比。

| 910B 阶段结论 | 310P V11 证据 | 本轮决策 |
|---|---|---|
| 增加 AIV 核数收益大 | Replay B32 从 1→7 核加速 6.46× | 7 核作为性能起点 |
| compact 非 flush 固定成本值得优先压缩 | 310P compact 设备 task 约 26 µs，verify 约 2820 µs | 先优化 verify |
| 批量 history K/D 搬运有效 | 310P verify 已沿用批量 history 路径，但 MTE2 仍较高 | 消融 status RMW，不先重写 history |
| 大 V tile 通常有收益 | 310P tile64 仅验证精度，tile128 未做 UB 规划 | 先测 32/64 性能，再决定 128 |
| SD 显存换并发 | 310P MTP3/5 的 FP16 state 比例更高 | 先测独立 ring payload，serve 收益待验证 |

## A. 请求 ring 的框架资源 demo 与显存探针（可与算子编译同时跑）

`vllm_ascend/_310p/ops/replayssm_sd_demo.py` 复用现有
`ReplaySSMSDRingSlots`，输出请求对应的 ring id 和新请求 `do_reset`。
`scripts/replayssm/probe_310p_sd_memory.py` 可选择真实分配 D/K/G ring，
核对张量字节数。它只计算 per-rank FP16 SSM checkpoint 与
FP16 D/K、FP32 G ring 的 payload；不包含卷积、attention KV、权重、
算子工作区和 vLLM 内存预留。模型参数必须以部署 config 和 TP 分片为准。
选择 NPU 分配时还会记录 `npu_allocator_delta_bytes`，以便与纯张量
payload 对照；两者不同可能来自分配器对齐或缓存策略。
可传 `--model-config /path/to/config.json --tp 2` 自动读取
`layer_types` 与 linear key/value head 数；若模型配置结构不同，
请核实后显式填写下方三个几何参数。

```bash
python3 -m unittest tests.ut.replayssm.test_310p_sd_demo
python3 scripts/replayssm/probe_310p_sd_memory.py \
  --linear-layers 30 --local-key-heads 8 --local-value-heads 16 \
  --max-num-seqs 32 --mtp 3 --active-requests 32 --allocate-device cpu \
  > /tmp/replayssm-310p-v12-memory-mtp3-b32.json
python3 scripts/replayssm/probe_310p_sd_memory.py \
  --linear-layers 30 --local-key-heads 8 --local-value-heads 16 \
  --max-num-seqs 64 --mtp 5 --active-requests 64 --allocate-device npu:0 \
  > /tmp/replayssm-310p-v12-memory-mtp5-b64.json
```

验收：`allocation_matches_projection=true`，首次分配 `[1,2]` 的
`do_reset=[1,1]`，释放并复用后对应 `[2,1]` 和 `[0,1]`。
该 demo 是框架资源契约，不改变当前 MambaSpec 的实际占用。
`projected_saved_bytes` 是完成 serve 接入后的估算，不能当成已取得的收益。
按 30 个 GDN 层、每卡 8/16 个 key/value head、FP16 state 的示例参数，
容量 32 时 ring 约 187.6 MiB，满载 MTP3/MTP5 分别预计节省约
1.22/2.16 GiB 每卡；容量 64 时 ring 约 369.4 MiB，满载分别约
2.45/4.33 GiB。请求未满载时固定 ring 仍占用全部容量，探针会输出
该时刻的净节省，低并发时可能为负。

## B. 全新构建、状态块消融精度

`status_variant=0` 完全保留 V11；`status_variant=1` 把同一 row/head 组的
8 个 int32 状态码一次对齐写入 32B 块。每个状态块仅由一个 AIV 写，
valid、padding、非法 metadata 都必须与 V11 逐元素一致。

```bash
cmake -S experimental/replayssm_sd_310p_ascendc \
  -B experimental/replayssm_sd_310p_ascendc/build \
  -DSOC_VERSION=Ascend310P3 \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.1 \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build experimental/replayssm_sd_310p_ascendc/build --clean-first -j 8
```

编译失败即停，不能使用旧 `.so`。先只跑 variant1 的 MTP3/5 精度和
动态图捕获，再比较 tile32/64；任一失败时停在首个 case：

```bash
for mtp in 3 5; do
  for tile in 32 64; do
    python3 experimental/replayssm_sd_310p_ascendc/run.py \
      --mtp "$mtp" --batch 16 --stage smoke --tile-v "$tile" --cores 7 \
      --status-variant 1 --device 0 \
      --output "/tmp/replayssm-310p-v12-mtp${mtp}-t${tile}-smoke.json"
    python3 experimental/replayssm_sd_310p_ascendc/run.py \
      --mtp "$mtp" --batch 2 --stage graph --tile-v "$tile" --cores 7 \
      --status-variant 1 --device 0 \
      --output "/tmp/replayssm-310p-v12-mtp${mtp}-t${tile}-graph.json"
  done
done
```

## C. 性能消融（B 全过后）

原生 V310 符号在自定义 OPP 包内。如 V11 环境，先设置正确的 vendor 路径：

```bash
export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
for mtp in 3 5; do
  for tile in 32 64; do
    for variant in 0 1; do
      python3 experimental/replayssm_sd_310p_ascendc/run.py \
        --mtp "$mtp" --batch 32 --stage perf --tile-v "$tile" --cores 7 \
        --status-variant "$variant" --warmup 20 --iterations 100 --device 0 \
        --output "/tmp/replayssm-310p-v12-mtp${mtp}-t${tile}-v${variant}-perf.json"
    done
  done
done
```

记录每个 `pos/flush` 的 verify、compact、cursor、combined、native
p50/p90/min 和 `timing_source`。本机 Event 时间戳若返回 107006，
脚本会注明 `wall_clock` 并在每次 replay 前后同步。比较 variant
时需保持同一计时口径；建议另抓 B32/MTP3/pos0 的 msprof task、
scalar、vector、MTE2、MTE3 绝对时间，确认 32B 写回消融是否真有收益。
若 variant1 无明显收益，继续以 V11 variant0 为基线。

## 回传

请回传构建退出码、第一条 CCE/设备错误、A 阶段两个 JSON、B 阶段
8 个 JSON、C 阶段 8 个 JSON，以及 msprof CSV 路径。性能若出现
精度或图捕获失败则不作结论。310P 服务侧的下一步是按 V1 runner
实际覆写链接入 ring 分配、GDN verify/compact 与 cursor；`vllm serve`
显存和稳定性验收放在这一步完成。

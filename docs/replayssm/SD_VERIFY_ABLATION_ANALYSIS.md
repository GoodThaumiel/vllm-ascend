# SD verify 消融分析：Vector、Scalar 与流水

环境口径：用户回传的 `8c666c8` / Ascend910B4 / CANN 9.2.0，TP4、history=12、T=4、ring=32 的 100 次 `msprof` 均值。两条路径都在 AIV 上执行，但 native `aclnnRecurrentGatedDeltaRule` 更新完整 recurrent state；Replay verify 从 checkpoint 和 ring 历史直接计算多个候选输出并写入 ring。指令数量不能仅用算子名字推断，需按各自数学工作量比较。

## 已证实的差异

| 源码操作 | native RGDR | Replay SD v3/v4 | 影响 |
| --- | --- | --- | --- |
| K=128 点积 | 每 token 对整块 state 做 `MatVecMul`，行归约统一进行 | 每个 `(candidate token, history key)` 单独 `Dot`；每次 `Mul → Add fold → WholeReduceSum → V_S` | history=16、T=4 时，每 value head 的 144 次独立点积/Vector→Scalar 同步；该开销随 history 增长 |
| 向量更新 | `MulAddDst` 直接更新 state 的整块 tile；`Broadcast` 后成批运算 | 每个历史 D 行分别 `Muls`、`Add`，两个方向各做一次 | 小向量命令与 `PipeBarrier` 远多于 native；仍有进一步融合空间 |
| 输入搬运 | 一次 `DataCopyPad` 搬入窗口 Q/K/V，状态按 V tile 预取 | v3/v4 已批量搬 gate/beta；history K/D 仍逐行搬入；Q/K 每 token 搬入 | `msprof` 的 MTE2 占比 TP4/B64：custom 16.7%，native 5.4% |
| 输出搬运 | 输出与 state 使用独立队列，可配置双缓冲 | 输出、D/K/G/status 共用一个单槽输出队列 | MTE3 绝对时间也较长；是否阻断 Vector 需事件时间线确认 |

用户报告中 TP4/tile64/B64：native `task_duration=371.6 µs`，v4 `800.9 µs`；native AIV Vector/Scalar/MTE2 为 `321.0/128.6/18.9 µs`，v4 为 `459.4/306.8/127.7 µs`。Vector 多 138.4 µs，Scalar 多 178.2 µs，MTE2 多 108.8 µs。`vec/aiv_time` 的 0.91 与 0.60 是**占用比例**，不能单独当作重叠率；这些分量并行执行，须看同一核的事件时间线与空泡才能判断因果。v4 对比 v3 无收益，因此原来的晚预取并未形成可测量的有效重叠。

以最重的 history=16、T=4 计，v3/v4 每 value head 预计算 `2×16×4 + 2×(0+1+2+3) + 4 = 144` 个 K=128 标量点积；每个 V tile 还要处理 `2×16×4 + 2×(0+1+2+3) = 140` 次长度为 `tile_v` 的历史/候选 D 累加，每次包含 `Muls → Add`。这些是源码级调用次数，不代表 CCE 最终指令数。Native 的每个 token 主要执行三次整块 `MatVecMul`、两次行归约和若干整块更新；其计算包含完整 state 写回，数学工作量与 Replay verify 不完全相同。Replay 的主要额外成本在许多细粒度 K 点积和 D 累加，而非单次向量算术本身。

## 新的可独立测试版本

- v5 从 v3 出发，按一个 candidate token 批量计算 16 行 history K 点积：两段 64 元素 `Mul`、一次折叠、一次 `WholeReduceSum`。一组行只做一次 V→S 同步。候选 K 点积也按行批量处理，并把当前 K·Q 合入同一次归约。history=16、T=4 时，点积归约批次与 V→S 同步从 144 次降至 15 次；读取结果的 `GetValue` 仍约 144 次，所以 Scalar 时间未必同比下降。这是源码级 API 调用数，不是 CCE 指令数或实测加速比。FP32 加法顺序可能带来微小精度变化。
- v6 在 v5 上，把两槽 checkpoint 预取提前到点积表计算之前，让 MTE2 搬运有机会与 Vector 归约同时执行。与 v4 的“点积结束后才预取”相比，这是单独验证流水排布的版本。源码显式 UB 申请量 tile64 约 136.9 KB，仍小于用户提供的 192 KiB；编译资源报告是最终依据。

v5/v6 保持 v3 的任务次序：相邻 value head 会访问同一 GQA key head，每个任务依次处理自己的 V tile，可复用 UB 中的 K 点积表。此时盲目重排 `blockIdx` 可能破坏 Q/K 的 L2 局部性。先测 v5 是否降低 Vector/Scalar 时间，再看 v6 是否降低 MTE2 暴露时间；若二者仍远慢于 native，下一轮应优先处理 history D 的逐行搬运和两方向的小向量累加，而不是继续增加 checkpoint 预取槽数。

上机步骤见 [LEVEL3_SD_DOTROWS.md](checklists/LEVEL3_SD_DOTROWS.md)。

# ReplaySSM SD ring 与 GDN checkpoint 解耦评估

## 当前问题

当前 d/k/g ring 是 `MambaSpec.shapes` 的第 3–5 个分量，跟 conv/checkpoint 一起占用**每个可分配 GDN block**。TP4 的 Qwen GDN 每层每 block 的 ring 逻辑载荷为 d `12×32×128×2=98,304 B`、k `4×32×128×2=32,768 B`、g `12×32×4=1,536 B`，合计 **132,608 B（129.5 KiB）**。48 个 GDN 层时，每个 ring 槽约 **6.07 MiB**。上一轮相同 KV 预算下，native `num_blocks=1315`，当前 Replay `1051`；页变大导致可分配 block 数减少约 20%。混合 cache group 可 overlay，不能将这个逻辑载荷直接当作净 HBM 差额。

## 可行方案

把 GDN cache spec 收回到 conv/checkpoint 两个分量，仍保持 `num_speculative_blocks=0`；在 NPU runner 中单独分配 d/k/g ring。独立 ring 按**同时活跃的请求槽**分配，而不是按所有 `num_gpu_blocks` 分配。若保留与物理 block 一样多的 ring 槽，只改变分配位置，无法消除大部分总占用。

每个请求在 admission 时拿一个稳定 ring 槽，完成/抢占后释放；请求在 batch 内重排时 ring 槽保持不变。metadata builder 将 GDN block id 与 ring slot id 同时传给 verify/compact/cursor：checkpoint 用 block id，d/k/g 与 cursor 用 ring slot id。图捕获只读取固定大小的设备端映射/索引，不能依赖 host 端同步动态改变 grid。`mamba-cache-mode none` 下，抢占恢复走重新计算并重置 ring；其他 cache mode 需要单独设计。新 ring buffer 的创建、释放、跨层共享和睡眠唤醒均需纳入 runner 生命周期。

按 48 层估算，32/64 个活跃请求槽的 ring 逻辑载荷约为 **194/389 MiB**。这是 ring 自身容量估算，不是预计净显存节省或吞吐提升：分配器会把释放的预算变成更多 KV block，最终收益要看 full-attention 与 GDN 哪一组先限制可运行请求数。

## 实施门槛

1. 先完成 L3.7 数值定位与单物理 block 的 serve 稳定性；当前分叉不能带入新生命周期重构。
2. 增加 request-id→ring-slot 的分配/释放与抢占/重排测试；验证混合 batch、padding、取消请求、block 回收、图 replay 和 32/64 并发。
3. 独立 ring 的绑定需要改 `MambaSpec`、NPU runner 的 cache 分配/绑定、metadata builder 和三个 AscendC kernel 的 slot 参数语义，属于**高侵入式插件改动**；若上游内存管理只允许 cache spec 内的 tensor 生命周期，可能需要上游接口协助。
4. 用相同预算和长上下文请求对比可分配 block、实际 running/waiting、完成吞吐与 HBM；达到更高可运行并发且数值稳定，才能确认预期收益。

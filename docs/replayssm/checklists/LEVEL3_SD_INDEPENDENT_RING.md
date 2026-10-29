# L3.8 ReplaySSM SD 独立请求 ring 上机清单

目标环境仍是 Ascend910B4 / CANN 9.2.0 / torch_npu 2.10.0.post4 / vLLM `0.30.0+empty` / 本分支源码编译的 vLLM Ascend。TP4、MTP3、BF16 模型与 conv/checkpoint、FP16 d/k ring、FP32 g ring；verify v13/tile128、compact v1/tile32、cores40 不变。本轮只改变 ring 的物理分配和槽号。服务继续使用 `--additional-config '{"replayssm_sd":true}'`，不传 `--use-replayssm`。

## 1. 构建与算子隔离

1. 记录 `git rev-parse HEAD`、`git status --short`、CANN 路径和 package 版本。从当前 checkout **重新编译** vllm_ascend；确认扩展和 AscendC kernel `.so` 生成时间/路径来自本轮，不运行旧 `.so`。若编译失败，回传第一条 CCE 或 C++ 错误及其上下文。
2. 运行 `python3 -m unittest tests.ut.replayssm.test_model_ar_integration tests.ut.replayssm.test_framework_meta tests.ut.replayssm.test_sd_serve_validator`。Meta schema 应多一个可选的 `ring_slots`，旧的同槽独立算子调用仍能运行。
3. 单卡先运行：

   ```bash
   python3 scripts/replayssm/check_sd_independent_ring.py --tp 4 --device 0 --output /tmp/replayssm-sd-independent-ring-tp4.json
   ```

   有空卡时用 `--tp 8` 再跑一次。探针比较两条路径：checkpoint block id 为 1/2 与 11/12，而 ring id 都为 1/2；覆盖 `pos0`、`pos7`、`pos12`、`pos16`、跨 ring 尾和 compact flush。所有输出、d/k/g、checkpoint、状态码必须一致。任一项失败先停在算子层。

## 2. `vllm serve` 启动与内存

沿用 [L3.6 启动命令](LEVEL3_SD_SERVE.md)及 L3.7 已通过的容器参数。增加 `VLLM_ASCEND_REPLAYSSM_MEMORY_PROBE=1`。至少跑两个容量：

- `--max-num-seqs 32` 且 `--additional-config '{"replayssm_sd":true}'`：默认请求 ring 容量等于 32。
- `--max-num-seqs 64` 且 `--additional-config '{"replayssm_sd":true,"replayssm_sd_ring_capacity":64}'`：显式固定 64。容量不得小于 `max-num-seqs`，slot 0 额外保留给 dummy。

先看服务启动是否完成图 capture、`/health` 是否为 200。每 worker 日志应出现 `ReplaySSM SD independent request ring` 和 `REPLAYSSM_MEMORY_JSON.independent_sd_ring`，包括 `capacity`、`allocated_bytes_per_rank`、`ring_len=32`。`cache_plan` 中 GDN MambaSpec 应只剩 conv/SSM 两个 state 分量，`num_speculative_blocks=0`。记录 `num_gpu_blocks`、GDN page bytes、`planned_device_backing_bytes`、`torch_allocated_bytes`、`torch_reserved_bytes` 和 `npu-smi`。与同容器同参数的 `68498ac07` 旧 Replay、native 依次启动对比；三个服务不要同时占相同卡。

## 3. 请求生命周期与数值

1. 先 B1，再 B32，最后 B64；短 prompt 和 32k 输入 / 512 输出分别测试。使用 `scripts/replayssm/validate_sd_serve.py` 保存 HTTP JSON、worker 日志和 metrics。L3.7 已知 index 21 可能因当前 FP16 ring 的数值敏感出现 token 分叉，本轮验收重点是**相对于旧 Replay 的首分叉分布不明显提前**、重复请求自身确定、无 status/NPU 错误；不要把这一处 token 不同误报成 ring 地址错误。
2. 在 B32/B64 中混合长短请求、让部分请求完成后补入新请求，并主动取消部分请求。检查 ring id 重用后新请求是否从 position 0 开始；正在运行的请求在 batch 重排后 ring id 不变。触发至少一次 waiting → resume/preemption，确认连续生成不出现历史污染。若出现首个错误，回传 request id、首个 worker traceback、cursor/verify/compact status 和相邻两步的 `block_slot`/`ring_slot`。
3. 图捕获和实际 replay 都要经过新双槽 ABI。捕获成功但首个真实请求失败时，优先检查 CPU→NPU ring id 元数据、图 padding 的 slot 0、请求 release/reassign。测试后确认卡健康，无残留进程。

## 4. 并发收益验收

重点复现用户场景：TP4、ISL 约 32k、OSL512、MTP3、`max-num-seqs=64`、同一 GPU 内存利用率。旧测量为 native 预估/实际 **14.7/15**，旧 Replay **16.2/16**。新 Replay 记录预估有效并发、实际 running 峰值、waiting 峰值、总完成时间、请求 latency p50/p90、TPOT 和每 rank `num_gpu_blocks`。至少同条件重复两轮。应同时看 block 数与实际并发；固定 ring 的显存已经记入 profile，启动后总 HBM 接近预算是正常的。

本轮代码属于插件内侵入式适配：GDN MambaSpec、runner 内存 profile 与请求槽池、attention metadata、verify/compact AscendC ABI 均已改变；没有修改 vLLM 源码。若服务器构建失败，不继续用旧二进制做服务结论。

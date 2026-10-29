# L3.7 SD 服务修复与数值定位上机清单

环境继续使用 Ascend910B4 / CANN 9.2.0 / torch_npu 2.10.0.post4 / vLLM `0.30.0+empty` / 本分支源码编译的 vLLM Ascend。保持 TP4、MTP3、BF16 model/cache、verify v13/tile128、compact v1/tile32。上一轮 `ef3d9c5` 加上服务器本地两处修复后可运行，但 B1 贪心输出在 index 21 分叉；本轮先定位数值，不把输出分叉直接归因为 FP16 ring。

## 1. 构建与覆写回归

1. 拉取本次提交，确保服务器临时修改的 `attention/utils.py`、`worker/worker.py` 不再重复应用；记录 `git rev-parse HEAD` 与 `git status --short`。在 9.2.0 容器按已验证方式 `pip install -e . --no-build-isolation`，确认加载新 `.so`。
2. 运行 `python3 -m unittest tests.ut.replayssm.test_model_ar_integration tests.ut.replayssm.test_sd_serve_validator tests.ut.replayssm.test_sd_sequence_diagnostics`。
3. 启动原清单 `LEVEL3_SD_SERVE.md` 的 Replay TP4 服务。检查日志中的 `num_gpu_blocks` 定容、17/17 图 capture、`/health`、4 worker 的内存探针；先 B1，再 B32。`num_gpu_blocks` 或 `decode-base` 原错误再现时，停止并回传首条 traceback。

## 2. 单层连续窗口数值探针

下面两次运行使用同一 seed、接受序列、FP64 dense 参考与当前 BF16 checkpoint/FP16 ring 的逐步参考。`atol=0.01, rtol=0.03` 沿用现有 SD 单算子 `lab.check_result` 验收口径；这只判定**单层算子路径**，不构成整网 token 一致性标准。脚本每步写 JSON，包含累计接受数、position、base、flush、三个 status，以及 output/cache/checkpoint 对两种参考的逐位置误差。

```bash
python3 scripts/replayssm/diagnose_sd_sequence.py \
  --tp 4 --device 0 --steps 24 --accepted 3,1,4,2 --seed 42 \
  --atol 0.01 --rtol 0.03 \
  --output /tmp/replayssm-sd-sequence-fp32-beta.json

python3 scripts/replayssm/diagnose_sd_sequence.py \
  --tp 4 --device 0 --steps 24 --accepted 3,1,4,2 --seed 42 \
  --beta-native-precision --atol 0.01 --rtol 0.03 \
  --output /tmp/replayssm-sd-sequence-bf16-beta.json
```

- [ ] 记录 `first_local_mismatch_step`。若 output 或 candidate cache 对**当前 cache 的本步参考**先超差，优先检查 verify/compact 或数据布局；附该步前后 position/base/flush 与首个错误索引。
- [ ] 若本步参考均通过，但 `output_vs_dense` 逐步扩大，量化/累加顺序是候选原因。比较两种 beta 口径，再决定是否需要其他 cache dtype 或 kernel 累加实验。
- [ ] 若只在 flush 后 checkpoint 对本步参考超差，聚焦 compact；若 checkpoint 本步通过但下一步 output 分叉，聚焦 cursor 的 base/position 与 ring 读取。

这是合成输入探针，不能单独解释实际模型的 index 21 分叉。请保留两个 JSON 的每步记录和首个异常日志。

## 3. 实际模型的首分叉与 logprob 间距

用相同服务参数、prompt、seed、输出长度重新各跑一次 B1；先 native 后 Replay，两个服务不要同时占卡。`--logprobs 5` 请求模型返回每步前五个候选；如果接口拒绝该参数，保存 HTTP 错误并继续用原 `--logprobs 0` 记录 token 序列。

```bash
python3 scripts/replayssm/validate_sd_serve.py \
  --model replayssm-sd --prompt-file /tmp/replayssm-sd-short.txt \
  --concurrency 1 --repetitions 2 --max-tokens 128 --logprobs 5 \
  --output /tmp/replayssm-sd-native-logprobs.json

python3 scripts/replayssm/validate_sd_serve.py \
  --model replayssm-sd --prompt-file /tmp/replayssm-sd-short.txt \
  --concurrency 1 --repetitions 2 --max-tokens 128 --logprobs 5 \
  --compare /tmp/replayssm-sd-native-logprobs.json \
  --output /tmp/replayssm-sd-replay-logprobs.json
```

Replay 对照因 token 分叉会返回状态 `FAILED`，这时 JSON 仍应写出；保留两个 JSON。重点回传 `first_divergences` 的 index、token IDs 和两侧 top-logprob margin。很小的 margin 支持 argmax 对数值敏感，较大的 margin 则需要优先检查状态/输入对齐；**两者都不能仅凭一次 B1 断言根因**。若要看多 prompt 的首分叉分布，可提供 JSONL，每行 `{"prompt":"..."}`，将上述 `--prompt-file` 改为 `--prompts-jsonl`、`--repetitions 1`。

## 4. 收益门槛与后续性能定位

上一轮 B1/B32/B64 p50 分别为 native `2.901/15.632/45.636s`、Replay `3.285/17.776/55.417s`，回退约 **13.2%/13.7%/21.4%**。B64 包含排队，不能把总差值全部归到 forward。`peak_waiting=32` 是 `max-num-seqs=32` 的硬上限，不能据此判断 SD 是否提高了显存受限并发。

数值定位完成后，用同容器、同 prompt 长度和采样配置，把 `max-num-seqs` 提至 64，分别测 4k、8k、32k 上下文与 512/1024 输出。记录各组的可分配 block 数、full-attention 与 GDN 每请求 block 需求、running/waiting 峰值、完成请求数、吞吐及 p50/p90；同时从 profiler 对照每层 verify、compact 与 native RGDR 的 task duration，以及每个 target forward 的调用次数。当前 Python dispatch 每个 GDN 层还执行五组 `zeros + index_copy` 候选打包及一次 output `index_select`，应在图 trace 中单独统计这些任务；若它们占比高，下一步针对全 T4 窗口做无打包快路径。若 full-attention KV 先耗尽，SD 的 GDN 缓存节省不会自动转化为排队收益。

本轮不要求立即跑完长上下文压力测试；先交付第 1–3 节的数值证据。TP8 可待 8 张空卡后补。

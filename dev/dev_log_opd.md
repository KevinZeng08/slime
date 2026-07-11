# OPD Teacher–Rollout 调研记录

## 背景

当前 OPD（On-Policy Distillation）实验使用：

- Student / rollout：Qwen3-8B，由 Slime 管理多个 SGLang rollout engine。
- Teacher：Qwen3-32B，通过独立 SGLang server 提供 token-level logprob。
- Teacher 请求使用 `max_new_tokens=0`、`return_logprob=True`、`logprob_start_len=0`，因此 workload 本质上是对完整 `prompt + response` 做 prefill-only scoring。
- rollout 最长 response 为 16K tokens。

当前 teacher 实验配置为 TP=2、每个 rank CPU offload 30GB，并在测试关闭 chunked prefill：

```bash
CUDA_VISIBLE_DEVICES=6,7 python3 -m sglang.launch_server \
    --model-path /opt/tiger/models/Qwen3-32B \
    --host 0.0.0.0 \
    --port "$TEACHER_PORT" \
    --tp 2 \
    --chunked-prefill-size -1 \
    --mem-fraction-static 0.37 \
    --cpu-offload-gb 30
```

## Teacher 与 rollout 的执行关系

Teacher 不是等整个 rollout batch 生成完后统一执行，也不是在每个训练 micro-batch 上执行。

实际流程是 sample 粒度的异步流水：

1. 某个 sample 完成 rollout generation。
2. 该 sample 释放 generation semaphore。
3. 随即异步请求 teacher 计算 logprob。
4. 其他 sample 可继续 generation，与 teacher scoring 重叠。
5. 同一 prompt group 使用 `asyncio.gather` 等待组内所有 sample 的 generation 和 teacher 请求。
6. 整个 rollout step 必须等待所有有效 sample 的 teacher logprob 返回，之后才进入 Megatron 训练。

因此系统属于“sample 粒度异步、rollout step 边界同步”。Teacher 足够快时，其耗时可被 rollout 长尾隐藏；teacher 变慢时，已完成 generation 的 sample 会积压等待，最终增加 `perf/rollout_time`。

Teacher 返回的原始结果位于：

```python
reward["meta_info"]["input_token_logprobs"]
```

Slime 截取 response 对应部分后保存到：

```python
sample.teacher_log_probs
train_data["teacher_log_probs"]
```

该数据默认只存在于内存 / Ray 数据流中，不写入 checkpoint。需要持久化时可使用 `--save-debug-rollout-data` 或 `--dump-details`。

## 已验证的实验结论

### 1. Teacher CPU offload 暂未成为 critical path

已完成以下对照：

- A：teacher 独立部署，不 offload。
- B：teacher 独立部署，开启 CPU offload。

目前观察到 A 与 B 的 rollout 时间基本一致。这表明在当前 batch、序列长度和并发下，offload 带来的 teacher 降速仍能被 rollout generation 长尾隐藏，teacher 尚未成为主导 critical path。

该结论只适用于当前 workload。若 response 变短、teacher 请求并发增加或 rollout 加速，teacher 可能重新暴露为瓶颈。

### 2. Colocate 下的 KV full 来自 rollout，不是 teacher

日志中的 `KV cache pool is full. Retract requests` 均由加载 Qwen3-8B 的 `SGLangEngine` rollout 进程输出。Teacher Qwen3-32B 日志中未出现对应 KV full。

未 offload teacher 时，rollout engine 的 KV 容量明显不均衡：

- 大多数 rollout engine：`max_total_num_tokens ≈ 116872`
- 与 teacher 争用显存的最差 engine：`max_total_num_tokens ≈ 24450`

Teacher offload 后，最差 engine 改善为约 64425 tokens，但仍低于其他 engine 的约 116872 tokens。

此外，116K KV pool 的正常 engine 也发生过 retract，说明问题不只来自 colocate，还包括：

- 最长 16K response；
- 单 worker 同时承载大量长请求；
- 请求长度不可预知；
- router 分配不均衡。

### 3. Router 未感知 colocate 造成的 worker 异构

当前 rollout router 使用默认 `cache_aware` policy。Teacher 请求通过独立 `rm-url` 直接发送，并不经过 rollout router。

更准确的问题是：colocate 使 rollout workers 在以下方面变得异构，而 router 没有显式感知：

- `max_total_num_tokens` 不同；
- 可用显存不同；
- 与 teacher 的算力、PCIe 和内存带宽竞争不同；
- 长输出的最终 token 数不可预知。

日志中曾出现某个 worker 同时运行约 19 个长请求。对于大量共享 prompt 模板但 response 很长的 workload，cache locality 带来的收益可能小于负载集中造成的 KV retract 成本。

因此需要以 `round_robin` 作为简单基线，并进一步考虑按 worker KV 容量加权的 routing。

### 4. 直接 colocate 实测性能损失较大，暂时搁置

直接把 teacher 与 rollout 共卡实测下来：teacher 的计算/显存争用明显拖慢了 rollout，端到端性能损失较大（KV retract 增多、worker 异构、长尾被 teacher 进一步拉长）。在缺少负载感知调度的前提下，共卡收益为负。

因此**共卡（colocate）方向暂时搁置**，转向下面的分离部署方案；共卡的完整设计（完全共卡 baseline、负载感知 router、动态并行度）保留在 `strong-to-weak-idea/design.md` 第 4 节，作为未来重启时的参考。

## 当前主攻方向（2026-07-11）：分离部署 + 最小 teacher GPU + 放大 rollout

### 核心思路

- 观察：teacher 开/关 CPU offload 对 rollout time 影响都不大（见「已验证结论 §1」）→ teacher 常驻的大量显存/算力对当前 workload 是**冗余**的。
- teacher 是 prefill-only 打分，在 sample 粒度异步流水中只要"足够快"就能被 rollout 长尾**掩盖**。
- 因此：用**最少 GPU** serve teacher（如 TP=1 + offload），只要保证 teacher 时间能被 rollout 掩盖；把省下的 GPU 还给 rollout，**放大并行度 + 加大 batch size**。

### 下一步（优先级最高）：减少 teacher 冗余显存占用

现在的 teacher 没调优好，既慢又占大量冗余显存，是"最小化 teacher GPU"的直接阻碍。下一步优先系统性砍冗余（详细手段见下方「后续优化方向」§2/§3/§4）：

1. **KV pool 过量（最大头）**：现 212K–223K tokens，实测峰值仅约 10–20K → 下调 `mem-fraction-static` / `--max-total-tokens` 封顶 / `--disable-radix-cache`。
2. **logprob / logits 瞬时峰值**：full-seq × vocab gather → 开 chunked prefill 限制单次 token 数；只 gather response 段（`logprob_start_len`）。
3. **常驻权重**：CPU offload（已用），后续 layer-wise / prefetch。
4. **更彻底**：causal-LM scoring 的 no-KV full-prefill 路径，理论省每卡约 27GB KV。

验收：teacher 显存高水位与卡数压到最小、不 OOM、仍能被 rollout 掩盖，最终以端到端吞吐确认净收益。

### 为什么端到端更快（弱扩展）

- 固定数据集下，per-step 产出更多样本 → **总 step 更少**。
- 每 engine 样本数保持不变（如 `16×4 / 4卡` → `20×4 / 5卡`，均为 16 样本/engine）→ **per-step rollout time 基本不变**。
- 净效果：吞吐（samples/s）提升，跑完固定数据的总墙钟下降。
- ⚠️ 这是"更快跑完数据 / 吞吐提升"，**不等于"更快收敛"**；batch 变大是优化的另一个轴，可能要调 LR。
- ⚠️ 加大 `rollout-batch-size` 必须同步保证 `总样本数（rollout_batch_size × n_samples_per_prompt）` 能被 `global-batch-size` 整除（如 `20×4=80` → `--global-batch-size 80` 或 `40`）。

### 约束与验证

- **teacher 必须能被掩盖**：rollout 变快、长尾变短后，teacher 可能反过来暴露为瓶颈，需监控 teacher 打分延迟/队列 vs rollout 长尾。
- **teacher 最小 GPU 的显存坑**：TP=1 + `chunked-prefill-size -1` + `return_logprob` 时，巨型 KV pool 挤掉 logits 瞬时显存会 OOM（已复现于 `run_opd_0711_teacher_offload_chunk_-1_tp1.log`：`logits_processor` 处 `CUDA out of memory, tried 6.5GiB`）。最小化 teacher 时必须同步下调 `mem-fraction-static` 并开 chunked prefill。
- **验证指标（重要）**：端到端性能看 `perf/step_time` 与**吞吐 samples/s**，跨配置比较必须用吞吐（step_time 会随 batch 变化）；`perf/train_wait_time` / `wait_time_ratio` 拆解瓶颈；`request/e2e_latency` 的 max/p99 + `request/queue_time` 仅作 rollout 内部长尾/排队的诊断，**不能当端到端指标**。

## SGLang 显存语义

### `mem-fraction-static`

SGLang 的 KV pool 近似按以下公式计算：

```text
KV pool
= 加载模型后的空闲显存
  - 加载模型前的空闲显存 × (1 - mem_fraction_static)
≈ 启动时空闲显存 × mem_fraction_static - GPU 常驻权重
```

因此 `mem_fraction_static` 表示“权重 + KV pool”的静态预算比例，而不是整个进程的显存上限。

CPU offload 后若保持相同的 `mem_fraction_static`，释放出的权重显存会被 SGLang 重新分配给 KV pool，teacher 静态显存未必明显下降。若目标是把显存让给其他 workload，需要同步降低 `mem_fraction_static`，或直接限制 `max_total_tokens`。

### Teacher 当前显存分解

在 TP=2、`cpu-offload-gb=30`、`mem-fraction-static=0.37` 的日志中，每卡启动阶段为：

- 加载前可用显存：约 77.80GB。
- GPU 常驻权重：约 1.53GB。
- K cache：13.63GB。
- V cache：13.63GB。
- KV pool 合计：27.26GB。
- CUDA Graph：约 1.07GB。
- 初始化完成后可用显存：约 47.18GB。
- 初始化静态占用：约 34.4GB。

运行期间 `nvidia-smi` 曾显示约 59.5GB/卡。相比启动阶段增加的约 25GB 主要可能来自：

- prefill activation；
- attention workspace；
- logits / logprob 临时张量；
- CPU offload 权重回传缓冲；
- PyTorch CUDA caching allocator 保留的历史高水位。

`cpu-offload-gb` 只控制移到 CPU 的权重规模，不是整个 GPU 进程的硬显存上限。

## 当前 CPU offload 行为

当前 Qwen3 teacher 使用 SGLang V1 offload：

1. 初始化时按 decoder layer 顺序将参数移到 CPU pinned memory，直到达到 `cpu_offload_gb`。
2. 某个被 offload 的 layer 执行 forward 时，将该 module 的参数临时复制回 GPU。
3. 使用 `functional_call` 完成该 layer forward。
4. 临时权重引用释放，显存块可能被 CUDA allocator 缓存。

V1 没有显式 prefetch，活跃临时权重通常约为当前一个 layer。Qwen3-32B TP=2 时，每卡单个 decoder layer 权重大约为 0.45–0.5GB。因此当前约 59.5GB 的显存占用并不是几十个 layer 同时被 onload，主要来源仍是 KV pool 和运行期临时内存。

V2 offload 支持分组和 prefetch，可以形成“当前 layer + 下一 layer 预取”，但当前 Qwen3 dense 路径没有接入 V2 所需的 offloader callbacks，不能直接使用。

## Chunked prefill 的影响

Teacher 对完整 `prompt + response` 计算输入 token logprob。对于 16K 输入：

```text
chunk=4096  -> 至少 4 次完整模型 forward
chunk=8192  -> 至少 2 次完整模型 forward
chunk=-1    -> 可能一次完整 forward
```

### 对 CPU offload

每次模型 forward 都会重新遍历所有被 offload 的 layer，并执行 CPU→GPU 权重搬运。以每 rank offload 30GB 粗略估算，单个 16K 请求在未考虑并发合批复用时：

```text
chunk=4096  -> 约 120GB 权重传输 / rank
chunk=8192  -> 约 60GB 权重传输 / rank
chunk=-1    -> 约 30GB 权重传输 / rank
```

chunk 越大，offload 权重传输越容易被更多 token 摊薄，teacher 吞吐通常越高。

### 对 logprob 显存

chunk 不改变最终 logprob 的语义和数量，但会改变单次临时 logits、activation 和 attention workspace 的规模。

Qwen3 词表大小为 151936。若完整 materialize BF16 logits，理论规模约为：

```text
4096 tokens  -> 1.16GiB
8192 tokens  -> 2.32GiB
16384 tokens -> 4.64GiB
```

因此小 chunk 用更多权重搬运和调度开销换取更低显存峰值；大 chunk 或关闭 chunking 可减少 offload 搬运，但提高 OOM 风险。

## Prefill-only scoring 与 KV cache

Teacher 使用 `max_new_tokens=0`，理论上完整序列一次 forward 时，每个 layer 的 attention 结束后即可丢弃该层 K/V，因为没有后续 decode。

但当前 SGLang scoring 路径仍使用 paged KV cache。开启 chunked prefill 时执行顺序为：

```text
chunk 1: layer 1 -> ... -> layer N
chunk 2: layer 1 -> ... -> layer N
```

chunk 2 的每个 layer 都需要读取 chunk 1 在相同 layer 的 K/V，因此请求完成前不能逐 layer discard 历史 KV。

SGLang 已有 `--prefill-only-disable-kv-cache`，但当前仅支持 embedding workload，并要求：

- `--is-embedding`
- `--chunked-prefill-size=-1`
- `--disable-radix-cache`
- FA3/FA4 prefill backend

源码明确说明 scoring workload 当前仍会通过 paged cache staging K/V。因此 teacher causal-LM scoring 暂不能直接启用该功能。

若未来为 teacher scoring 实现 no-KV 路径，按当前配置可省去每卡完整的 27.26GB KV pool，TP=2 合计约 54.5GB。实际运行峰值不会同比下降，因为 full prefill 的 activation、logits、workspace 和 offload 缓冲仍然存在。

## 后续优化方向

### 1. 建立完整实验矩阵

继续完成以下对照：

- A：teacher 分离部署，不 offload。
- B：teacher 分离部署，CPU offload。
- C：teacher offload，利用释放显存增加 rollout 资源。

每种配置至少运行多个 warmup 后 rollout step，并固定 prompt、seed 和输出 token 分布。重点记录：

- `perf/rollout_time`
- `perf/step_time`
- response tokens / GPU / second
- teacher reward-model 延迟和队列长度
- 各 rollout worker 的 `max_total_num_tokens`
- 各 worker 的 `#running-req`、`token usage` 和 `#queue-req`
- KV retract 次数及重算 token 数
- teacher 和 rollout 的 GPU 显存高水位、SM 利用率、PCIe 带宽

### 2. 降低 teacher KV 浪费

当前 teacher KV pool 为 223245 tokens，而日志观测到的瞬时 `token usage` 最高约 0.09，即约 20K tokens。由于 `pending-token` 尚未计入已占用 KV，不能直接将 pool 缩到 20K，但当前 223K 仍可能明显过量。

可依次测试：

1. 降低 teacher `mem_fraction_static`。
2. 使用 `--max-total-tokens` 显式限制 teacher KV pool，避免 offload 释放的显存被自动重新填充为 KV。
3. 使用 `--disable-radix-cache`，避免完成请求的 KV 因 prefix cache 保留；该参数不能消除 chunk 之间的工作 KV。
4. 观察 teacher queue、token usage 和延迟，保留足够并发余量。

### 3. 优化 chunk size

比较以下配置：

- 4096：最低临时显存，最高 offload 搬运次数。
- 8192：预期是显存与传输的折中点。
- `-1`：最少 offload 搬运，最高 full-prefill 显存峰值。

当前脚本正在测试 `-1`。需要重点确认：

- 16K 长输入和并发请求下是否 OOM；
- teacher latency 是否下降；
- `perf/rollout_time` 是否变化；
- GPU 显存高水位是否仍允许后续 colocate。

### 4. 为 causal-LM scoring 实现 no-KV prefill

这是潜在收益最大的 teacher 专用优化：

1. 将 embedding 的 FA `fa_skip_kv_cache` 路径扩展到 `max_new_tokens=0` 的 scoring 请求。
2. 限制为完整 prefill、无 decode、无 prefix cache 的请求。
3. 支持 `return_logprob=True` 和 `logprob_start_len=0`。
4. 使用 NoOp KV pool，避免分配 27GB/卡 paged KV。
5. 增加 correctness test，对比标准路径的 token logprob。
6. 测试不同序列长度、TP 配置和并发下的显存峰值及吞吐。

### 5. 改善 rollout routing

短期基线：

```bash
--router-policy round_robin
```

用于判断默认 `cache_aware` 是否因公共 prompt prefix 将大量长请求集中到少数 workers。

长期方案是 capacity-aware routing：

- worker 注册时上报 `max_total_num_tokens`；
- 按 KV capacity 对 worker 分配权重；
- 使用实时 token usage，而不只使用请求数；
- 将预测输出长度或 `max_new_tokens` 纳入 admission cost；
- 对发生 retract 或高 queue 的 worker 动态降权；
- 感知 worker 是否与 teacher colocate 以及实时 GPU 资源竞争。

### 6. 控制 rollout 并发与长尾

可测试：

- `--sglang-max-running-requests`，但固定全局值无法适配 64K/116K 的异构 workers，过低会降低健康 worker 利用率。
- 降低不必要的 `rollout-max-response-len`。
- 基于 KV token budget 而不是 request count 做 admission。
- 按预测长度进行负载均衡。

优先级应是先解决 worker 容量不均和 router 不感知问题，再使用全局并发上限作为保护机制。

### 7. 改进 Qwen3 offload

若 teacher offload 最终成为 critical path，可考虑：

- 给 Qwen3 dense model 接入 V2 offloader callbacks；
- 支持 layer-level async prefetch；
- 调节预取深度，在 1–2 个 layer 峰值显存内隐藏 H2D；
- 与 full-prefill/no-KV 路径联合优化；
- 使用 Nsight Systems 分析 H2D、GEMM 和 logits kernel 是否重叠。

## 当前结论

1. Teacher CPU offload 在当前分离部署实验中尚未增加 rollout critical path，说明 teacher 常驻权重显存存在优化空间。
2. 直接 colocate **实测性能损失较大**（teacher 拖慢 rollout：KV retract、worker 异构、长尾被拉长），在缺少负载感知调度前收益为负，**暂时搁置**。
3. **当前主攻改为分离部署**：用最少 GPU serve teacher（只要能被 rollout 掩盖），把释放的 GPU 用于放大 rollout 并行度与 batch size，通过弱扩展提升吞吐、减少总 step、缩短固定数据集总墙钟。
4. 端到端性能应看 `step_time` / 吞吐 samples/s；`e2e_latency`、`queue_time` 只是 rollout 内部长尾/排队的诊断指标，不能当端到端指标。
5. Teacher 当前最大的可消除静态显存项不是权重，而是每卡 27.26GB 的 KV pool。
6. CPU offload 与 chunked prefill 存在明显冲突：小 chunk 降低显存峰值，但重复搬运大量权重。
7. 长期最有潜力的方向是 teacher 专用的 full-prefill、return-logprob、no-KV scoring 路径，再配合 capacity-aware rollout routing。

## 分离部署首个吞吐对比（2026-07-11）

对比两个 run（**均为 8 GPU 等硬件**，脚本/数据见 `rl/docs/opd/strong-to-weak-idea/perf_compare.py`、`perf_compare_*.csv`、`perf_compare.png`）：

- baseline `r7vyme3x`：teacher TP=2（2 GPU，slime 默认分卡），rollout engine=4，batch 16×4=64 样本/step，取 step 0–11。
- exp `793ky8wp`：teacher TP=1 + CPU offload（1 GPU），rollout engine=5，batch 20×4=80 样本/step，取 step 0–9。
- 两者每 engine 样本数都是 16（弱扩展，单卡负载不变）。

| 指标 | baseline | exp | exp vs baseline |
|---|---|---|---|
| `perf/rollout_time` mean | 203.88s | 223.51s | +9.6%（per-step 更慢） |
| `perf/step_time` mean（端到端） | 296.70s | 339.53s | +14.4%（per-step 更慢） |
| `perf/tokens_per_gpu_per_sec` mean（单卡） | 773.5 | 703.8 | −9.0% |
| samples/s rollout-only mean（=样本/rollout_time，偏乐观） | 0.323 | 0.361 | +11.7% |
| **samples/s 端到端** mean（=样本/step_time，结论以此为准） | 0.221 | 0.238 | **+7.7%** |
| samples/s 端到端 median | 0.230 | 0.236 | +2.4% |
| 聚合 token 吞吐（tok/gpu/s×engine，仅 rollout 段） | 3094 | 3519 | +13.7% |

要点：

1. per-step `rollout_time` / `step_time` 更慢是弱扩展的**预期行为**（每 engine 样本数不变），收益体现在端到端吞吐/总墙钟，不是 per-step 时间。
2. teacher 从 2 GPU 砍到 1 GPU + offload 后，rollout_time 只微增、未暴露为 critical path，验证了"teacher 只要够快就能被 rollout 长尾掩盖"的假设。
3. **两种吞吐口径要分清**：`samples/s = 样本 / rollout_time` 只含 rollout（偏乐观 +11.7%）；`samples/s = 样本 / step_time` 才是端到端（含 actor train + update weights，+7.7%），后者对应"固定数据集总墙钟"，结论以此为准。exp step_time 增加更多是因 batch 更大（80 vs 64）训练时间变长。
4. **收益对口径敏感**：端到端 median 仅 +2.4%；baseline step10 是 outlier（rollout_time=323s / step_time=425s），剔除后端到端 mean 降到 +4.6%。综合端到端真实收益约 **+2%～+8%，中心 ~+3–5%**。样本窗口短（10–12 步）含 warmup，后续需多跑几步、去 warmup、以端到端 samples/s / 总墙钟复核再定论。

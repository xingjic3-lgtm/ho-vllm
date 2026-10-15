# ho-vllm

一个面向学习与性能分析的轻量级 Qwen3 推理引擎。使用 PyTorch 实现模型计算、请求调度和分页 KV Cache，并用 Triton 探索算子融合。

项目关注的不只是“让模型生成文本”，而是理解一次推理如何经过调度、输入组织、KV 访问和 GPU 计算，以及如何从性能报告定位开销，再通过代码修改验证优化。

> 本说明覆盖 `main`、`baseline`、`opt-rope` 和 `opt-triton` 四个分支；下述完整优化实现对应 `main` / `opt-triton`。项目不是官方 vLLM，也不是生产级推理服务；当前主要面向单卡、Qwen3-0.6B 和 Linux / WSL2 CUDA 环境。

## 已实现的功能

- Qwen3 模型结构与本地 Safetensors 权重加载。
- 自回归生成、贪心采样与 temperature / top-k / top-p 采样。
- Continuous Batching：按步调度请求，完成后释放资源。
- 按 token budget 分块处理输入，使用 packed tokens 组织不同请求。
- 分页 KV Cache：通过 Block Table 管理逻辑位置到物理缓存块的映射。
- Selective Logits：仅对需要采样的位置计算词表输出。
- CPU 元数据复用、RoPE 系数复用、Triton Paged Gather + GQA Expand。
- 三类合成负载的重复测量，以及用于性能分析的 NVTX 标记。

## 执行流程

```text
文本 / Token IDs
      ↓
Engine：加入请求、运行生成循环
      ↓
Scheduler：选择请求，分配 token budget 与 KV blocks
      ↓
ModelRunner：打包 tokens、positions、Block Table、Slot Mapping
      ↓
Qwen3：Embedding → Decoder Layers → 选定位置的 Logits
      ↓
Sampler：采样下一个 token
      ↓
更新请求状态；继续调度，或结束并释放 KV blocks
```

## 优化实现

### 1. 避免 Attention 内重复的 `.item()` 同步

`query_start_loc` 和 `seq_lens` 本来由 CPU 构造。保留它们为 Python 列表，使 Attention 能直接读取请求边界和长度，不再先上传 GPU，再逐层调用 `.item()` 取回标量。

这项优化消除的是这部分元数据的 GPU → CPU 标量读取和相关同步，并不意味着整个引擎没有 D2H：例如生成循环仍需要把采样结果取回 CPU。

实现位置：[model_runner.py](model_runner.py)、[model/attention.py](model/attention.py)。

### 2. 每次 Forward 统一构造 RoPE 系数

在一次模型 forward 内，所有层使用相同的 packed positions。第一层构造 `cos` / `sin` 并存入当前 `AttentionMetadata`，后续层及 Q、K 共用它们。

对于 28 层模型，原先每层 Q、K 分别构造一次，即 56 次；现在每次模型 forward 只构造 1 次。这里的 forward 指一次调度步的模型执行，不是整个请求的生成过程。

当前版本的实际旋转仍由 PyTorch 实现，**不包含 Triton RoPE 旋转融合算子**。系数针对本轮输入 positions 构造，而非预先分配模型最大上下文长度的完整缓存。

实现位置：[forward_context.py](forward_context.py)、[model/attention.py](model/attention.py)。

### 3. Triton 融合 Paged Gather + GQA Expand

一个 kernel 同时完成当前请求的 K/V 读取和 GQA head 展开：

1. 根据 token 位置查找逻辑块与块内偏移。
2. 通过该请求的 Block Table 获取物理块。
3. 将 query head 映射到对应 KV head。
4. 直接从物理 KV Cache 读取，写入展开后的 K/V 张量。

这样将分页 gather 与 head 展开的多步操作合为一次 kernel launch，减少中间处理。当前实现仍会生成展开后的 K/V 张量，后续 Attention 使用 PyTorch 矩阵乘法和 Softmax；它不是融合式 Paged Attention 或 FlashAttention。

实现位置：[model/kernels/paged_gather_gqa.py](model/kernels/paged_gather_gqa.py)。

## 快速开始

### 环境与依赖

本地开发环境使用 WSL2、Python 3.12、NVIDIA GeForce RTX 5060 Ti，以及 CUDA 12.8 版 PyTorch。当前 Triton 路径请在 Linux / WSL2 内运行，不使用 Windows 原生 Python。

主要依赖为 `torch`、`triton`、`transformers` 和 `safetensors`。仓库目前没有完整锁定依赖版本，跨机器运行需要确认驱动、PyTorch 与 Triton 的兼容性。

已有环境的本机运行方式：

```bash
source /home/chen/projects/dzyy-vllm-local/.venv-cu128/bin/activate
export QWEN3_MODEL_PATH=/home/chen/projects/dzyy-vllm-local/models/Qwen3-0.6B
cd /mnt/d/Project/dzyy-vllm/ho-vllm
python run.py
```

其他机器请将环境和模型路径替换为自己的路径。已配置兼容环境时，进入克隆后的仓库即可直接 `python run.py`，不需要专用启动脚本。

可以先检查依赖和 GPU：

```bash
python -c "import torch, triton, transformers, safetensors; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available())"
```

### 准备模型

模型权重不包含在仓库中，需要自行准备本地 Qwen3-0.6B 模型目录，并设置 `QWEN3_MODEL_PATH`：

```bash
export QWEN3_MODEL_PATH=/absolute/path/to/Qwen3-0.6B
python run.py
```

目录需要包含模型配置、Tokenizer 文件和权重，例如：

```text
Qwen3-0.6B/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── model.safetensors
```

当前加载器直接读取单个 `model.safetensors`，未实现分片权重索引加载。Tokenizer 也只从本地读取，不会自动下载缺失文件。

`run.py` 展示添加两个请求并执行生成的最小示例。它输出的是包含输入 prompt 的完整 token 序列解码结果。

## Benchmark

运行当前学习版测试：

```bash
python bench/perf_experiment2.py
```

| 负载名称 | 请求数 | 每请求输入 tokens | 每请求最大输出 tokens |
|---|---:|---:|---:|
| Prefill-heavy | 8 | 1024 | 16 |
| Mixed | 8 | 512 | 64 |
| Decode-heavy | 8 | 128 | 256 |

测试使用固定随机种子生成合成 token 输入；每组输入预热 1 次，正式测量 5 次，报告每次耗时以及中位数、最小值和最大值。可在脚本顶部修改 `WORKLOADS`、`WARMUP_RUNS` 和 `MEASURE_RUNS`。

计时范围是 `engine.run()`，前后使用 CUDA 同步。模型初始化、请求构造与入队不在计时范围内。每次测量重新创建请求，但复用已加载的引擎。

解读结果时请注意：

- 三个名称表示不同输入/输出比例，测量的均是完整生成过程，不是单独的 Prefill 或 Decode 阶段耗时。
- 请求数 8 是提交的请求总数；实际并发受 `engine.py` 中调度配置约束，当前 `max_num_seqs=4`。
- 输出长度是上限，遇到 EOS 可能提前结束。比较版本时应同时核对实际输出 token 数和内容。
- 该脚本不自动做输出一致性回归，也不报告 TTFT、TPOT 或峰值显存。
- 性能对比应统一输入、测试脚本、调度配置、环境和测量口径，并逐个运行版本，避免同时争用 GPU。

本 README 不将其他实验分支的加速比作为当前学习版本的结果。可复现的性能结论需要绑定具体提交、硬件、负载和正确性检查。

## 代码导航

```text
ho-vllm/
├── run.py                       # 最小生成示例
├── engine.py                    # 请求入口与生成循环
├── scheduler.py                 # 请求状态、调度与 token budget
├── kv_cache_manager.py          # KV 缓存块分配与回收
├── model_runner.py              # 输入打包、KV 绑定与模型执行
├── forward_context.py           # 本轮 Attention 元数据
├── sampler.py                   # 采样策略
├── config.py                    # 读取模型配置
├── model/
│   ├── model.py                 # Qwen3 模型与 Decoder Layer
│   ├── layers.py                # RMSNorm、MLP 等基础层
│   ├── attention.py             # RoPE、KV 写入与 Attention
│   ├── loader.py                # 本地权重加载
│   └── kernels/paged_gather_gqa.py
├── bench/
│   ├── perf_experiment2.py      # 当前学习版多负载 benchmark
│   └── results/                 # 历史测量与 profiler 产物
└── docs/                        # 分阶段实现笔记
```

建议按 `run.py → engine.py → scheduler.py → model_runner.py → model/attention.py → Triton kernel` 的顺序阅读。

## 版本与学习路线

| 提交 | 阶段 |
|---|---|
| `5441612` | 共同基础：性能优化前的学习起点 |
| `219f826` | 保留 CPU 元数据，复用每轮 RoPE 系数 |
| `5f1798e` | Triton Paged Gather + GQA Expand |
| `1986073` | 添加多负载重复测量与 token 请求接口 |

`.item()` 优化与 RoPE 系数复用保存在同一个提交中；benchmark 版本没有继续修改模型计算或 Triton 算子。

旧 AI 实验路线不作为当前发布版本；历史代码仍保存在外部备份。四个发布分支保留各自阶段的模型实现，并使用完全相同的 `bench/perf_experiment2.py`。

历史设计说明：[v0](docs/v0.md)、[v1](docs/v1.md)、[v2](docs/v2.md)。历史笔记可能描述旧接口，当前运行方式以本 README 和实际代码为准。

## 当前边界

- 面向单卡学习实验，未实现分布式推理、在线服务接口和生产级容错。
- KV Cache 容量与调度参数为静态配置；大负载需要考虑显存及缓存容量，不能无限增加请求长度或并发。
- Attention 仍显式构造 scores 与 causal mask，长上下文下内存与计算开销较大。
- Triton kernel 针对当前 KV 布局设计，不是支持任意 stride 和模型结构的通用实现。
- 修改算子后应先验证多请求、跨页访问、不同序列长度下的输出正确性，再比较性能。

## 克隆与依赖安装

```bash
git clone https://github.com/xingjic3-lgtm/ho-vllm.git
cd ho-vllm
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
export QWEN3_MODEL_PATH=/absolute/path/to/Qwen3-0.6B
python run.py
```

依赖版本取自本机开发环境；其他 GPU / 驱动环境需要自行验证兼容性。模型、虚拟环境不提交到仓库。

## 用 Worktree 对比四个阶段

| 分支 | 模型计算 | 用途 |
|---|---|---|
| `main` | 完整学习版：CPU 元数据 + RoPE 复用 + Triton Gather/GQA | 默认入口与完整说明 |
| `baseline` | 原始优化前实现 | 测量基线 |
| `opt-rope` | CPU 元数据 + RoPE 系数复用 | 分阶段比较 |
| `opt-triton` | 在 RoPE 阶段增加 Triton Gather/GQA | 最终优化阶段 |

`main` 和 `opt-triton` 的模型计算相同，不应被视作两次不同的加速优化。每个阶段都已加入 token 请求接口和同一套 benchmark，旧版本并非缺少完整功能的最终版，而是有意保留的对照实现。

克隆后，在主仓库执行：

```bash
git fetch origin
git worktree add --detach ../ho-vllm-perf-baseline origin/baseline
git worktree add --detach ../ho-vllm-rope origin/opt-rope
git worktree add --detach ../ho-vllm-triton origin/opt-triton
git worktree list
```

激活同一个环境、设置同一个模型路径，依次执行，**不要同时启动多个 GPU 测试**：

```bash
python ../ho-vllm-perf-baseline/bench/perf_experiment2.py
python ../ho-vllm-rope/bench/perf_experiment2.py
python ../ho-vllm-triton/bench/perf_experiment2.py
```

每个脚本根据自己所在目录导入该版本的模型代码，不需要切换主工作目录的分支。不同 worktree 的文件修改互不影响，但共享 Git 历史。若要长期保存某个 worktree 的修改，在其目录内运行 `git switch -c 自己的分支名` 后再提交。

### 历史版本：保留探索过程

旧 benchmark、正确性实验和 profiler 脚本已从当前发布分支移除，但仍保留在历史提交中。例如，`1986073` 是清理前的 benchmark 快照：

```bash
git worktree add --detach ../ho-vllm-benchmark 1986073
python ../ho-vllm-benchmark/bench/perf_experiment2.py
```

也可以将提交替换为 `5441612`、`219f826`、`5f1798e` 查看原始阶段。注意：原始提交不一定具备统一测试接口；复现表中的阶段比较请使用上面的发布分支。目录已存在时不要重复创建。

用完后，在主仓库中移除不再需要的 worktree：

```bash
git worktree remove ../ho-vllm-benchmark
```

有未提交修改时 Git 会拒绝移除，请先保存需要的内容。不建议直接删除目录；若目录已手动删除，可用 `git worktree prune` 清理失效登记。

## 已记录的端到端性能

以下为作者保存在各版本脚本末尾的历史测量，整理时未重新运行完整 benchmark。每项为预热 1 次、测量 5 次的中位数，单位为秒。

| 阶段 | Prefill-heavy | Mixed | Decode-heavy |
|---|---:|---:|---:|
| baseline | 4.566 | 11.988 | 49.136 |
| CPU 元数据 + RoPE | 2.217 | 6.728 | 25.258 |
| Triton Gather + GQA | 2.065 | 6.028 | 21.631 |
| 最终阶段相对基线 | 2.21× | 1.99× | 2.27× |

原始五次数据见 [历史测量记录](bench/results/historical_benchmark.md)。这些结果存在运行波动，且当前脚本不自动验证输出一致性或记录实际生成长度，因此属于学习实验记录，不构成严格控制条件下的通用性能保证。加速比按中位耗时相除计算。

# 模型执行性能优化实验日志

## 2026-10-08 / 00：现场与实验约束

- 用户指定仓库 `D:\Project\dzyy-vllm\learn-vllm`；实际执行直接使用对应 WSL 路径 `/mnt/d/Project/dzyy-vllm/learn-vllm`，不使用 Linux 下旧副本。
- 原始提交 `b014b64`，独立分支 `perf/model-execution-20261008`。开始时存在用户修改 `bench/bench_serving.py` 和未跟踪的 `bench/analyze_gpu_gaps.py`，保留且不纳入提交。
- 环境说明：`wsl.exe -d Ubuntu-24.04`，激活 `/home/chen/projects/dzyy-vllm-local/.venv-cu128/bin/activate`；Nsight Systems 路径 `/opt/nvidia/nsight-systems-cli/2026.4.1/bin`。
- GPU：RTX 5060 Ti，8151 MiB，Windows/WSL，共享显示设备。无法假定独占或锁定频率，记录重复测量分布。
- 边界：只改模型执行、Attention、底层算子；Engine、Scheduler、Request、Sampler 不改。

## 01：执行链路与待验证假设

`Engine.step → Scheduler.schedule → ModelRunner.prepare_inputs → execute_model → embedding → 28 个 DecoderLayer → final norm → selective lm_head → 原始 Sampler → request 更新`。

KV 缓存由 Runner 分配，形状 `[2, layers, blocks, block_size, kv_heads, head_dim]`；每层先归一化 Q/K、RoPE、写入分页 KV，再按请求收集 KV、复制 GQA heads、显式 QK/softmax/PV。chunked prefill 使用绝对 position mask。调度默认 4 sequences / 256 tokens，块池 256 × 16。

观察（目前仅静态代码证据，尚非性能结论）：每个请求每层有 3 次 GPU `.item()`；每层重复分页索引构造、GQA 展开及 RoPE 频率计算。优先测量同步与 launch 开销，不能据此直接声称提速。

原 `bench/evaluation.py` 指向失效模型路径，且按旧 forward 接口调用、未建立 ForwardContext，不能直接作为当前正确性验证。新增独立实验 harness，不改用户 Benchmark；固定请求、随机种子、调度参数，计时 `Engine.run()`（包括调度、采样、CPU/GPU 同步），准备请求和 correctness 记录不计时。正式数据不开 Profiler；每个工作负载预热后重复测量，同时保存全部输出 token。

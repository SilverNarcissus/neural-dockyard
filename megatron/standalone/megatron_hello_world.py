"""
Megatron-LM Hello World — AI Infra 学习项目

这个程序演示 Megatron-LM 核心基础设施的关键概念：
1. PyTorch 分布式初始化 (模拟多GPU环境)
2. 张量并行的基本原理 (Column/Row 切分)
3. 进程组管理
4. 分布式通信操作 (all-reduce, all-gather)

可以在 K8s Pod 中运行，体验完整的容器化部署流程。
"""

import os
import socket
import datetime
import torch
import torch.distributed as dist


def print_header(title: str):
    width = 60
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_section(title: str):
    print(f"\n{'─' * 50}")
    print(f"  📌 {title}")
    print(f"{'─' * 50}")


def demo_environment_info():
    """展示运行环境信息 — 对应 JD 中 '了解GPU硬件架构'"""
    print_section("1. 运行环境信息")

    print(f"  主机名:     {socket.gethostname()}")
    print(f"  时间:       {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  PyTorch:    {torch.__version__}")
    print(f"  CUDA可用:   {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  GPU数量:    {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU[{i}]:     {props.name}")
            print(f"    显存:     {props.total_mem / 1024**3:.1f} GB")
            print(f"    SM数量:   {props.multi_processor_count}")
            print(f"    算力:     {props.major}.{props.minor}")
    else:
        print("  (CPU模式 — K8s集群未配GPU节点，使用CPU模拟)")
        print(f"  CPU核心数:  {os.cpu_count()}")

    # 显示关键环境变量 — 这些是分布式训练必须的
    env_vars = [
        "MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK",
        "KUBERNETES_SERVICE_HOST", "HOSTNAME",
    ]
    print("\n  关键环境变量:")
    for var in env_vars:
        val = os.environ.get(var, "(未设置)")
        print(f"    {var}: {val}")


def demo_distributed_init():
    """演示分布式初始化 — 对应 Megatron initialize.py"""
    print_section("2. 分布式初始化 (模拟 Megatron initialize_megatron)")

    # Megatron 使用 torchrun 注入这些环境变量
    # 在单进程模式下我们手动设置
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    if world_size > 1 and dist.is_available():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
        print(f"  ✅ 分布式初始化完成: backend={backend}, world_size={world_size}, rank={rank}")
    else:
        print(f"  ℹ️  单进程模式: world_size={world_size}, rank={rank}")
        print("  (多节点训练时，torchrun 会自动设置 WORLD_SIZE/RANK)")

    return world_size, rank


def demo_tensor_parallel():
    """
    演示张量并行原理 — 对应 Megatron tensor_parallel/layers.py

    ColumnParallelLinear: 权重按列切分
      W = [W1 | W2]  (每个TP rank持有一列)
      Y_i = X @ W_i  (本地GEMM)
      Y = concat(Y_1, Y_2)  (all-gather)

    RowParallelLinear: 权重按行切分
      W = [W1; W2]  (每个TP rank持有一行)
      Y_i = X_i @ W_i
      Y = sum(Y_1, Y_2)  (all-reduce)
    """
    print_section("3. 张量并行原理演示 (Tensor Parallelism)")

    # 模拟一个简单的矩阵乘法，展示 Column/Row 切分
    torch.manual_seed(42)
    hidden_size = 8
    tp_size = 2  # 模拟 2-way 张量并行

    X = torch.randn(2, hidden_size)  # [batch, hidden]
    W = torch.randn(hidden_size, hidden_size)  # [hidden, hidden]

    # --- 标准计算 ---
    Y_standard = X @ W
    print(f"  输入 X:       shape={list(X.shape)}")
    print(f"  权重 W:       shape={list(W.shape)}")
    print(f"  标准输出 Y:   shape={list(Y_standard.shape)}")

    # --- Column Parallel (Megatron ColumnParallelLinear) ---
    print(f"\n  📐 Column Parallel (TP={tp_size}):")
    col_chunk = hidden_size // tp_size
    W_col_splits = [W[:, i*col_chunk:(i+1)*col_chunk] for i in range(tp_size)]
    Y_col_parts = [X @ w_i for w_i in W_col_splits]
    Y_col = torch.cat(Y_col_parts, dim=-1)  # all-gather
    print(f"    每个TP rank权重: shape={list(W_col_splits[0].shape)}")
    print(f"    每个TP rank输出: shape={list(Y_col_parts[0].shape)}")
    print(f"    all-gather后:    shape={list(Y_col.shape)}")
    print(f"    结果一致: {torch.allclose(Y_standard, Y_col, atol=1e-6)} ✅")

    # --- Row Parallel (Megatron RowParallelLinear) ---
    print(f"\n  📐 Row Parallel (TP={tp_size}):")
    row_chunk = hidden_size // tp_size
    W_row_splits = [W[i*row_chunk:(i+1)*row_chunk, :] for i in range(tp_size)]
    X_splits = [X[:, i*row_chunk:(i+1)*row_chunk] for i in range(tp_size)]
    Y_row_parts = [x_i @ w_i for x_i, w_i in zip(X_splits, W_row_splits)]
    Y_row = sum(Y_row_parts)  # all-reduce
    print(f"    每个TP rank输入: shape={list(X_splits[0].shape)}")
    print(f"    每个TP rank权重: shape={list(W_row_splits[0].shape)}")
    print(f"    all-reduce后:    shape={list(Y_row.shape)}")
    print(f"    结果一致: {torch.allclose(Y_standard, Y_row, atol=1e-6)} ✅")


def demo_pipeline_parallel():
    """
    演示流水线并行原理 — 对应 Megatron pipeline_parallel/schedules.py

    1F1B Schedule:
      Stage 0: F F F F | F B F B F B F B | B B B B
      Stage 1:   F F F F | F B F B F B F B | B B B B
    """
    print_section("4. 流水线并行调度演示 (Pipeline Parallelism)")

    pp_size = 4
    num_microbatches = 8

    print(f"  PP stages: {pp_size}, Microbatches: {num_microbatches}")
    print(f"\n  1F1B Schedule 可视化:")

    for stage in range(pp_size):
        warmup = pp_size - stage - 1
        steady = num_microbatches - warmup
        cooldown = warmup

        schedule = []
        # Warmup: only forward
        schedule.extend(["F"] * warmup)
        # Steady: 1F1B
        for _ in range(steady):
            schedule.extend(["F", "B"])
        # Cooldown: only backward
        schedule.extend(["B"] * cooldown)

        line = " ".join(schedule)
        padding = "  " * stage
        print(f"    Stage {stage}: {padding}{line}")

    bubble = (pp_size - 1) / num_microbatches * 100
    print(f"\n  气泡率 (Bubble): {bubble:.1f}%")
    print(f"  公式: (pp_size - 1) / num_microbatches = ({pp_size}-1)/{num_microbatches}")


def demo_distributed_optimizer():
    """
    演示分布式优化器原理 — 对应 Megatron optimizer/distrib_optimizer.py

    标准:    每个rank持有完整optimizer state → 18 bytes/param
    分布式:  state按DP切分 → 6 + 12/DP bytes/param
    """
    print_section("5. 分布式优化器内存估算 (ZeRO-like)")

    model_params_billion = 7  # 7B model
    dp_sizes = [1, 2, 4, 8, 16, 32, 64]

    print(f"  模型参数量: {model_params_billion}B")
    print(f"  混合精度训练 (fp16 model + fp32 optimizer states)")
    print()
    print(f"  {'DP Size':>8} | {'标准(GB)':>10} | {'分布式(GB)':>12} | {'节省':>8}")
    print(f"  {'─'*8}-+-{'─'*10}-+-{'─'*12}-+-{'─'*8}")

    for dp in dp_sizes:
        standard_bytes = 18 * model_params_billion * 1e9  # 18 bytes/param
        distributed_bytes = (6 + 12 / dp) * model_params_billion * 1e9

        standard_gb = standard_bytes / 1024**3
        distributed_gb = distributed_bytes / 1024**3
        saving = (1 - distributed_gb / standard_gb) * 100

        print(f"  {dp:>8} | {standard_gb:>10.1f} | {distributed_gb:>12.1f} | {saving:>7.1f}%")


def demo_communication_patterns():
    """
    演示分布式通信模式 — 对应 Megatron 中的 NCCL 通信

    实际训练中这些操作通过 NCCL 在 GPU 间执行，
    走 NVLink (节点内) 或 InfiniBand/RDMA (节点间)
    """
    print_section("6. 分布式通信模式演示")

    torch.manual_seed(42)

    # All-Reduce: 梯度同步 (DP)
    print("  📡 All-Reduce (数据并行梯度同步):")
    grads = [torch.randn(4) for _ in range(4)]  # 4个rank的梯度
    reduced = sum(grads) / len(grads)
    print(f"    Rank 0 grad: {grads[0].tolist()[:2]}...")
    print(f"    All-reduce结果: {reduced.tolist()[:2]}...")
    print(f"    → 实际走 NCCL all-reduce, 节点内NVLink/节点间IB")

    # All-Gather: 参数收集 (TP/分布式优化器)
    print("\n  📡 All-Gather (张量并行/分布式优化器参数收集):")
    shards = [torch.randn(2) for _ in range(4)]
    gathered = torch.cat(shards)
    print(f"    Rank 0 shard: {shards[0].tolist()}")
    print(f"    All-gather结果: shape={list(gathered.shape)}")
    print(f"    → 分布式优化器 step 后 all-gather 更新后的参数")

    # Reduce-Scatter: 梯度分片 (分布式优化器)
    print("\n  📡 Reduce-Scatter (分布式优化器梯度分片):")
    print(f"    每个rank的完整梯度 → reduce-scatter → 每个rank只得到自己shard的梯度")
    print(f"    → 比 all-reduce 节省 (DP-1)/DP 的显存")

    # P2P Send/Recv: 流水线并行
    print("\n  📡 P2P Send/Recv (流水线并行stage间通信):")
    print(f"    Stage 0 → send activation → Stage 1")
    print(f"    Stage 1 → send gradient  → Stage 0")

    # All-to-All: MoE专家并行
    print("\n  📡 All-to-All (MoE专家并行 token dispatch):")
    print(f"    Router分配token到不同expert")
    print(f"    All-to-All 把token发送到持有对应expert的rank")


def demo_checkpoint_sharding():
    """
    演示分布式Checkpoint原理 — 对应 Megatron dist_checkpointing/
    """
    print_section("7. 分布式Checkpoint演示")

    print("  传统Checkpoint:")
    print("    rank 0 收集所有参数 → 单文件保存 → 瓶颈!")
    print()
    print("  Megatron 分布式Checkpoint:")
    print("    每个rank保存自己的shard → 并行写盘 → 高吞吐")
    print()
    print("  ShardedTensor 元数据:")

    # 模拟一个 ShardedTensor
    global_shape = (4096, 4096)
    tp_size = 4
    for tp_rank in range(tp_size):
        offset = tp_rank * (global_shape[1] // tp_size)
        local_shape = (global_shape[0], global_shape[1] // tp_size)
        print(f"    TP Rank {tp_rank}: global_shape={global_shape}, "
              f"local_shape={local_shape}, offset=(0, {offset})")

    print()
    print("  异步Checkpoint流程:")
    print("    plan → write(后台线程) → finalize(写metadata.json)")
    print("    → 训练不停，写盘与训练重叠")
    print()
    print("  Resharding (跨并行配置加载):")
    print("    TP=4保存 → TP=8加载: 自动拆分shard")
    print("    → 只交换元数据，每个rank读自己需要的部分")


def main():
    print_header("🚀 Megatron-LM Hello World — AI Infra 学习项目")
    print(f"  运行在 Kubernetes Pod 中" if os.environ.get("KUBERNETES_SERVICE_HOST") else "  本地运行模式")
    print()

    demo_environment_info()
    demo_distributed_init()
    demo_tensor_parallel()
    demo_pipeline_parallel()
    demo_distributed_optimizer()
    demo_communication_patterns()
    demo_checkpoint_sharding()

    print_section("✅ 总结")
    print("  本程序演示了 Megatron-LM 的核心 AI Infra 概念:")
    print("  1. 分布式初始化 (NCCL/Gloo, torchrun 环境变量)")
    print("  2. 张量并行 (Column/Row 切分, all-reduce/all-gather)")
    print("  3. 流水线并行 (1F1B 调度, 气泡率)")
    print("  4. 分布式优化器 (ZeRO-like, reduce-scatter/all-gather)")
    print("  5. 通信模式 (NVLink/IB, NCCL)")
    print("  6. 分布式Checkpoint (分片/异步/resharding)")
    print()
    print("  下一步: 在多GPU K8s集群上用 torchrun 启动多进程训练!")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Megatron-LM 数据并行 (Data Parallel) 分布式训练演示

在 K8s 上使用 torchrun 启动多个 worker，体验真正的分布式训练：
- 每个 Pod 是一个 worker (rank)
- 数据按 rank 切分，每个 worker 只看到 1/N 的数据
- 前向/反向后，梯度通过 all-reduce 同步
- 所有 worker 的模型参数始终一致

对应 Megatron 中 DP 的核心逻辑:
  megatron/core/distributed/distributed_data_parallel.py
"""

import os
import socket
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler


def print_rank0(msg, rank):
    """只在 rank 0 打印，避免多进程输出混乱"""
    if rank == 0:
        print(msg, flush=True)


def print_all(msg, rank, world_size):
    """所有 rank 按顺序打印"""
    for i in range(world_size):
        if rank == i:
            print(f"  [Rank {rank}] {msg}", flush=True)
        dist.barrier()


def demo_distributed_init():
    """
    步骤 1: 分布式初始化

    对应 Megatron: megatron/training/initialize.py → initialize_megatron()
    在 K8s 中，torchrun 通过环境变量注入:
      MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, LOCAL_RANK
    """
    # torchrun 已经设置好了这些环境变量
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # 选择通信后端: GPU用nccl, CPU用gloo
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    print_rank0("=" * 60, rank)
    print_rank0("  🚀 Megatron DP 分布式训练演示", rank)
    print_rank0("=" * 60, rank)
    print_rank0("", rank)
    print_rank0("📌 步骤 1: 分布式初始化", rank)
    print_rank0(f"  Backend:    {backend}", rank)
    print_rank0(f"  World Size: {world_size}", rank)
    print_rank0(f"  Device:     {device}", rank)
    print_rank0("", rank)

    # 每个 rank 报到
    print_all(f"hostname={socket.gethostname()}, device={device}", rank, world_size)

    return rank, world_size, local_rank, device


def demo_data_parallel_split(rank, world_size, device):
    """
    步骤 2: 数据切分

    DP 的核心: 每个 rank 只看到 1/N 的数据
    对应 Megatron: megatron/training/datasets/data_samplers.py
      → MegatronPretrainingSampler 按 DP rank 切分连续 batch
    """
    print_rank0("\n📌 步骤 2: 数据并行切分", rank)

    # 创建一个简单的回归数据集: y = 2x + 1
    torch.manual_seed(42)
    num_samples = 1000
    X = torch.randn(num_samples, 10)
    Y = X @ torch.ones(10, 1) * 2 + 1 + torch.randn(num_samples, 1) * 0.1

    dataset = TensorDataset(X, Y)

    # DistributedSampler: 每个 rank 自动获得不重叠的数据子集
    # 这就是 DP 的数据切分!
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    samples_per_rank = len(sampler)
    print_rank0(f"  总样本数:        {num_samples}", rank)
    print_rank0(f"  每个rank样本数:  {samples_per_rank} (= {num_samples}/{world_size})", rank)
    print_rank0(f"  Batch size:      32", rank)
    print_rank0(f"  每个rank步数:    {len(dataloader)}", rank)

    # 验证数据确实不重叠
    sampler_indices = list(sampler)
    print_all(f"数据索引范围: [{min(sampler_indices)}, {max(sampler_indices)}], "
              f"共 {len(sampler_indices)} 个样本", rank, world_size)

    return dataloader, sampler


def demo_ddp_model(rank, world_size, device):
    """
    步骤 3: 模型包装为 DDP

    对应 Megatron: megatron/core/distributed/distributed_data_parallel.py
    DDP 做了什么:
      1. 广播 rank 0 的参数到所有 rank (确保初始状态一致)
      2. 注册 backward hook: 梯度计算完成后自动 all-reduce
      3. 支持 bucket 级梯度同步 (overlap with backward)
    """
    print_rank0("\n📌 步骤 3: 构建 DDP 模型", rank)

    # 简单的 MLP (模拟 Transformer 的一小部分)
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    ).to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print_rank0(f"  模型参数量: {total_params:,}", rank)
    print_rank0(f"  内存占用:   ~{total_params * 4 / 1024:.1f} KB (fp32)", rank)

    # 包装为 DDP — 这是关键一步!
    ddp_model = DDP(model, device_ids=None)  # CPU 模式不传 device_ids

    # 验证所有 rank 的初始参数一致
    param_sum = sum(p.sum().item() for p in ddp_model.parameters())
    print_all(f"参数校验和: {param_sum:.6f}", rank, world_size)
    print_rank0("  ✅ 所有 rank 参数一致 (DDP 已广播 rank 0 的参数)", rank)

    # 内存对比: DDP vs 分布式优化器
    print_rank0(f"\n  💡 内存对比 (假设 7B 参数模型):", rank)
    print_rank0(f"     DDP (标准):      每个rank 18 bytes/param = 117 GB", rank)
    print_rank0(f"     Megatron分布式:  每个rank 6+12/{world_size} = "
                f"{6 + 12/world_size:.1f} bytes/param = "
                f"{(6 + 12/world_size) * 7:.1f} GB", rank)
    print_rank0(f"     → Megatron 分布式优化器节省 "
                f"{(1 - (6+12/world_size)/18)*100:.0f}% 显存!", rank)

    return ddp_model


def demo_training_loop(rank, world_size, device, ddp_model, dataloader, sampler):
    """
    步骤 4: 分布式训练循环

    对应 Megatron: megatron/training/training.py → train()
    每个 step:
      1. 前向传播 (每个 rank 用自己的 data shard)
      2. 反向传播 (计算本地梯度)
      3. 梯度 all-reduce (DDP 自动做, Megatron 用 bucket 级 overlap)
      4. 优化器更新 (每个 rank 独立更新, 但因为梯度一致所以参数也一致)
    """
    print_rank0("\n📌 步骤 4: 分布式训练", rank)

    optimizer = optim.Adam(ddp_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    num_epochs = 3
    print_rank0(f"  Epochs: {num_epochs}", rank)
    print_rank0(f"  Optimizer: Adam (lr=0.01)", rank)
    print_rank0(f"  Loss: MSELoss", rank)
    print_rank0("", rank)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # 确保每个 epoch 数据 shuffle 不同
        ddp_model.train()

        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = ddp_model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()         # ← DDP 在这里自动触发 all-reduce!
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        elapsed = time.time() - start_time

        # 收集所有 rank 的 loss 取平均 (用于日志)
        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        global_avg_loss = loss_tensor.item()

        print_rank0(f"  Epoch {epoch+1}/{num_epochs}: "
                    f"loss={global_avg_loss:.4f}, "
                    f"time={elapsed:.2f}s, "
                    f"batches/rank={num_batches}", rank)

    return global_avg_loss


def demo_gradient_sync_detail(rank, world_size, device, ddp_model, dataloader):
    """
    步骤 5: 梯度同步细节展示

    展示 all-reduce 前后的梯度变化
    对应 Megatron: megatron/core/distributed/param_and_grad_buffer.py
      → bucket 级 reduce-scatter + all-gather
    """
    print_rank0("\n📌 步骤 5: 梯度同步细节", rank)

    criterion = nn.MSELoss()

    # 取一个 batch
    batch_x, batch_y = next(iter(dataloader))
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    # 清零梯度
    ddp_model.zero_grad()

    # 前向 + 反向 (DDP 会 all-reduce)
    output = ddp_model(batch_x)
    loss = criterion(output, batch_y)
    loss.backward()

    # DDP 已经完成 all-reduce，现在梯度应该在所有 rank 上一致
    first_param_grad = list(ddp_model.parameters())[0].grad
    grad_sum = first_param_grad.sum().item()
    grad_norm = first_param_grad.norm().item()

    print_all(f"第一层梯度: sum={grad_sum:.6f}, norm={grad_norm:.6f}", rank, world_size)
    print_rank0("  ✅ 所有 rank 梯度一致 — 这就是 all-reduce 的效果!", rank)

    print_rank0(f"\n  📡 通信量分析 (对应 Megatron 的通信优化):", rank)
    total_params = sum(p.numel() for p in ddp_model.parameters())
    comm_bytes = total_params * 4  # fp32
    print_rank0(f"     标准 all-reduce: 2 × {comm_bytes/1024:.1f} KB = "
                f"{2*comm_bytes/1024:.1f} KB", rank)
    print_rank0(f"     (ring all-reduce: 每个rank发送 2×(N-1)/N × 数据量)", rank)
    print_rank0(f"\n  💡 Megatron 优化:", rank)
    print_rank0(f"     1. overlap_grad_reduce: 梯度reduce与backward重叠", rank)
    print_rank0(f"     2. bucket级同步: 不等所有梯度算完就开始通信", rank)
    print_rank0(f"     3. 分布式优化器: reduce-scatter替代all-reduce, 省显存", rank)


def demo_model_consistency_check(rank, world_size, device, ddp_model):
    """
    步骤 6: 验证模型一致性

    训练后所有 rank 的参数应该完全一致
    """
    print_rank0("\n📌 步骤 6: 训练后模型一致性验证", rank)

    param_sum = sum(p.sum().item() for p in ddp_model.parameters())
    print_all(f"训练后参数校验和: {param_sum:.6f}", rank, world_size)
    print_rank0("  ✅ 所有 rank 参数一致 — DP 保证了数学等价性!", rank)

    # 用相同输入验证输出一致
    torch.manual_seed(123)
    test_input = torch.randn(1, 10).to(device)
    ddp_model.eval()
    with torch.no_grad():
        test_output = ddp_model(test_input)

    print_all(f"相同输入的输出: {test_output.item():.6f}", rank, world_size)
    print_rank0("  ✅ 相同输入 → 相同输出 — 数据并行的正确性验证!", rank)


def print_summary(rank, world_size):
    """总结"""
    print_rank0("\n" + "=" * 60, rank)
    print_rank0("  ✅ 分布式 DP 训练完成!", rank)
    print_rank0("=" * 60, rank)
    print_rank0(f"""
  本次演示了 {world_size} 个 worker 的数据并行训练:

  🔑 核心流程:
  1. torchrun 启动 {world_size} 个进程, 注入 RANK/WORLD_SIZE
  2. dist.init_process_group() 建立通信 (对应 Megatron initialize)
  3. DistributedSampler 切分数据 (每个rank看 1/{world_size} 数据)
  4. DDP 包装模型 (广播参数 + 注册梯度同步 hook)
  5. 训练循环中 backward() 自动触发 all-reduce
  6. 所有 rank 梯度/参数始终一致

  📡 通信模式:
  - 标准 DP:    all-reduce 梯度 (Megatron DDP)
  - 优化版本:   reduce-scatter 梯度 + all-gather 参数 (Megatron 分布式优化器)

  🔗 K8s 中的分布式:
  - 每个 Pod 是一个 worker
  - Headless Service 提供 Pod 间网络发现
  - MASTER_ADDR 指向 rank-0 Pod
  - torchrun 通过 c10d rendezvous 协调启动

  下一步: 体验张量并行 (TP) — 切分模型权重!
""", rank)


def main():
    rank, world_size, local_rank, device = demo_distributed_init()
    dataloader, sampler = demo_data_parallel_split(rank, world_size, device)
    ddp_model = demo_ddp_model(rank, world_size, device)
    demo_training_loop(rank, world_size, device, ddp_model, dataloader, sampler)
    demo_gradient_sync_detail(rank, world_size, device, ddp_model, dataloader)
    demo_model_consistency_check(rank, world_size, device, ddp_model)
    print_summary(rank, world_size)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

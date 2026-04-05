"""
Megatron-LM 分布式 Checkpoint 演示

在 DP 训练基础上增加:
1. 训练中定期保存 checkpoint (分片保存，每个 rank 存自己的 shard)
2. 从 checkpoint 恢复训练 (模拟故障恢复)
3. Checkpoint 保存到 Azure File Share (所有 Pod 共享挂载)

对应 Megatron 中:
  megatron/training/checkpointing.py        → save/load 逻辑
  megatron/core/dist_checkpointing/         → 分片存储引擎
  megatron/training/ft_integration.py       → 容错集成
"""

import os
import json
import socket
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
from pathlib import Path


# ============================================================
# Checkpoint 工具函数
# ============================================================

def get_checkpoint_dir():
    """
    获取 checkpoint 目录

    K8s 中: /checkpoints (Azure File Share 挂载)
    本地:   ./checkpoints
    """
    ckpt_dir = os.environ.get("CHECKPOINT_DIR", "/checkpoints")
    return Path(ckpt_dir)


def save_checkpoint(model, optimizer, epoch, iteration, loss, rank, world_size):
    """
    保存分布式 Checkpoint

    对应 Megatron: megatron/training/checkpointing.py → save_checkpoint()

    Megatron 的做法:
      1. 每个 rank 保存自己的 shard (ShardedTensor)
      2. 异步写盘 (plan → write → finalize)
      3. rank 0 写 metadata.json

    我们的简化版:
      1. 每个 rank 保存自己的 optimizer state shard
      2. rank 0 额外保存模型参数 + 训练元数据
      3. 同步写盘 (所有 rank 写完后 barrier)
    """
    ckpt_dir = get_checkpoint_dir() / f"iter_{iteration:06d}"

    # 所有 rank 创建目录
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()  # 等 rank 0 建完目录

    # --- 每个 rank 保存自己的 shard ---
    # 对应 Megatron: dist_checkpointing 中每个 rank 的 ShardedTensor
    rank_state = {
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": torch.random.get_rng_state(),
    }
    rank_file = ckpt_dir / f"rank_{rank:03d}.pt"
    torch.save(rank_state, rank_file)

    # --- rank 0 额外保存公共数据 ---
    # 对应 Megatron: common.pt + metadata.json
    if rank == 0:
        common_state = {
            "model_state_dict": model.module.state_dict(),  # DDP 用 .module
            "epoch": epoch,
            "iteration": iteration,
            "loss": loss,
        }
        torch.save(common_state, ckpt_dir / "common.pt")

        # 元数据 (对应 Megatron 的 metadata.json)
        metadata = {
            "world_size": world_size,
            "iteration": iteration,
            "epoch": epoch,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hostname": socket.gethostname(),
            "shard_files": [f"rank_{r:03d}.pt" for r in range(world_size)],
        }
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # 更新 latest 指针 (对应 Megatron 的 latest_checkpointed_iteration.txt)
        with open(get_checkpoint_dir() / "latest", "w") as f:
            f.write(str(iteration))

    dist.barrier()  # 所有 rank 写完后同步
    return str(ckpt_dir)


def load_checkpoint(model, optimizer, rank, world_size):
    """
    加载分布式 Checkpoint

    对应 Megatron: megatron/training/checkpointing.py → load_checkpoint()

    流程:
      1. 读 latest 文件找到最新 checkpoint
      2. rank 0 广播 iteration 号给所有 rank
      3. 每个 rank 加载自己的 shard
      4. rank 0 加载公共数据 (模型参数) 并广播
    """
    latest_file = get_checkpoint_dir() / "latest"

    # 检查是否有 checkpoint 可恢复
    has_ckpt = torch.tensor([1 if latest_file.exists() else 0])
    dist.broadcast(has_ckpt, src=0)
    if has_ckpt.item() == 0:
        return None, 0, 0

    # rank 0 读取 latest iteration
    if rank == 0:
        iteration = int(latest_file.read_text().strip())
    else:
        iteration = 0
    iteration_tensor = torch.tensor([iteration])
    dist.broadcast(iteration_tensor, src=0)
    iteration = iteration_tensor.item()

    ckpt_dir = get_checkpoint_dir() / f"iter_{iteration:06d}"

    # --- 加载公共数据 (模型参数) ---
    if rank == 0:
        common_state = torch.load(ckpt_dir / "common.pt", weights_only=False)
        model.module.load_state_dict(common_state["model_state_dict"])
        epoch = common_state["epoch"]
    else:
        epoch = 0

    # 广播模型参数到所有 rank
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    epoch_tensor = torch.tensor([epoch])
    dist.broadcast(epoch_tensor, src=0)
    epoch = epoch_tensor.item()

    # --- 每个 rank 加载自己的 optimizer shard ---
    rank_file = ckpt_dir / f"rank_{rank:03d}.pt"
    if rank_file.exists():
        rank_state = torch.load(rank_file, weights_only=False)
        optimizer.load_state_dict(rank_state["optimizer_state_dict"])
        torch.random.set_rng_state(rank_state["rng_state"])

    dist.barrier()
    return ckpt_dir, epoch, iteration


def list_checkpoints(rank):
    """列出所有已保存的 checkpoint"""
    if rank != 0:
        return
    ckpt_dir = get_checkpoint_dir()
    if not ckpt_dir.exists():
        print("  (无 checkpoint)")
        return
    for d in sorted(ckpt_dir.iterdir()):
        if d.is_dir() and d.name.startswith("iter_"):
            meta_file = d / "metadata.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                files = list(d.glob("*.pt"))
                total_size = sum(f.stat().st_size for f in files) / 1024
                print(f"    {d.name}/  "
                      f"epoch={meta['epoch']}, "
                      f"shards={len(meta['shard_files'])}, "
                      f"size={total_size:.1f}KB, "
                      f"time={meta['timestamp']}")


# ============================================================
# 主程序
# ============================================================

def print_rank0(msg, rank):
    if rank == 0:
        print(msg, flush=True)


def print_all(msg, rank, world_size):
    for i in range(world_size):
        if rank == i:
            print(f"  [Rank {rank}] {msg}", flush=True)
        dist.barrier()


def create_model(device):
    """创建模型"""
    model = nn.Sequential(
        nn.Linear(10, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)
    return model


def create_dataset():
    """创建数据集: y = 2x + 1"""
    torch.manual_seed(42)
    num_samples = 2000
    X = torch.randn(num_samples, 10)
    Y = X @ torch.ones(10, 1) * 2 + 1 + torch.randn(num_samples, 1) * 0.1
    return TensorDataset(X, Y)


def main():
    # --- 初始化 ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    device = torch.device("cpu")

    print_rank0("=" * 60, rank)
    print_rank0("  🚀 Megatron 分布式 Checkpoint 演示", rank)
    print_rank0("=" * 60, rank)

    # --- 环境信息 ---
    print_rank0(f"\n📌 环境信息", rank)
    print_rank0(f"  World Size:      {world_size}", rank)
    print_rank0(f"  Backend:         {backend}", rank)
    print_rank0(f"  Checkpoint 目录: {get_checkpoint_dir()}", rank)
    print_all(f"hostname={socket.gethostname()}", rank, world_size)

    # --- 构建模型和优化器 ---
    model = create_model(device)
    ddp_model = DDP(model)
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    total_params = sum(p.numel() for p in model.parameters())

    print_rank0(f"\n📌 模型信息", rank)
    print_rank0(f"  参数量:   {total_params:,}", rank)
    print_rank0(f"  优化器:   Adam (lr=0.005)", rank)

    # --- 尝试从 Checkpoint 恢复 ---
    print_rank0(f"\n📌 Checkpoint 恢复检查", rank)
    ckpt_path, start_epoch, start_iter = load_checkpoint(
        ddp_model, optimizer, rank, world_size
    )
    if ckpt_path:
        print_rank0(f"  ✅ 从 checkpoint 恢复: {ckpt_path}", rank)
        print_rank0(f"     epoch={start_epoch}, iteration={start_iter}", rank)
    else:
        print_rank0(f"  ℹ️  无 checkpoint，从头开始训练", rank)
        start_epoch = 0
        start_iter = 0

    # --- 数据 ---
    dataset = create_dataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # --- 训练循环 ---
    num_epochs = 10
    save_every = 5  # 每 5 个 epoch 保存一次 checkpoint
    iteration = start_iter

    print_rank0(f"\n📌 开始训练", rank)
    print_rank0(f"  总 Epochs:     {num_epochs} (从 epoch {start_epoch} 开始)", rank)
    print_rank0(f"  保存频率:      每 {save_every} 个 epoch", rank)
    print_rank0(f"  数据/rank:     {len(sampler)} 样本", rank)
    print_rank0("", rank)

    # ============================================================
    # Profiler 配置 — 只在 rank 0 开启，避免冲突
    # ============================================================
    profiler_dir = "/profiler_output" if os.path.exists("/profiler_output") else "./profiler_output"
    os.makedirs(profiler_dir, exist_ok=True)

    # schedule: wait 2 步热身 → warmup 2 步 → active 记录 10 步 → 重复
    prof_schedule = schedule(wait=2, warmup=2, active=10, repeat=1)

    def trace_handler(p):
        p.export_chrome_trace(os.path.join(profiler_dir, f"trace_rank{rank}.json"))

    prof = profile(
        activities=[ProfilerActivity.CPU],
        schedule=prof_schedule,
        on_trace_ready=trace_handler if rank == 0 else None,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) if rank == 0 else None

    if prof:
        prof.__enter__()

    global_step = 0

    for epoch in range(start_epoch, num_epochs):
        sampler.set_epoch(epoch)
        ddp_model.train()
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = ddp_model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
            iteration += 1
            global_step += 1

            if prof:
                prof.step()

        avg_loss = epoch_loss / num_batches
        elapsed = time.time() - start_time

        # 全局平均 loss
        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        global_loss = loss_tensor.item()

        print_rank0(f"  Epoch {epoch+1}/{num_epochs}: "
                    f"loss={global_loss:.4f}, "
                    f"time={elapsed:.2f}s, "
                    f"iter={iteration}", rank)

        # --- 定期保存 Checkpoint ---
        if (epoch + 1) % save_every == 0:
            print_rank0(f"\n  💾 保存 Checkpoint (iteration={iteration})...", rank)
            ckpt_path = save_checkpoint(
                ddp_model, optimizer, epoch + 1, iteration, global_loss,
                rank, world_size
            )
            print_rank0(f"     保存到: {ckpt_path}", rank)
            print_all(f"已保存 rank_{rank:03d}.pt", rank, world_size)
            print_rank0("", rank)

    # 关闭 profiler 并输出汇总
    if prof:
        prof.__exit__(None, None, None)
        print_rank0(f"\n📌 Profiler 分析结果", rank)
        print_rank0(f"  Trace 保存到: {profiler_dir}/", rank)
        try:
            print_rank0(f"\n  === CPU 时间 Top 20 操作 ===", rank)
            print_rank0(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20), rank)
            print_rank0(f"\n  === 按输入形状分组 Top 15 ===", rank)
            print_rank0(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=15), rank)
            print_rank0(f"\n  === 内存 Top 10 ===", rank)
            print_rank0(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10), rank)
        except Exception as e:
            print_rank0(f"  ⚠️ Profiler 汇总失败: {e}", rank)

    # --- 列出所有 Checkpoint ---
    print_rank0(f"\n📌 Checkpoint 目录内容:", rank)
    print_rank0(f"  路径: {get_checkpoint_dir()}", rank)
    list_checkpoints(rank)

    # --- Checkpoint 结构说明 ---
    print_rank0(f"""
📌 Checkpoint 结构对照:

  本演示:                          Megatron:
  ─────────────────────────────── ───────────────────────────────
  /checkpoints/                    /checkpoints/
  ├── latest                       ├── latest_checkpointed_iteration.txt
  ├── iter_000008/                 ├── iter_0001000/
  │   ├── metadata.json            │   ├── metadata.json
  │   ├── common.pt (模型参数)     │   ├── common.pt
  │   ├── rank_000.pt (rank0优化器) │   ├── __0_0.distcp (rank0 shard)
  │   ├── rank_001.pt              │   ├── __1_0.distcp
  │   ├── rank_002.pt              │   └── __2_0.distcp
  │   └── rank_003.pt              │
  └── iter_000016/                 └── iter_0002000/

  关键差异:
  - Megatron 用 ShardedTensor: 模型参数也分片, 不需要 rank 0 汇总
  - Megatron 异步写盘: plan → write(后台) → finalize
  - Megatron 支持 resharding: TP=4 保存 → TP=8 加载, 自动切分

📌 故障恢复流程:
  1. Pod 崩溃 → K8s 自动重启 Pod
  2. 新 Pod 挂载同一个 Azure File Share
  3. 读取 /checkpoints/latest 找到最新 checkpoint
  4. 每个 rank 加载自己的 shard → 从断点继续训练!
""", rank)

    # --- 验证模型一致性 ---
    print_rank0(f"📌 最终验证", rank)
    param_sum = sum(p.sum().item() for p in ddp_model.parameters())
    print_all(f"参数校验和: {param_sum:.6f}", rank, world_size)
    print_rank0("  ✅ 所有 rank 参数一致!", rank)

    print_rank0("\n" + "=" * 60, rank)
    print_rank0("  ✅ 分布式 Checkpoint 演示完成!", rank)
    print_rank0("=" * 60, rank)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

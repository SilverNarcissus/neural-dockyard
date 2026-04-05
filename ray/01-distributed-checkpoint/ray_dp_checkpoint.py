"""
Ray Train 分布式训练 + Checkpoint 演示

功能与 Megatron 版 dp_checkpoint_training.py 完全对等:
1. 多 worker 数据并行训练
2. 定期保存 checkpoint (模型 + 优化器 + epoch)
3. 从 checkpoint 恢复训练 (故障恢复)

对比 Megatron(手动torch.distributed) vs Ray(框架托管):
┌──────────────────────────────────────────────────────────────┐
│  Megatron / 手动 DDP                │  Ray Train              │
├──────────────────────────────────────────────────────────────┤
│  手动 dist.init_process_group()     │  Ray 自动初始化          │
│  手动 DistributedSampler            │  prepare_data_loader()  │
│  手动 DDP(model)                    │  prepare_model()        │
│  手动 torch.save() 每个 rank        │  ray.train.report()     │
│  手动 load + broadcast              │  resume_from_checkpoint │
│  手动 env vars (RANK/WORLD_SIZE)    │  Ray 自动注入           │
│  K8s: 自己写 StatefulSet + Service  │  K8s: RayJob/RayCluster │
│  容错: 自己实现 signal handler      │  Ray 内建容错            │
└──────────────────────────────────────────────────────────────┘

核心优势: Ray 把分布式训练的"脚手架"全部自动化了，
你只需要写单机训练逻辑，Ray 负责分布式、checkpoint、调度。
"""

import os
import json
import socket
import tempfile
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig, Checkpoint


# ============================================================
# 1. 模型定义 (与 Megatron 版完全一样)
# ============================================================
def create_model():
    return nn.Sequential(
        nn.Linear(10, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


def create_dataset():
    """y = 2x + 1 回归数据集"""
    torch.manual_seed(42)
    num_samples = 2000
    X = torch.randn(num_samples, 10)
    Y = X @ torch.ones(10, 1) * 2 + 1 + torch.randn(num_samples, 1) * 0.1
    return TensorDataset(X, Y)


# ============================================================
# 2. 训练函数 — 每个 Ray Worker 执行这个函数
# ============================================================
# 对比:
#   Megatron: 需要手动 dist.init_process_group, DDP, DistributedSampler
#   Ray: 调 prepare_model() 和 prepare_data_loader()，其余自动
#
def train_func(config):
    """
    Ray Worker 的训练入口

    Ray 在幕后做了:
    1. 启动 N 个 worker 进程
    2. 每个 worker 自动 init_process_group (backend=gloo/nccl)
    3. prepare_model() 自动包装 DDP
    4. prepare_data_loader() 自动加 DistributedSampler
    5. ray.train.report() 自动收集 metrics + 保存 checkpoint
    """
    epochs = config["epochs"]
    save_every = config["save_every"]
    lr = config["lr"]
    batch_size = config["batch_size"]

    # --- 获取分布式信息 ---
    # Megatron: 手动读 os.environ["RANK"] / os.environ["WORLD_SIZE"]
    # Ray: 自动注入
    world_size = ray.train.get_context().get_world_size()
    rank = ray.train.get_context().get_world_rank()

    print(f"[Rank {rank}] Worker 启动: hostname={socket.gethostname()}, "
          f"world_size={world_size}")

    # --- 创建模型 ---
    model = create_model()
    # Megatron: model = DDP(model)
    # Ray: 自动 DDP 包装
    model = ray.train.torch.prepare_model(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # --- 从 Checkpoint 恢复 ---
    # Megatron: 手动读 latest 文件 → load state dict → broadcast
    # Ray: 自动检测并加载
    start_epoch = 0
    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as ckpt_dir:
            state = torch.load(os.path.join(ckpt_dir, "checkpoint.pt"),
                               weights_only=False)
            model.module.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            start_epoch = state["epoch"] + 1
            if rank == 0:
                print(f"✅ 从 checkpoint 恢复: epoch={start_epoch}")
    else:
        if rank == 0:
            print("ℹ️  无 checkpoint，从头开始训练")

    # --- 数据加载 ---
    dataset = create_dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Megatron: 手动 DistributedSampler + 手动 set_epoch
    # Ray: 自动处理数据切分
    dataloader = ray.train.torch.prepare_data_loader(dataloader)

    total_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  🚀 Ray 分布式训练开始")
        print(f"{'='*60}")
        print(f"  Workers:    {world_size}")
        print(f"  参数量:     {total_params:,}")
        print(f"  Epochs:     {epochs} (从 {start_epoch} 开始)")
        print(f"  保存频率:   每 {save_every} 个 epoch")
        print(f"  数据/worker: {len(dataloader.dataset) // world_size} 样本")
        print()

    # --- 训练循环 ---
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()  # Ray DDP 自动 all-reduce 梯度
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        elapsed = time.time() - start_time

        if rank == 0:
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"loss={avg_loss:.4f}, time={elapsed:.2f}s")

        # --- 保存 Checkpoint ---
        # Megatron: 手动 torch.save() 每个 rank 的 shard + metadata.json
        # Ray: 统一通过 ray.train.report() 上报 metrics + checkpoint
        #      Ray 自动管理 checkpoint 存储、版本、清理
        checkpoint_to_report = None
        if (epoch + 1) % save_every == 0:
            # Ray 需要目录在 report 时仍然存在
            # 用 tempfile.mkdtemp() 而非 TemporaryDirectory context manager
            tmpdir = tempfile.mkdtemp()
            state = {
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
                "world_size": world_size,
            }
            torch.save(state, os.path.join(tmpdir, "checkpoint.pt"))

            metadata = {
                "epoch": epoch + 1,
                "loss": round(avg_loss, 4),
                "world_size": world_size,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(os.path.join(tmpdir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            checkpoint_to_report = Checkpoint.from_directory(tmpdir)

            if rank == 0:
                print(f"  💾 Checkpoint saved (epoch={epoch+1})")

        # 上报 metrics (所有 epoch) + checkpoint (仅保存 epoch)
        ray.train.report(
            {"loss": avg_loss, "epoch": epoch + 1},
            checkpoint=checkpoint_to_report,
        )

    # --- 训练完成 ---
    if rank == 0:
        # 验证模型一致性
        param_sum = sum(p.sum().item() for p in model.parameters())
        print(f"\n📌 训练完成验证")
        print(f"  参数校验和: {param_sum:.6f}")

        print(f"""
{'='*60}
  ✅ Ray 分布式训练完成!
{'='*60}

  对比 Megatron 手动 DDP:
  ┌────────────────────────────────────────────────────┐
  │  你不需要写的代码 (Ray 自动处理):                     │
  │  ✗ dist.init_process_group()                        │
  │  ✗ DistributedSampler + set_epoch()                 │
  │  ✗ DDP(model)                                       │
  │  ✗ RANK/WORLD_SIZE/MASTER_ADDR 环境变量              │
  │  ✗ Headless Service + StatefulSet (K8s)             │
  │  ✗ 手动 torch.save/load 每个 rank 的 shard          │
  │  ✗ 信号处理 / 故障检测 / 进程重启                     │
  ├────────────────────────────────────────────────────┤
  │  你只需要写:                                         │
  │  ✓ 模型定义                                         │
  │  ✓ 训练循环                                         │
  │  ✓ ray.train.report(metrics, checkpoint)            │
  └────────────────────────────────────────────────────┘
""")


# ============================================================
# 3. 启动分布式训练
# ============================================================
if __name__ == "__main__":
    # 初始化 Ray
    # K8s 中: 自动连接到 Ray Cluster (通过 RAY_ADDRESS 环境变量)
    # 本地: 启动单机 Ray 集群
    ray.init()

    num_workers = int(os.environ.get("NUM_WORKERS", "4"))

    print(f"🎯 Ray 分布式训练配置:")
    print(f"   Workers:  {num_workers}")
    print(f"   Backend:  {'nccl' if torch.cuda.is_available() else 'gloo'}")
    print()

    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            "epochs": 6,
            "save_every": 2,      # 每 2 个 epoch 保存 checkpoint
            "lr": 0.005,
            "batch_size": 64,
        },
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=torch.cuda.is_available(),
        ),
        run_config=RunConfig(
            name="ray-dp-checkpoint-demo",
            storage_path="/shared-storage/ray_results",
            checkpoint_config=CheckpointConfig(
                num_to_keep=3,  # 最多保留 3 个 checkpoint (自动清理旧的!)
            ),
        ),
    )

    # 运行训练
    result = trainer.fit()

    # 输出结果
    print(f"\n📊 训练结果:")
    print(f"   最终 loss: {result.metrics.get('loss', 'N/A')}")
    print(f"   最新 checkpoint: {result.checkpoint}")
    print(f"   日志目录: {result.path}")

    ray.shutdown()

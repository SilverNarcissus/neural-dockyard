# 🚢 Neural Dockyard

> AI 训练基础设施的实验船坞 — 在 Kubernetes 上部署和探索分布式训练框架

**Dock** 的双关：Docker 容器 + K8s 编排 + 船坞（造大船/大模型的地方）

## 📁 项目结构

```
neural-dockyard/
├── 步骤文档.md
└── megatron/
    ├── standalone/                        # 单机运行
    │   ├── megatron_hello_world.py
    │   ├── Dockerfile
    │   └── k8s-job.yaml
    └── distributed/                       # 分布式训练
        ├── 01-data-parallel/              # DP 数据并行
        │   ├── dp_distributed_training.py
        │   ├── Dockerfile
        │   └── k8s-dp-distributed.yaml
        └── 02-dp-checkpoint/              # DP + 分布式 Checkpoint
            ├── dp_checkpoint_training.py
            ├── Dockerfile
            └── k8s-dp-checkpoint.yaml
```

## 🎯 学习路线

- [x] **Megatron-LM** — 5D并行 / 分布式Checkpoint / 融合算子 / AKS部署
- [ ] **DeepSpeed** — ZeRO优化 / Offload策略
- [ ] **vLLM** — 推理引擎 / PagedAttention
- [ ] **Ray** — 分布式计算框架 / Train + Serve

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| 容器化 | Docker / ACR |
| 编排调度 | Kubernetes (AKS) |
| 训练框架 | Megatron-LM / PyTorch |
| 通信 | NCCL / Gloo |
| 云平台 | Azure |

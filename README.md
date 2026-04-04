# 🚢 Neural Dockyard

> AI 训练基础设施的实验船坞 — 在 Kubernetes 上部署和探索分布式训练框架

**Dock** 的双关：Docker 容器 + K8s 编排 + 船坞（造大船/大模型的地方）

## 📁 项目结构

```
neural-dockyard/
└── megatron/         # Megatron-LM 分布式训练框架学习
    ├── megatron_hello_world.py   # AI Infra 核心概念演示
    ├── Dockerfile                # PyTorch 容器化
    ├── k8s-job.yaml              # K8s Job 部署
    └── 步骤文档.md                # 完整部署步骤记录
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

# Training Operator — K8s 分布式训练任务管理

## 这是什么？

一个最小化的 K8s Operator，用于管理分布式训练任务。用户提交一个 `TrainingJob` CR，Operator 自动创建所需的 Service + StatefulSet + 环境变量注入。

**对比手写 YAML**（如 `02-dp-checkpoint/k8s-dp-checkpoint.yaml`）：
- 手写：需要 ~140 行 YAML（Service + StatefulSet + 环境变量 + 存储）
- Operator：用户只写 ~20 行 TrainingJob YAML，其余全自动

## 架构

```
用户                        Operator                         K8s
 │                            │                               │
 │  kubectl apply             │                               │
 │  TrainingJob CR ──────────►│                               │
 │                            │  watch: 新CR!                 │
 │                            │  reconcile:                   │
 │                            │    1. 创建 Headless Service ──►│
 │                            │    2. 创建 StatefulSet ───────►│── 创建 Pod 0~3
 │                            │    3. 注入 RANK, MASTER_ADDR   │── DNS 注册
 │                            │    4. 挂载 Checkpoint PVC      │── 挂载存储
 │                            │    5. 更新 status ◄────────────│── Pod Running
 │  kubectl get tj            │                               │
 │  ◄─────────────────────────│                               │
 │  NAME             WORKERS  PHASE    READY                  │
 │  megatron-dp-demo 4        Running  4                      │
 │                            │                               │
 │  kubectl delete tj         │                               │
 │  ──────────────────────────►  OwnerRef 级联删除 ───────────►│── 清理所有资源
```

## 目录结构

```
training-operator/
├── main.go                    # 入口: 连接K8s, 启动Controller
├── api/v1/types.go            # CRD 类型定义 (TrainingJob Spec/Status)
├── controller/controller.go   # Controller: Watch → Reconcile → Create/Delete
├── config/
│   ├── crd.yaml               # CRD 注册 YAML (kubectl apply -f)
│   └── example-job.yaml       # 示例训练任务
├── go.mod
└── README.md
```

## 核心概念

### CRD (Custom Resource Definition)
向 K8s 注册新的资源类型 `TrainingJob`，之后用户就可以 `kubectl get tj`。

### Controller 控制循环
```
while true:
    event = watch(TrainingJob)
    current_state = 检查实际子资源 (Service? StatefulSet? Pod Ready?)
    desired_state = 读取 CR 的 spec
    if current_state != desired_state:
        reconcile(创建/更新/删除子资源)
        update CR status
```

### OwnerReference 级联删除
创建子资源时设置 OwnerReference 指向 TrainingJob CR。
删除 TrainingJob 时，K8s GC 自动清理所有子资源（Service、StatefulSet、Pod）。

## 使用方式

```bash
# 1. 注册 CRD
kubectl apply -f config/crd.yaml

# 2. 启动 Operator
go run main.go

# 3. 提交训练任务
kubectl apply -f config/example-job.yaml

# 4. 查看状态
kubectl get tj
kubectl describe tj megatron-dp-demo

# 5. 删除任务 (自动清理)
kubectl delete tj megatron-dp-demo
```

## 和手写 YAML 的对比

| | 手写 YAML | Operator |
|---|---|---|
| 用户需要写 | ~140 行 (Service+StatefulSet+Env+PV) | ~20 行 (TrainingJob CR) |
| RANK/MASTER_ADDR | 手动配置 | 自动注入 |
| 扩缩容 | 改 YAML 重新 apply | 改 workers 数字 |
| 状态监控 | kubectl get pods | kubectl get tj (Phase/Ready) |
| 清理资源 | 手动删 Service + StatefulSet | 删 CR 自动级联清理 |
| 故障恢复 | StatefulSet 重建 Pod | 同上 + Controller 更新状态 |

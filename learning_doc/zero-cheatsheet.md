# ZeRO 速记卡

## 一、训练显存构成（7B 模型为例）

| 组件 | 精度 | 大小 | 说明 |
|------|------|------|------|
| Model Params (M) | BF16 | 14 GB | 权重 + 偏置 |
| Gradients (G) | FP32 | 28 GB | 和参数一一对应，同形状 |
| Optimizer State (O) | FP32 | 56 GB | Adam 的 m(一阶动量) + v(二阶动量)，各 28GB |
| **总计** | | **98 GB** | Optimizer 占大头(57%) |

> Adam 的 m 和 v 必须 FP32（累加操作精度敏感），所以 O = 参数量 × 4bytes × 2

## 二、各模式显存分布（4 GPUs）

```
DDP:     每卡存 M全量 + G全量 + O全量  = 98 GB/卡   冗余 75%
ZeRO-1:  每卡存 M全量 + G全量 + O/4    = 56 GB/卡   ↓ 43%
ZeRO-2:  每卡存 M全量 + G/4   + O/4    = 28 GB/卡   ↓ 71%
ZeRO-3:  每卡存 M/4   + G/4   + O/4    = 24.5 GB/卡 ↓ 75%
```

## 三、各模式通信方式

### DDP（无 ZeRO）

```
反向传播 → AllReduce(梯度) → Optimizer全量更新
          = ReduceScatter + AllGather
          每层梯度出来就可以发，和反向计算完美重叠
通信量: 2× model_size
```

### ZeRO-1（分片 Optimizer）

```
反向传播 → ReduceScatter(梯度)  → 每人只更新 1/N Optimizer → AllGather(新参数)
          和反向重叠 ✅            只做 1/N 工作              必须等Optimizer完成 ⚠️
通信量: 2× model_size（和DDP一样）
代价: AllGather参数无法和计算重叠，慢 1~5%
```

> **核心技巧**: 把 AllGather 从 Optimizer 之前（拼梯度）挪到之后（拼参数），对象换了但大小一样 → "免费午餐"

### ZeRO-2（+ 分片 Gradients）

```
和 ZeRO-1 通信完全一样！
唯一区别: ReduceScatter 后立刻 del 不属于自己的梯度
通信量: 2× model_size
代价: 和 ZeRO-1 一样，也是"免费午餐"
```

### ZeRO-3（+ 分片 Model Params）

```
前向每层: AllGather(该层参数) → 计算 → 丢弃参数
反向每层: AllGather(该层参数) → 计算梯度 → ReduceScatter(梯度) → 丢弃参数和梯度
Optimizer: 更新 1/N（不需要额外通信，下轮前向会自己 gather）
通信量: 3× model_size（多了 50%！）
代价: 前向+反向每层都要等通信，慢 15~30%
```

### 通信量汇总

| 模式 | 通信操作 | 通信量 | 速度代价 |
|------|---------|--------|---------|
| DDP | AllReduce(梯度) | 2× | 基准 |
| ZeRO-1 | RS(梯度) + AG(参数) | 2× | 1~5% |
| ZeRO-2 | RS(梯度) + AG(参数) | 2× | 1~5% |
| ZeRO-3 | RS(梯度) + AG(参数)×2 | 3× | 15~30% |

> RS = ReduceScatter, AG = AllGather

### 选择原则

```
放得下 → ZeRO-1/2（免费午餐，优先 ZeRO-2）
放不下 → ZeRO-3（唯一选择，接受速度代价）
```

## 四、AllReduce = ReduceScatter + AllGather

```
4 GPU，每个有完整梯度 [a, b, c, d]

ReduceScatter: 每人得到一个分片的聚合
  GPU0→[Σa]  GPU1→[Σb]  GPU2→[Σc]  GPU3→[Σd]
  通信量 ≈ 1× model_size

AllGather: 每人拼回完整结果
  每个GPU → [Σa, Σb, Σc, Σd]
  通信量 ≈ 1× model_size
```

## 五、为什么不能逐层更新 Optimizer？

理论上每层梯度出来就能更新，但 **gradient clipping 需要全局梯度范数**：

```
global_norm = sqrt(Σ ‖grad_i‖²)  ← 必须等所有层算完
```

所以 Optimizer 必须等所有梯度就绪 → AllGather 参数无法和反向重叠。

## 六、业界主流方案（2024-2025）

**没有人只用 ZeRO，实际是 3D/4D 混合并行 + ZeRO：**

| 维度 | 解决什么问题 | 通信特点 |
|------|------------|---------|
| TP（张量并行） | 单层参数太大放不下单卡 | 机内 NVLink，通信频繁但带宽高 |
| PP（流水线并行） | 模型太深层数太多 | 机间，只传激活值，数据量小 |
| DP（数据并行） | 训练数据要并行 | 机间，梯度同步，用 ZeRO 优化 |
| EP（专家并行） | MoE 专家数量太多 | 跨卡分配不同专家 |

### 代表方案

```
DeepSeek-V3 (671B MoE):  PP×16 + EP + ZeRO-1        ← 不用TP! MoE每token只激活37B
LLaMA-3 (405B Dense):    TP×8(机内) + PP + DP + FSDP ← 经典3D并行
MiniMax-01 (456B MoE):   TP + PP + EP + DP           ← 四维并行
ByteDance Seed:          AsyncHZP(异步分层ZeRO)       ← 自研优化
```

### 典型 3D 并行布局（128卡 = TP8 × PP4 × DP4）

```
机器1 [8卡NVLink] → TP组, PP Stage 0, DP Replica 0
机器2 [8卡NVLink] → TP组, PP Stage 1, DP Replica 0
机器3 [8卡NVLink] → TP组, PP Stage 2, DP Replica 0
机器4 [8卡NVLink] → TP组, PP Stage 3, DP Replica 0
机器5~8           → 同上, DP Replica 1
机器9~16          → DP Replica 2, 3

TP 放机内（带宽高），PP/DP 放机间（数据量小或可优化）
```

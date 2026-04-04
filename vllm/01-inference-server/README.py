"""
vLLM 推理服务 Hello World

在 K8s 上部署 vLLM 推理服务，体验:
1. OpenAI 兼容 API (与 ChatGPT API 格式一致)
2. PagedAttention KV Cache 管理
3. Continuous Batching (连续批处理)
4. 模型加载与推理

使用 facebook/opt-125m (125M 参数，CPU 可跑)

vLLM 核心架构:
┌─────────────────────────────────────────┐
│  Client (curl / Python)                  │
│  POST /v1/completions                    │
├─────────────────────────────────────────┤
│  vLLM Engine                             │
│  ├── Scheduler (连续批处理调度器)         │
│  │   └── 动态插入/移除请求，不等batch完成  │
│  ├── PagedAttention (分页注意力)          │
│  │   └── KV Cache 按 block 分配，零浪费  │
│  ├── Worker (模型执行)                   │
│  │   └── 支持 TP 多卡推理                │
│  └── Tokenizer                           │
├─────────────────────────────────────────┤
│  Model: facebook/opt-125m                │
└─────────────────────────────────────────┘
"""

# 这个文件是概念说明，实际的 vLLM 服务由 vllm serve 命令启动
# 见 Dockerfile 和 k8s-vllm-server.yaml

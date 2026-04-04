"""
推理服务器 — 模拟 vLLM 的 OpenAI 兼容 API

用 HuggingFace transformers + FastAPI 实现，注释对照 vLLM 真实架构。
部署在 K8s 上，通过 curl 调用。

vLLM 核心架构对照:
┌───────────────────────────────────────────────────────┐
│  本演示 (简化版)            │  vLLM (生产版)              │
├───────────────────────────────────────────────────────┤
│  FastAPI server             │  AsyncLLMEngine             │
│  transformers.generate()    │  PagedAttention + Worker    │
│  顺序处理请求               │  Continuous Batching        │
│  KV Cache 由 HF 管理        │  Block Manager 分页管理     │
│  单进程                     │  多 Worker + TP 分布式      │
│  CPU/GPU 自动选择           │  GPU 优先 + 自定义 kernel   │
└───────────────────────────────────────────────────────┘
"""

import os
import time
import uuid
import json
import asyncio
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn


# ============================================================
# 1. 模型加载 — 对应 vLLM 的 ModelLoader
# ============================================================
# vLLM 做了什么不同的事:
#   - 用 PagedAttention 替换标准 Attention，让 KV Cache 按 block 分配
#   - 支持 TP (张量并行)，把模型切分到多卡
#   - 支持 FP8/AWQ/GPTQ 量化加载
#   - 支持 prefix caching (相同前缀共享 KV Cache blocks)

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "256"))

print(f"Loading model: {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
).to(DEVICE)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

total_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded: {total_params/1e6:.0f}M params, device={DEVICE}")


# ============================================================
# 2. KV Cache 模拟 — 对应 vLLM 的 BlockManager + PagedAttention
# ============================================================
# vLLM 的核心创新: PagedAttention
#
# 传统方式:
#   为每个请求预分配 max_seq_len 大小的连续 KV Cache
#   → 大量内存浪费 (短请求也占满空间)
#   → 内存碎片化 (已结束请求留下 "空洞")
#
# vLLM PagedAttention:
#   KV Cache 切成固定大小的 block (如 16 tokens/block)
#   block 不需要连续，通过 block_table 映射
#   → 按需分配: 请求来一个 block 给一个 block
#   → 零浪费: 请求结束立即回收 block
#   → 共享: 相同前缀的请求共享 KV Cache blocks (prefix caching)
#
# 下面用简单的字典模拟:

@dataclass
class RequestState:
    """模拟 vLLM 的 SequenceGroup"""
    request_id: str
    prompt: str
    prompt_tokens: list = field(default_factory=list)
    generated_tokens: list = field(default_factory=list)
    max_tokens: int = 50
    temperature: float = 1.0
    finished: bool = False
    arrival_time: float = 0.0
    # vLLM 中还有: block_table, num_kv_blocks, prefix_hash 等


# ============================================================
# 3. 调度器 — 对应 vLLM 的 Scheduler
# ============================================================
# vLLM Continuous Batching:
#   传统: 一个 batch 的所有请求必须同时结束
#   vLLM: 每个 decode step 动态调整 batch
#     → 请求完成立即移除，新请求立即加入
#     → GPU 利用率最大化
#
# 本演示用简单队列模拟:

request_queue: asyncio.Queue = asyncio.Queue()
active_requests: dict[str, RequestState] = {}
stats = {"total_requests": 0, "total_tokens_generated": 0, "total_prompt_tokens": 0}


async def process_request(req: RequestState) -> dict:
    """
    处理单个推理请求

    vLLM 中这对应:
    1. Scheduler.schedule() — 选择要执行的 sequences
    2. Worker.execute_model() — 运行一步 decode
    3. Scheduler.update() — 更新 sequence 状态
    循环直到所有 sequence 生成 eos 或达到 max_tokens
    """
    input_ids = tokenizer.encode(req.prompt, return_tensors="pt").to(DEVICE)
    req.prompt_tokens = input_ids[0].tolist()

    start_time = time.time()

    with torch.no_grad():
        # vLLM 不用 model.generate()!
        # 它自己实现了 decode loop:
        #   1. 每步只生成 1 token (autoregressive)
        #   2. 用 PagedAttention kernel 查询 KV Cache
        #   3. 支持 continuous batching (多个请求同时 decode)
        #   4. 支持 speculative decoding (投机解码加速)
        outputs = model.generate(
            input_ids,
            max_new_tokens=req.max_tokens,
            temperature=max(req.temperature, 0.01),
            do_sample=req.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    elapsed = time.time() - start_time

    prompt_tokens = len(req.prompt_tokens)
    completion_tokens = len(generated_ids)
    tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0

    stats["total_requests"] += 1
    stats["total_tokens_generated"] += completion_tokens
    stats["total_prompt_tokens"] += prompt_tokens

    # 返回 OpenAI 格式响应
    return {
        "id": f"cmpl-{req.request_id}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{
            "text": generated_text,
            "index": 0,
            "finish_reason": "length" if completion_tokens >= req.max_tokens else "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        # 额外性能信息 (vLLM 也提供)
        "_performance": {
            "elapsed_seconds": round(elapsed, 3),
            "tokens_per_second": round(tokens_per_sec, 1),
            "device": DEVICE,
        }
    }


# ============================================================
# 4. API 层 — 对应 vLLM 的 OpenAI 兼容 entrypoint
# ============================================================
# vLLM 提供与 OpenAI API 完全兼容的接口:
#   /v1/completions      — 文本补全
#   /v1/chat/completions — 对话补全
#   /v1/models           — 模型列表
#   /health              — 健康检查
#   /v1/embeddings       — 向量化 (可选)

app = FastAPI(title="vLLM-style Inference Server")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    """对应 vLLM GET /v1/models"""
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "owned_by": "neural-dockyard",
            "permission": [],
            "_info": {
                "parameters": f"{total_params/1e6:.0f}M",
                "device": DEVICE,
                "max_model_len": MAX_MODEL_LEN,
            }
        }]
    }


@app.post("/v1/completions")
async def create_completion(request: dict):
    """
    对应 vLLM POST /v1/completions

    vLLM 的处理流程:
    1. 解析请求 → SamplingParams
    2. 加入 Scheduler 队列
    3. Continuous Batching: 每个 step 动态调度
    4. PagedAttention: 按需分配 KV Cache blocks
    5. 返回结果 (支持 streaming)
    """
    prompt = request.get("prompt", "Hello")
    max_tokens = min(request.get("max_tokens", 50), MAX_MODEL_LEN)
    temperature = request.get("temperature", 1.0)

    req = RequestState(
        request_id=str(uuid.uuid4())[:8],
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        arrival_time=time.time(),
    )

    result = await process_request(req)
    return JSONResponse(content=result)


@app.get("/stats")
async def get_stats():
    """服务器统计 — vLLM 也有类似的 metrics endpoint"""
    return {
        "total_requests": stats["total_requests"],
        "total_tokens_generated": stats["total_tokens_generated"],
        "total_prompt_tokens": stats["total_prompt_tokens"],
        "model": MODEL_NAME,
        "device": DEVICE,
    }


@app.get("/v1/concepts")
async def vllm_concepts():
    """额外的学习 endpoint — 解释 vLLM 核心概念"""
    return {
        "vllm_core_concepts": {
            "PagedAttention": {
                "what": "KV Cache 按固定大小 block 分配，不需要连续内存",
                "why": "消除内存碎片和浪费，支持更多并发请求",
                "how": "block_table 映射 logical block → physical block",
                "benefit": "内存利用率从 ~50% 提升到 ~95%",
            },
            "ContinuousBatching": {
                "what": "每个 decode step 动态调整 batch 中的请求",
                "why": "传统 static batch 中短请求要等长请求完成",
                "how": "Scheduler 每步决定哪些 sequence 执行/等待/抢占",
                "benefit": "吞吐量提升 2-4x",
            },
            "PrefixCaching": {
                "what": "相同前缀的请求共享 KV Cache blocks",
                "why": "系统提示 / few-shot 示例大量重复",
                "how": "hash(prefix_tokens) → 查找已有 blocks",
                "benefit": "TTFT (首 token 延迟) 大幅降低",
            },
            "SpeculativeDecoding": {
                "what": "用小模型快速猜多个 token，大模型验证",
                "why": "LLM decode 是 memory-bound，GPU 计算浪费",
                "how": "draft model 猜 K 个 token → target model 并行验证",
                "benefit": "延迟降低 1.5-3x，输出不变",
            },
        }
    }


# ============================================================
# 5. 启动
# ============================================================
if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════════════════════╗
║  🚀 vLLM-style Inference Server                     ║
║  Model:  {MODEL_NAME:<42} ║
║  Device: {DEVICE:<42} ║
║  Params: {total_params/1e6:.0f}M{' ' * 38}║
║  Port:   8000                                        ║
╠══════════════════════════════════════════════════════╣
║  Endpoints:                                          ║
║    GET  /health           — 健康检查                  ║
║    GET  /v1/models        — 模型列表                  ║
║    POST /v1/completions   — 文本补全 (OpenAI 兼容)    ║
║    GET  /stats            — 服务器统计                ║
║    GET  /v1/concepts      — vLLM 核心概念解释         ║
╚══════════════════════════════════════════════════════╝
""")
    uvicorn.run(app, host="0.0.0.0", port=8000)

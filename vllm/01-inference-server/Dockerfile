FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers fastapi uvicorn

# 预下载 GPT-2 模型
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('gpt2'); \
    AutoModelForCausalLM.from_pretrained('gpt2')"

WORKDIR /app
COPY inference_server.py .

EXPOSE 8000
CMD ["python", "inference_server.py"]

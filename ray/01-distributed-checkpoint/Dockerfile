FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir "ray[train,default]"

WORKDIR /app
COPY ray_dp_checkpoint.py .

CMD ["python", "ray_dp_checkpoint.py"]

FROM pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime

WORKDIR /app

COPY dp_distributed_training.py .
COPY dp_checkpoint_training.py .

# 默认运行 hello world，可通过 K8s command 覆盖
CMD ["python", "megatron_hello_world.py"]

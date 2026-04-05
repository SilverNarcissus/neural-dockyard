FROM pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime

WORKDIR /app
COPY dp_profiled_training.py .

CMD ["python", "dp_profiled_training.py"]

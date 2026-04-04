FROM pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime

WORKDIR /app

COPY dp_checkpoint_training.py .

CMD ["python", "dp_checkpoint_training.py"]

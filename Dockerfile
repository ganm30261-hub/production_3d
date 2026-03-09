FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# 只放训练相关代码，不放 b_data
COPY c_models/   ./c_models/
COPY g_training/ ./g_training/

ENV DATA_ROOT=/data_volume
ENV SAVE_DIR=/data_volume/runs

CMD ["python", "g_training/main.py", "--mode", "train_only"]
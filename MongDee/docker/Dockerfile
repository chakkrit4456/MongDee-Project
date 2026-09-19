# MongDee vision system — CPU image.
# For NVIDIA GPU, start FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 and
# install the CUDA torch wheels instead (see the pip line below).

FROM python:3.12-slim

# OpenCV runtime libs + ffmpeg (opencv-python ships its own ffmpeg, but
# libGL/libglib are still needed for cv2 to import)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU torch wheels keep the image ~2 GB instead of ~7 GB. For GPU:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir \
        opencv-python-headless ultralytics fastapi "uvicorn[standard]" \
        jinja2 python-multipart numpy requests psutil

COPY camera/ camera/
COPY vision/ vision/
COPY backend/ backend/
COPY scripts/ scripts/
COPY configs/ configs/
COPY yolo11n.pt .

# pre-fetch model weights into the image so first start is fast / offline-safe
RUN python scripts/download_models.py --torchvision resnet18

EXPOSE 8100
VOLUME ["/app/data"]

# override the config path at run time:
#   docker run -v $PWD/configs:/app/configs mongdee configs/mongdee.json
ENTRYPOINT ["python", "-m", "backend.main"]
CMD ["configs/mongdee.example.json"]

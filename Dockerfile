# ── Base: RunPod PyTorch 2.8.0 + CUDA 12.8.1 + Ubuntu 24.04 ──────────
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive

# ── System deps ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        nginx \
        libgl1 \
        libglib2.0-0 \
        curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps (skip torch/torchvision — base image has them) ────────
# Use /opt/app, not /workspace: RunPod mounts /workspace over the image and hides baked files.
WORKDIR /opt/app

# Install Python deps first for layer caching.
# Base image may ship distro cryptography without a pip RECORD; overlay a wheel first.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --ignore-installed cryptography \
    && pip install --no-cache-dir \
        fal \
        aiortc \
        av \
        opencv-python \
        pydantic>=2.0 \
        ultralytics \
        pillow \
        "numpy<2" \
        insightface \
        huggingface_hub \
        gfpgan \
        transformers

# onnxruntime-gpu for CUDA 12
RUN pip install --no-cache-dir onnxruntime-gpu \
        --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# fal calls _print_python_packages() at startup; some base-image dists have metadata=None → TypeError.
COPY scripts/patch-fal-app-metadata.py /tmp/patch-fal-app-metadata.py
RUN python /tmp/patch-fal-app-metadata.py && rm -f /tmp/patch-fal-app-metadata.py

# ── Copy project ──────────────────────────────────────────────────────
COPY . .

# ── Build frontend ────────────────────────────────────────────────────
# Container always runs fal --local; frontend must use direct WebSocket, not fal cloud tokens.
RUN cd frontend && npm ci && VITE_LOCAL_MODE=true npm run build

# ── Pre-download models at build time ─────────────────────────────────
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# yolov8n — ultralytics auto-downloads on first use, trigger it now
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# insightface buffalo_l + inswapper via our own code
RUN python -c "\
from insightface.app import FaceAnalysis; \
fa = FaceAnalysis(name='buffalo_l'); \
fa.prepare(ctx_id=-1, det_size=(640, 640)); \
print('buffalo_l downloaded') \
"
RUN python -c "\
from huggingface_hub import hf_hub_download; \
import os; \
path = hf_hub_download( \
    repo_id='hacksider/deep-live-cam', \
    filename='inswapper_128_fp16.onnx', \
    token=os.environ.get('HF_TOKEN'), \
); \
print(f'inswapper downloaded to {path}') \
"

# Clear the build arg so it doesn't leak into the runtime image
ENV HF_TOKEN=""

# ── cloudflared (Cloudflare Tunnel) ───────────────────────────────────
RUN curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb \
        -o /tmp/cloudflared.deb \
    && dpkg -i /tmp/cloudflared.deb \
    && rm -f /tmp/cloudflared.deb

# ── nginx + start script ─────────────────────────────────────────────
# Replace the base image's nginx.conf entirely (it has its own server blocks that conflict).
COPY nginx.conf /etc/nginx/nginx.conf
COPY start.sh /opt/start.sh
RUN chmod +x /opt/start.sh

EXPOSE 8888

CMD ["/opt/start.sh"]

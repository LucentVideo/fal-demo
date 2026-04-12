# lucent-base-gpu — shared base image for GPU apps.
#
# Build from lucent-serverless/:
#   docker buildx build --platform linux/amd64 -f images/gpu/Dockerfile \
#       -t raylightdimi/lucent-base-gpu:latest .
#
# Based on RunPod's PyTorch 2.8.0 + CUDA 12.8.1 image.
# Includes common heavy ML deps so user apps only need thin extras.

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive

# ── System deps ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── uv for fast dep installs at boot ────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Use /opt/app — RunPod mounts /workspace over the image at runtime
WORKDIR /opt/app

# ── Install lucent-serverless package (runner + SDK) ─────────────────
COPY pyproject.toml /opt/app/pyproject.toml
COPY lucent_serverless /opt/app/lucent_serverless
RUN uv pip install --system --break-system-packages --no-cache .

# ── Common heavy ML deps (pre-baked so users don't wait) ─────────────
# Base image already has: torch, torchvision, torchaudio, numpy
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --ignore-installed cryptography \
    && pip install --no-cache-dir \
        opencv-python \
        pydantic>=2.0 \
        pillow \
        "numpy<2" \
        transformers \
        huggingface_hub

# onnxruntime-gpu for CUDA 12
RUN pip install --no-cache-dir onnxruntime-gpu \
        --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# cupy-cuda12x for GPU-side array ops
RUN pip install --no-cache-dir cupy-cuda12x

# ── Entrypoint ───────────────────────────────────────────────────────
COPY images/gpu/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV LUCENT_PORT=8000
EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]

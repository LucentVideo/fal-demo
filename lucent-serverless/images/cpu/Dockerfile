# lucent-base-cpu — shared base image for CPU apps.
#
# Build from lucent-serverless/:
#   docker buildx build --platform linux/amd64 -f images/cpu/Dockerfile \
#       -t raylightdimi/lucent-base-cpu:latest .
#
# At pod boot the entrypoint:
#   1. Downloads user code from LUCENT_CODE_URL (tarball)
#   2. Installs extra deps from requirements.txt (if present) via uv
#   3. Starts the lucent runner

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# uv for fast dependency installs at boot
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install the lucent-serverless package (runner + SDK)
COPY pyproject.toml /app/pyproject.toml
COPY lucent_serverless /app/lucent_serverless
RUN uv pip install --system --no-cache .

# Entrypoint script
COPY images/cpu/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV LUCENT_PORT=8000
EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]

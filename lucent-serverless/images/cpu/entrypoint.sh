#!/bin/bash
set -e

# ── Download user code ───────────────────────────────────────────────
if [ -n "$LUCENT_CODE_URL" ]; then
    echo "[lucent] downloading app code from $LUCENT_CODE_URL"
    curl -fsSL "$LUCENT_CODE_URL" | tar xz -C /app
fi

# ── Install extra deps ──────────────────────────────────────────────
if [ -f /app/requirements.txt ]; then
    echo "[lucent] installing requirements.txt"
    uv pip install --system --no-cache -r /app/requirements.txt
fi

# ── Start runner ─────────────────────────────────────────────────────
echo "[lucent] starting runner"
exec python -m lucent_serverless.runner

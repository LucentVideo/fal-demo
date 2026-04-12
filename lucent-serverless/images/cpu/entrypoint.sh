#!/bin/bash
set -e

# Record boot start
python3 -c "import time,json; json.dump({'boot_start':time.time()}, open('/tmp/lucent_boot.json','w'))"

# ── Download user code ───────────────────────────────────────────────
if [ -n "$LUCENT_CODE_URL" ]; then
    echo "[lucent] downloading app code from $LUCENT_CODE_URL"
    curl -fsSL "$LUCENT_CODE_URL" | tar xz -C /app
    python3 -c "import time,json; d=json.load(open('/tmp/lucent_boot.json')); d['code_downloaded']=time.time(); json.dump(d,open('/tmp/lucent_boot.json','w'))"
fi

# ── Install extra deps ──────────────────────────────────────────────
if [ -f /app/requirements.txt ]; then
    echo "[lucent] installing requirements.txt"
    uv pip install --system --no-cache -r /app/requirements.txt
    python3 -c "import time,json; d=json.load(open('/tmp/lucent_boot.json')); d['deps_installed']=time.time(); json.dump(d,open('/tmp/lucent_boot.json','w'))"
fi

# ── Start runner ─────────────────────────────────────────────────────
echo "[lucent] starting runner"
exec python -m lucent_serverless.runner

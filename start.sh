#!/usr/bin/env bash
set -e

cd /opt/app

# Start fal backend in background
echo "Starting fal backend..."
fal run --local app.py::MultiPerceptionWebRTC &
FAL_PID=$!

# Wait for backend to be ready
echo "Waiting for backend on port 8080..."
for i in $(seq 1 60); do
    if curl -s -o /dev/null http://127.0.0.1:8080/ 2>/dev/null; then
        echo "Backend ready."
        break
    fi
    sleep 1
done

# Kill any nginx the base-image entrypoint may have started
nginx -s stop 2>/dev/null || true
sleep 0.5

echo "Starting nginx on port 8888..."
nginx -g "daemon off;" &
NGINX_PID=$!

# Start Cloudflare Tunnel if token is set
CF_PID=""
if [ -n "$CLOUDFLARE_TUNNEL_TOKEN" ]; then
    echo "Starting Cloudflare Tunnel..."
    cloudflared tunnel --no-autoupdate run --token "$CLOUDFLARE_TUNNEL_TOKEN" &
    CF_PID=$!
else
    echo "CLOUDFLARE_TUNNEL_TOKEN not set; skipping tunnel."
fi

# Trap signals for clean shutdown
trap "kill $FAL_PID $NGINX_PID $CF_PID 2>/dev/null; exit 0" SIGTERM SIGINT

echo "All services running. Frontend: http://0.0.0.0:8888"
wait

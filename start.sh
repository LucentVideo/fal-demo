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

# Start nginx in foreground
echo "Starting nginx on port 8888..."
nginx -g "daemon off;" &
NGINX_PID=$!

# Trap signals for clean shutdown
trap "kill $FAL_PID $NGINX_PID 2>/dev/null; exit 0" SIGTERM SIGINT

echo "All services running. Frontend: http://0.0.0.0:8888"
wait

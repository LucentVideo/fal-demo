# CPU-only base image for the echo example. Build from the lucent-serverless/
# directory:
#
#   docker build -f examples/echo/Dockerfile -t lucent-echo .
#   docker run --rm -p 8000:8000 lucent-echo
#
# Then connect a websocket client to ws://localhost:8000/realtime and curl
# http://localhost:8000/health.

FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
        fastapi==0.115.* \
        'uvicorn[standard]==0.32.*' \
        websockets==13.*

COPY lucent_serverless /app/lucent_serverless
COPY examples/echo/app.py /app/app.py

ENV LUCENT_APP_MODULE=app
ENV LUCENT_APP_ID=echo
ENV LUCENT_PORT=8000

EXPOSE 8000

CMD ["python", "-m", "lucent_serverless.runner"]

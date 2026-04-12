FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.115.* \
    "uvicorn[standard]==0.32.*" \
    httpx==0.28.* \
    python-dotenv==1.*

# Worker SDK (needed for runpod_client)
COPY lucent-serverless/lucent_serverless /app/lucent_serverless

# Controller
COPY lucent-serverless-controller/lucent_serverless_controller /app/lucent_serverless_controller

EXPOSE 8000

CMD ["python", "-m", "lucent_serverless_controller.main"]

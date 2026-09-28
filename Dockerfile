FROM python:3.11-slim

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (PyTorch CPU build)
COPY ai_service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy AI modules and AI service (includes ai/models/transformer.pth)
COPY ai/ ./ai/
COPY ai_service/ ./ai_service/

ENV PYTHONPATH=/app

CMD ["sh", "-c", "uvicorn ai_service.main:app --host 0.0.0.0 --port ${PORT:-10000}"]

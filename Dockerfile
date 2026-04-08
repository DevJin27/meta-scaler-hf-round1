FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV ENABLE_WEB_INTERFACE=true
ENV EMAILTRIAGE_HOST=0.0.0.0
ENV PORT=8000
ENV EMAILTRIAGE_MAX_CONCURRENT_ENVS=4

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY __init__.py ./
COPY baseline.py ./
COPY client.py ./
COPY email_core.py ./
COPY models.py ./
COPY openenv.yaml ./
COPY data ./data
COPY server ./server

RUN python -m pip install --upgrade pip && \
    python -m pip install .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]

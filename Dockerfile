# EmailTriage OpenEnv Environment
# Multi-stage build using the official openenv base image.
#
# Build:
#   docker build -t email-triage-env .
#
# Run:
#   docker run -p 8000:8000 email-triage-env
#   # With LLM baseline:
#   docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... email-triage-env

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest

# ---------------------------------------------------------------------------
# Stage 1: builder — install dependencies into a virtual environment
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Ensure uv is available
RUN if ! command -v uv >/dev/null 2>&1; then \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv  /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

COPY pyproject.toml ./
# Copy lock file if present (allows reproducible builds)
COPY uv.lock* ./

# Install production dependencies (no extras)
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

# Copy application code
COPY . .

# Install the package itself
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# ---------------------------------------------------------------------------
# Stage 2: runtime — lean image with only what's needed to serve
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy the application code
COPY --from=builder /app /app/env

# Make venv's executables available on PATH
ENV PATH="/app/.venv/bin:$PATH"

# Ensure Python can find the package when running from /app/env
ENV PYTHONPATH="/app/env:${PYTHONPATH}"

# Enable the OpenEnv web playground
ENV ENABLE_WEB_INTERFACE=true

# Runtime configuration (override with -e flags)
ENV EMAILTRIAGE_HOST=0.0.0.0
ENV PORT=8000
ENV EMAILTRIAGE_MAX_CONCURRENT_ENVS=4

# Health check — polls the /health endpoint provided by openenv-core
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE 8000

# Start the FastAPI server
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]

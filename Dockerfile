# ─────────────────────────────────────────────────────────────────────────────
# ShieldAI — Intelligent Threat Detection Platform
# Multi-stage build: keeps the final image lean and free of build tools.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies (needed to compile some ML wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Build wheels into a local directory so the runtime stage can install them
# without needing a compiler or internet access.
RUN pip install --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# libgomp1 is required at runtime by XGBoost for OpenMP threading
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pre-built wheels — no compiler needed in this stage
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* \
 && rm -rf /wheels

# Copy application source
COPY brain/    ./brain/
COPY api/      ./api/
COPY ui/       ./ui/

# Create directories the app expects at runtime
RUN mkdir -p brain/models data logs

# Non-root user — never run production containers as root
RUN groupadd -r shieldai && useradd -r -g shieldai -d /app shieldai \
 && chown -R shieldai:shieldai /app
USER shieldai

# Expose the API port
EXPOSE 8000

# Healthcheck — Docker marks the container unhealthy if the API stops responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" \
    || exit 1

# Start the server
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

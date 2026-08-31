# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /app

# System dependencies needed to build/install Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgdal-dev \
        gdal-bin \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# CPU-only PyTorch — keeps image ~2GB smaller than default CUDA build
RUN pip install --no-cache-dir \
        torch \
        torchvision \
        --index-url https://download.pytorch.org/whl/cpu

# API + worker dependencies
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.13 /usr/local/lib/python3.13
COPY --from=builder /usr/local/bin /usr/local/bin

# Runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        # GDAL runtime — needed by rasterio at runtime
        libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m appuser

# ── Application code ─────────────────────────────────────────────────────────
# All modules needed by both API and worker are copied here.
# The same image runs as API (uvicorn api.main:app) or worker
# (python -m worker.job_runner) — only the CMD differs.

COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser storage/      ./storage/
COPY --chown=appuser:appuser db/           ./db/
COPY --chown=appuser:appuser worker/       ./worker/
COPY --chown=appuser:appuser config.py     ./config.py

# Checkpoint baked in — can be overridden by volume mount in docker-compose
COPY --chown=appuser:appuser checkpoints/best_model.pth ./checkpoints/best_model.pth

# Results and DB directories — will be mounted as volumes in docker-compose
RUN mkdir -p /app/data/results /app/db_data \
    && chown -R appuser:appuser /app/data /app/db_data

USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Default: run the API
# Override in docker-compose worker service: ["python", "-m", "worker.job_runner"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
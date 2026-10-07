# =============================================================================
# Backend Dockerfile — multi-stage, production-ready
#
# Stage 1 (builder): installs Python deps into a venv
# Stage 2 (runtime): copies only the venv + app code, runs as non-root
#
# Secrets (SECRET_KEY, POSTGRES_PASSWORD, etc.) are NEVER embedded here.
# They are passed at runtime via environment variables or Docker secrets.
# =============================================================================

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System libraries needed to compile some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment so Stage 2 can copy it cleanly
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies (pinned in requirements.txt — no floating deps)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data into the venv so the container runs offline
RUN python -c " \
    import nltk; \
    nltk.download('stopwords',      quiet=True); \
    nltk.download('wordnet',        quiet=True); \
    nltk.download('punkt',          quiet=True); \
    nltk.download('punkt_tab',      quiet=True); \
    nltk.download('averaged_perceptron_tagger', quiet=True); \
    "


# ── Stage 2: production runtime ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Metadata
LABEL org.opencontainers.image.title="VeriFact AI Backend"
LABEL org.opencontainers.image.description="Fake News Detection — FastAPI backend"

# Runtime system deps only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── Non-root user ─────────────────────────────────────────────────────────────
# Running as root in a container is a security risk.
RUN groupadd --gid 1001 appgroup \
 && useradd  --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy application source
COPY --chown=appuser:appgroup . .

# Copy container entrypoint
COPY --chown=appuser:appgroup docker/entrypoint.sh /app/docker/entrypoint.sh
RUN chmod +x /app/docker/entrypoint.sh

# Create directories for ML models and logs that the app needs to write to
RUN mkdir -p ml/saved_models logs \
 && chown -R appuser:appgroup /app

USER appuser

# ── Runtime configuration ─────────────────────────────────────────────────────
# These are safe defaults. Override at runtime via environment variables.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000

EXPOSE 8000

# ── Health check ─────────────────────────────────────────────────────────────
# Docker will mark the container healthy/unhealthy based on this.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# ── Entry point ───────────────────────────────────────────────────────────────
# Production: no --reload, single worker (scale with replicas instead)
# Override CMD in docker-compose for development (add --reload --workers 1)
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]

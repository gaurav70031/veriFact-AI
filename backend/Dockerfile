# =============================================================================
# Backend Dockerfile — multi-stage, production-ready
# Render-compatible: all required system libs included
# =============================================================================

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System libraries for compiling Python packages
# libxml2 + libxslt needed by newspaper3k / lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    libxml2-dev \
    libxslt1-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data
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

LABEL org.opencontainers.image.title="VeriFact AI Backend"
LABEL org.opencontainers.image.description="Fake News Detection — FastAPI backend"

# Runtime system deps (libxml2/libxslt needed at runtime by lxml too)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    libxml2 \
    libxslt1.1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Non-root user
RUN groupadd --gid 1001 appgroup \
 && useradd  --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy app source
COPY --chown=appuser:appgroup . .

# Make entrypoint executable
RUN chmod +x /app/docker/entrypoint.sh

# Create dirs the app writes to
RUN mkdir -p ml/saved_models logs \
 && chown -R appuser:appgroup /app

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Entrypoint: waits for DB, runs migrations, then starts app
ENTRYPOINT ["/app/docker/entrypoint.sh"]

# Single worker on free tier (512 MB RAM)
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]

# =============================================================================
# Frontend Dockerfile — multi-stage
#
# Stage 1 (builder): npm ci + vite build
# Stage 2 (runtime): nginx serving static files + API proxy
#
# The VITE_API_BASE_URL build arg controls where the frontend sends API calls.
# In production this should be your backend's public URL.
# =============================================================================

# ── Stage 1: Node build ───────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Copy manifests first for better layer caching
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts

# Copy source and build
COPY . .

# VITE_API_BASE_URL is injected at build time.
# Default to /api/v1 so nginx can proxy it (recommended for production).
# Override: --build-arg VITE_API_BASE_URL=https://api.example.com/api/v1
ARG VITE_API_BASE_URL=/api/v1
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

RUN npm run build


# ── Stage 2: nginx runtime ────────────────────────────────────────────────────
FROM nginx:1.25-alpine AS runtime

LABEL org.opencontainers.image.title="VeritasAI Frontend"
LABEL org.opencontainers.image.description="Fake News Detection — React frontend"

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.prod.conf /etc/nginx/conf.d/default.conf

# Remove default nginx config
RUN rm -f /etc/nginx/conf.d/default.conf.bak

EXPOSE 80

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -q --spider http://localhost/index.html || exit 1

CMD ["nginx", "-g", "daemon off;"]

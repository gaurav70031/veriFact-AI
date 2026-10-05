#!/usr/bin/env sh
# =============================================================================
# Backend container entrypoint.
#
# Steps on every container start:
#   1. Wait for PostgreSQL to be ready (up to 60 s)
#   2. Run Alembic migrations (idempotent — safe to run multiple times)
#   3. Start the FastAPI server
#
# The script is intentionally POSIX sh (not bash) so it works in the
# python:3.11-slim image which may not have bash installed.
# =============================================================================

set -e

# ── 1. Wait for PostgreSQL ────────────────────────────────────────────────────
echo "[entrypoint] Waiting for PostgreSQL at ${POSTGRES_HOST}:${POSTGRES_PORT}..."

RETRIES=30
until python -c "
import sys, socket
try:
    s = socket.create_connection(('${POSTGRES_HOST}', int('${POSTGRES_PORT:-5432}')), timeout=2)
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; do
    RETRIES=$((RETRIES - 1))
    if [ "$RETRIES" -le 0 ]; then
        echo "[entrypoint] ERROR: PostgreSQL did not become available in time."
        exit 1
    fi
    echo "[entrypoint] PostgreSQL not ready yet — retrying in 2s ($RETRIES retries left)..."
    sleep 2
done

echo "[entrypoint] PostgreSQL is ready."

# ── 2. Run Alembic migrations ────────────────────────────────────────────────
echo "[entrypoint] Running database migrations..."
cd /app
alembic upgrade head
echo "[entrypoint] Migrations complete."

# ── 3. Start the application ──────────────────────────────────────────────────
echo "[entrypoint] Starting application..."
exec "$@"

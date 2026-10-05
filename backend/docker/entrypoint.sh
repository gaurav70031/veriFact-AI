#!/usr/bin/env sh
# =============================================================================
# Backend container entrypoint.
# Waits for PostgreSQL, runs Alembic migrations, then starts the app.
# =============================================================================

set -e

# ── 1. Wait for PostgreSQL ────────────────────────────────────────────────────
POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
MAX_RETRIES=30

echo "[entrypoint] Waiting for PostgreSQL at ${POSTGRES_HOST}:${POSTGRES_PORT}..."

i=0
until python -c "
import sys, socket
try:
    s = socket.create_connection(('${POSTGRES_HOST}', int('${POSTGRES_PORT}')), timeout=2)
    s.close()
    sys.exit(0)
except Exception as e:
    sys.exit(1)
" 2>/dev/null; do
    i=$((i + 1))
    if [ "$i" -ge "$MAX_RETRIES" ]; then
        echo "[entrypoint] ERROR: PostgreSQL did not become ready after ${MAX_RETRIES} attempts."
        exit 1
    fi
    echo "[entrypoint] Not ready yet (attempt ${i}/${MAX_RETRIES}). Retrying in 2s..."
    sleep 2
done
echo "[entrypoint] PostgreSQL is ready."

# ── 2. Run Alembic migrations ─────────────────────────────────────────────────
echo "[entrypoint] Running Alembic migrations..."
cd /app
alembic upgrade head
echo "[entrypoint] Migrations complete."

# ── 3. Start application ──────────────────────────────────────────────────────
echo "[entrypoint] Starting application: $*"
exec "$@"

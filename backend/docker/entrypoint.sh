#!/usr/bin/env sh
# =============================================================================
# Backend container entrypoint.
# 1. Downloads NLTK data (runtime — build env blocks network)
# 2. Waits for PostgreSQL
# 3. Runs Alembic migrations
# 4. Starts the app
# =============================================================================

set -e

# ── 1. Download NLTK data ─────────────────────────────────────────────────────
echo "[entrypoint] Downloading NLTK data..."
python -c "
import nltk, os
nltk.data.path.insert(0, os.environ.get('NLTK_DATA', '/app/nltk_data'))
for pkg in ['stopwords', 'wordnet', 'punkt', 'punkt_tab', 'averaged_perceptron_tagger']:
    try:
        nltk.download(pkg, quiet=True, download_dir=os.environ.get('NLTK_DATA', '/app/nltk_data'))
    except Exception as e:
        print(f'[entrypoint] Warning: could not download {pkg}: {e}')
print('[entrypoint] NLTK data ready.')
" || echo "[entrypoint] Warning: NLTK download failed — continuing anyway."

# ── 2. Wait for PostgreSQL ────────────────────────────────────────────────────
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
except Exception:
    sys.exit(1)
" 2>/dev/null; do
    i=$((i + 1))
    if [ "$i" -ge "$MAX_RETRIES" ]; then
        echo "[entrypoint] ERROR: PostgreSQL not ready after ${MAX_RETRIES} attempts."
        exit 1
    fi
    echo "[entrypoint] Not ready yet (attempt ${i}/${MAX_RETRIES}). Retrying in 2s..."
    sleep 2
done
echo "[entrypoint] PostgreSQL is ready."

# ── 3. Run Alembic migrations ─────────────────────────────────────────────────
echo "[entrypoint] Running Alembic migrations..."
cd /app
alembic upgrade head
echo "[entrypoint] Migrations complete."

# ── 4. Start application ──────────────────────────────────────────────────────
echo "[entrypoint] Starting application..."
exec "$@"

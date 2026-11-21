#!/usr/bin/env sh
# =============================================================================
# Backend container entrypoint.
#
# Steps on every container start:
#   1. Download NLTK data (cached after first run)
#   2. Parse DATABASE_URL if individual POSTGRES_* vars aren't set
#   3. Wait for PostgreSQL to be ready (up to 60 s)
#   4. Run Alembic migrations (idempotent)
#   5. Start the FastAPI server
# =============================================================================

set -e

# ── 1. Download NLTK data ────────────────────────────────────────────────────
echo "[entrypoint] Downloading NLTK data..."
python -c "
import nltk, os
data_dir = os.environ.get('NLTK_DATA', '/app/nltk_data')
os.makedirs(data_dir, exist_ok=True)
for pkg in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
    try:
        nltk.download(pkg, download_dir=data_dir, quiet=True)
    except Exception:
        pass
print('[entrypoint] NLTK data ready.')
"

# ── 2. Parse DATABASE_URL if set (Render provides this automatically) ────────
# Render injects DATABASE_URL in the format:
#   postgresql://user:password@host:port/dbname
# Parse it into individual POSTGRES_* vars if they're not already set correctly.

if [ -n "$DATABASE_URL" ]; then
    echo "[entrypoint] Parsing DATABASE_URL..."
    # Strip the scheme
    _url="${DATABASE_URL#postgresql://}"
    _url="${_url#postgres://}"

    # Extract user:password@host:port/dbname
    _userinfo="${_url%%@*}"
    _hostinfo="${_url#*@}"

    POSTGRES_USER="${_userinfo%%:*}"
    POSTGRES_PASSWORD="${_userinfo#*:}"

    _hostport="${_hostinfo%%/*}"
    POSTGRES_DB="${_hostinfo#*/}"
    # Remove any query string from db name
    POSTGRES_DB="${POSTGRES_DB%%\?*}"

    if echo "$_hostport" | grep -q ':'; then
        POSTGRES_HOST="${_hostport%%:*}"
        POSTGRES_PORT="${_hostport##*:}"
    else
        POSTGRES_HOST="$_hostport"
        POSTGRES_PORT="5432"
    fi

    export POSTGRES_USER POSTGRES_PASSWORD POSTGRES_HOST POSTGRES_PORT POSTGRES_DB
    echo "[entrypoint] DB host: $POSTGRES_HOST  port: $POSTGRES_PORT  db: $POSTGRES_DB"
fi

# ── 3. Wait for PostgreSQL ────────────────────────────────────────────────────
echo "[entrypoint] Waiting for PostgreSQL at ${POSTGRES_HOST}:${POSTGRES_PORT}..."

RETRIES=30
until python -c "
import sys, socket
try:
    s = socket.create_connection(('${POSTGRES_HOST}', int('${POSTGRES_PORT:-5432}')), timeout=2)
    s.close()
    sys.exit(0)
except Exception as e:
    sys.exit(1)
" 2>/dev/null; do
    RETRIES=$((RETRIES - 1))
    if [ "$RETRIES" -le 0 ]; then
        echo "[entrypoint] ERROR: PostgreSQL not ready after 30 attempts."
        exit 1
    fi
    echo "[entrypoint] Not ready yet (attempt $((30 - RETRIES))/30). Retrying in 2s..."
    sleep 2
done

echo "[entrypoint] PostgreSQL is ready."

# ── 4. Run Alembic migrations ─────────────────────────────────────────────────
echo "[entrypoint] Running database migrations..."
cd /app
alembic upgrade head
echo "[entrypoint] Migrations complete."

# ── 5. Start the application ──────────────────────────────────────────────────
echo "[entrypoint] Starting application..."
exec "$@"

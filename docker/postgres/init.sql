-- PostgreSQL initialisation script
-- Runs once when the container is first created (docker-entrypoint-initdb.d)
-- The actual schema is managed by Alembic migrations, not this file.
-- This file only creates extensions that Alembic itself cannot easily enable.

-- Enable pg_trgm for trigram-based text search (used in GIN indexes)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Enable pgcrypto for gen_random_uuid() (used in future features)
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Confirm extensions installed
SELECT extname, extversion FROM pg_extension ORDER BY extname;

"""
Alembic environment configuration.

Supports both:
- Online mode  (run_migrations_online): used by `alembic upgrade head`
- Offline mode (run_migrations_offline): generates SQL scripts without a live DB

The database URL is read from the application's Settings object so it is
always consistent with what the app itself uses.  Never hard-code credentials
here.

Because SQLAlchemy 2.x async engines cannot be used directly with Alembic's
synchronous migration runner, we use a *synchronous* psycopg2 URL for
migrations only.
"""

import asyncio
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# ---------------------------------------------------------------------------
# Make the app package importable (alembic.ini sets prepend_sys_path = .)
# ---------------------------------------------------------------------------
from app.core.config import get_settings
from app.db.base import Base

# Import all models so their tables are registered on Base.metadata
import app.models  # noqa: F401  — side-effect import

# ---------------------------------------------------------------------------
# Alembic Config object (provides access to values in alembic.ini)
# ---------------------------------------------------------------------------
config = context.config

# Set up Python logging from the alembic.ini [loggers] section
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The metadata object Alembic inspects to detect schema differences
target_metadata = Base.metadata

# ---------------------------------------------------------------------------
# Inject the synchronous DB URL at runtime
# ---------------------------------------------------------------------------
def get_sync_url() -> str:
    """
    Return a psycopg2 (synchronous) URL derived from app settings.
    Alembic's migration runner is synchronous and cannot use asyncpg.
    """
    settings = get_settings()
    url = settings.database_url
    # Force psycopg2 driver explicitly — SQLAlchemy 2.1+ defaults to psycopg (v3)
    # which is not installed. psycopg2-binary is our sync driver.
    url = url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
    url = url.replace("postgres://", "postgresql+psycopg2://")
    url = url.replace("postgresql://", "postgresql+psycopg2://")
    return url


# ---------------------------------------------------------------------------
# Offline migration (generates SQL without connecting to the DB)
# ---------------------------------------------------------------------------
def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    Generates a .sql script that can be reviewed before applying.
    Useful for audits and environments where direct DB access is restricted.
    """
    url = get_sync_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online migration (connects to the DB and applies changes directly)
# ---------------------------------------------------------------------------
def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode — the default for `alembic upgrade head`.
    Uses a synchronous psycopg2 connection.
    """
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_sync_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # no connection pooling during migrations
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,              # detect column type changes
            compare_server_default=True,    # detect default value changes
        )

        with context.begin_transaction():
            context.run_migrations()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

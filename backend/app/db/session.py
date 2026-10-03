"""
Async SQLAlchemy engine + session factory.

Usage inside FastAPI route/service:

    async def my_endpoint(db: AsyncSession = Depends(get_db)):
        result = await db.execute(select(User))
        ...
"""

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
    AsyncEngine,
)
from sqlalchemy.pool import NullPool

from app.core.config import get_settings

settings = get_settings()

# asyncpg driver is required:  pip install asyncpg
_DATABASE_URL = settings.database_url.replace(
    "postgresql://", "postgresql+asyncpg://"
).replace(
    "postgres://", "postgresql+asyncpg://"   # handle both forms
)

# NullPool is recommended for async engines to avoid connection leaks in
# short-lived serverless / test environments.  For a long-running server
# switch to AsyncAdaptedQueuePool (default) by removing pool_class.
engine: AsyncEngine = create_async_engine(
    _DATABASE_URL,
    echo=settings.debug,           # log SQL only in debug mode
    pool_pre_ping=True,            # verify connections before use
    pool_recycle=1800,             # recycle stale connections every 30 min
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,        # objects remain usable after commit
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncSession:          # type: ignore[return]
    """
    FastAPI dependency that yields a database session.
    Commits on success, rolls back on any exception, always closes.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

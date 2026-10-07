"""
Create (or reset) the admin user.

Usage
-----
Run from the backend/ directory:

    python scripts/create_admin.py

The script connects to PostgreSQL using the same .env settings the app uses.
If the admin account already exists it updates the password and role instead
of creating a duplicate.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
# Make sure `app` is importable when running from backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.core.config import get_settings
from app.core.auth_utils import hash_password
from app.models.user import User, UserRole
import app.models  # noqa: registers all ORM models


# ── Admin credentials ─────────────────────────────────────────────────────────
ADMIN_EMAIL    = "admin@veritasai.com"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "Admin@1234"       # change after first login
ADMIN_FULLNAME = "VeritasAI Admin"


async def create_admin() -> None:
    settings = get_settings()
    engine   = create_async_engine(settings.database_url, echo=False)
    factory  = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as session:
        # Check if already exists
        result = await session.execute(
            select(User).where(User.email == ADMIN_EMAIL)
        )
        existing = result.scalar_one_or_none()

        if existing:
            # Update role + password in case they were changed
            existing.hashed_password = hash_password(ADMIN_PASSWORD)
            existing.role            = UserRole.ADMIN
            existing.is_active       = True
            await session.commit()
            print(f"✓  Admin account already existed — password and role reset.")
        else:
            admin = User(
                email           = ADMIN_EMAIL,
                username        = ADMIN_USERNAME,
                hashed_password = hash_password(ADMIN_PASSWORD),
                full_name       = ADMIN_FULLNAME,
                role            = UserRole.ADMIN,
                is_active       = True,
                is_verified     = True,
            )
            session.add(admin)
            await session.commit()
            print(f"✓  Admin account created.")

        print()
        print("  ┌─────────────────────────────────────────┐")
        print(f"  │  Email    : {ADMIN_EMAIL:<29}│")
        print(f"  │  Username : {ADMIN_USERNAME:<29}│")
        print(f"  │  Password : {ADMIN_PASSWORD:<29}│")
        print(f"  │  Role     : admin                       │")
        print("  └─────────────────────────────────────────┘")
        print()
        print("  ⚠  Change the password after your first login.")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(create_admin())

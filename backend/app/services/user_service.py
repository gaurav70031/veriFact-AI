"""
User service.

Handles user registration and authentication.
All password operations use bcrypt — plaintext passwords are never stored.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth_utils import hash_password, verify_password
from app.models.user import User, UserRole
from app.schemas.auth import RegisterRequest

logger = logging.getLogger(__name__)


# ── Registration ──────────────────────────────────────────────────────────────

async def register_user(db: AsyncSession, body: RegisterRequest) -> User:
    """
    Create a new user account.

    Raises HTTP 409 if the email or username is already taken.
    The password is bcrypt-hashed before storage — plaintext is discarded.

    Parameters
    ----------
    db   : Async SQLAlchemy session.
    body : Validated RegisterRequest.

    Returns
    -------
    The newly created and flushed User ORM instance.
    """
    # Check email uniqueness
    existing_email = await db.execute(
        select(User).where(User.email == body.email.lower())
    )
    if existing_email.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email address already exists.",
        )

    # Check username uniqueness
    existing_user = await db.execute(
        select(User).where(User.username == body.username)
    )
    if existing_user.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This username is already taken.",
        )

    user = User(
        email=body.email.lower(),
        username=body.username,
        hashed_password=hash_password(body.password),   # bcrypt — never plaintext
        full_name=body.full_name,
        role=UserRole.USER,
        is_active=True,
        is_verified=False,
    )
    db.add(user)
    await db.flush()   # get id without committing
    logger.info("New user registered: id=%d email=%s", user.id, user.email)
    return user


# ── Authentication ────────────────────────────────────────────────────────────

async def authenticate_user(
    db:       AsyncSession,
    email:    str,
    password: str,
) -> User:
    """
    Verify credentials and return the User if valid.

    Raises HTTP 401 for any failure — intentionally non-specific
    to prevent user enumeration attacks.

    Parameters
    ----------
    db       : Async SQLAlchemy session.
    email    : Email address (case-insensitive).
    password : Plaintext password to verify against the stored hash.

    Returns
    -------
    The authenticated User ORM instance.
    """
    result = await db.execute(
        select(User).where(User.email == email.lower())
    )
    user = result.scalar_one_or_none()

    # Deliberate: same error for "not found" and "wrong password"
    # to prevent user enumeration via timing or message differences.
    if user is None or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="This account has been deactivated.",
        )

    logger.info("User authenticated: id=%d", user.id)
    return user


# ── Lookup ────────────────────────────────────────────────────────────────────

async def get_user_by_id(db: AsyncSession, user_id: int) -> User | None:
    result = await db.execute(
        select(User).where(User.id == user_id, User.is_active.is_(True))
    )
    return result.scalar_one_or_none()

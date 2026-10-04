"""
FastAPI request dependencies for authentication.

get_current_user   — requires a valid JWT cookie; raises 401 otherwise.
get_optional_user  — returns the User if a valid cookie is present, else None.
                     Used for endpoints that work anonymously but can attach
                     analyses to a user when logged in (e.g. POST /analyze/*).
require_admin      — requires the current user to have role=admin.

The JWT is read from the httpOnly cookie set by POST /auth/login.
The browser sends it automatically — no client-side JS can read it.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Cookie, Depends, HTTPException, status
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth_utils import decode_access_token
from app.db.session import get_db
from app.models.user import User

logger = logging.getLogger(__name__)

# ── Cookie name constant (keep in sync with settings.cookie_name) ─────────────
_COOKIE_NAME = "access_token"


async def _user_from_token(
    token: Optional[str],
    db:    AsyncSession,
) -> Optional[User]:
    """Shared resolution logic: token → User or None."""
    if not token:
        return None

    try:
        payload = decode_access_token(token)
        user_id_str = payload.get("sub")
        token_type  = payload.get("type")
        if user_id_str is None or token_type != "access":
            return None
        user_id = int(user_id_str)
    except (JWTError, ValueError, TypeError):
        logger.debug("Invalid JWT token (decode failed).")
        return None

    result = await db.execute(
        select(User).where(User.id == user_id, User.is_active.is_(True))
    )
    return result.scalar_one_or_none()


# ── Public dependency: requires auth ─────────────────────────────────────────

async def get_current_user(
    access_token: Optional[str] = Cookie(default=None, alias=_COOKIE_NAME),
    db:           AsyncSession  = Depends(get_db),
) -> User:
    """
    FastAPI dependency that requires a valid, active authenticated user.

    Raises HTTP 401 if:
      - No cookie is present
      - The token is expired or tampered
      - The user does not exist or is inactive

    Usage:
        @router.get("/protected")
        async def protected(user: User = Depends(get_current_user)):
            ...
    """
    user = await _user_from_token(access_token, db)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Cookie"},
        )
    return user


# ── Optional dependency: auth if present ─────────────────────────────────────

async def get_optional_user(
    access_token: Optional[str] = Cookie(default=None, alias=_COOKIE_NAME),
    db:           AsyncSession  = Depends(get_db),
) -> Optional[User]:
    """
    FastAPI dependency that returns the authenticated user or None.

    Used for endpoints that work both anonymously and as a logged-in user —
    for example, POST /analyze/* attaches the analysis to the user when
    logged in, but still works without authentication.

    Usage:
        @router.post("/analyze/text")
        async def analyse_text(
            body: AnalyzeTextRequest,
            db:   AsyncSession     = Depends(get_db),
            user: User | None      = Depends(get_optional_user),
        ):
            user_id = user.id if user else None
            ...
    """
    return await _user_from_token(access_token, db)


# ── Admin-only dependency ─────────────────────────────────────────────────────

async def require_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    """Raises HTTP 403 if the current user is not an admin."""
    from app.models.user import UserRole
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required.",
        )
    return current_user

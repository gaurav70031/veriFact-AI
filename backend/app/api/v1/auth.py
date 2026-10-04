"""
Authentication endpoints.

POST /api/v1/auth/register  — create account, set cookie
POST /api/v1/auth/login     — verify credentials, set cookie
POST /api/v1/auth/logout    — delete cookie
GET  /api/v1/auth/me        — return current user profile

JWT token lifecycle
-------------------
1. /login sets an httpOnly cookie: access_token=<JWT>
2. Every subsequent request sends the cookie automatically.
3. /logout deletes the cookie.
4. The token is signed with SECRET_KEY from environment variables.
   The secret is NEVER embedded in source code.

Security properties of httpOnly cookies
-----------------------------------------
- JavaScript cannot read the cookie (XSS mitigation).
- The browser attaches it automatically on same-origin requests.
- SameSite=lax prevents most CSRF attacks for state-mutating endpoints.
- Secure=True in production (requires HTTPS) prevents transmission over HTTP.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth_utils import create_access_token
from app.core.config import get_settings
from app.core.dependencies import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.auth import AuthResponse, LoginRequest, RegisterRequest, UserOut
from app.services.user_service import authenticate_user, register_user

logger   = logging.getLogger(__name__)
settings = get_settings()
router   = APIRouter(prefix="/auth", tags=["Authentication"])


def _set_auth_cookie(response: Response, user: User) -> None:
    """Create a JWT and attach it as an httpOnly cookie."""
    token = create_access_token(user_id=user.id, role=user.role.value)
    response.set_cookie(
        key=settings.cookie_name,
        value=token,
        httponly=True,                              # JS cannot read it
        secure=settings.cookie_secure,              # True = HTTPS only
        samesite=settings.cookie_samesite,          # "lax" prevents CSRF
        max_age=settings.access_token_expire_minutes * 60,
        path="/",
    )


# ── Register ──────────────────────────────────────────────────────────────────

@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new account",
    description=(
        "Register with email, username, and password.\n\n"
        "Password is bcrypt-hashed before storage — plaintext is never saved.\n\n"
        "On success, sets an httpOnly JWT cookie and returns the user profile."
    ),
)
async def register(
    body:     RegisterRequest,
    response: Response,
    db:       AsyncSession = Depends(get_db),
) -> AuthResponse:
    user = await register_user(db, body)
    _set_auth_cookie(response, user)
    return AuthResponse(user=UserOut.model_validate(user))


# ── Login ─────────────────────────────────────────────────────────────────────

@router.post(
    "/login",
    response_model=AuthResponse,
    status_code=status.HTTP_200_OK,
    summary="Login",
    description=(
        "Authenticate with email + password.\n\n"
        "On success, sets an httpOnly JWT cookie (max-age = "
        f"{settings.access_token_expire_minutes} minutes) and returns the user profile.\n\n"
        "On failure, returns HTTP 401.  The same error is returned for both "
        "'email not found' and 'wrong password' to prevent user enumeration."
    ),
)
async def login(
    body:     LoginRequest,
    response: Response,
    db:       AsyncSession = Depends(get_db),
) -> AuthResponse:
    user = await authenticate_user(db, email=body.email, password=body.password)
    _set_auth_cookie(response, user)
    return AuthResponse(user=UserOut.model_validate(user), message="Login successful.")


# ── Logout ────────────────────────────────────────────────────────────────────

@router.post(
    "/logout",
    status_code=status.HTTP_200_OK,
    summary="Logout",
    description=(
        "Deletes the access_token cookie.\n\n"
        "Always returns 200 — even if no cookie was present."
    ),
)
async def logout(response: Response) -> dict:
    response.delete_cookie(
        key=settings.cookie_name,
        path="/",
        httponly=True,
        samesite=settings.cookie_samesite,
    )
    return {"message": "Logged out successfully."}


# ── Me ────────────────────────────────────────────────────────────────────────

@router.get(
    "/me",
    response_model=UserOut,
    status_code=status.HTTP_200_OK,
    summary="Current user profile",
    description=(
        "Returns the profile of the authenticated user.\n\n"
        "Returns HTTP 401 when not authenticated.\n\n"
        "The frontend calls this on page load to restore session state."
    ),
)
async def me(current_user: User = Depends(get_current_user)) -> UserOut:
    return UserOut.model_validate(current_user)

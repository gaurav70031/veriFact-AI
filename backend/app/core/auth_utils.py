"""
Authentication utilities.

Password hashing
----------------
Uses passlib with bcrypt backend.  Plaintext passwords are NEVER stored —
only their bcrypt hashes.  Work factor is passlib's default (12 rounds).

JWT tokens
----------
Uses python-jose with HS256 algorithm.
The signing secret is read from config.secret_key which must be set via
the SECRET_KEY environment variable in production.  The weak default in
config.py is intentionally obvious so it cannot be accidentally deployed.

Token payload shape
-------------------
{
    "sub":  "42",            # string — user.id
    "role": "user",          # user.role.value
    "type": "access",
    "exp":  1234567890,      # UTC Unix timestamp
    "iat":  1234567800,      # issued-at
}

The token is stored in an httpOnly cookie — never in localStorage or a
JS-accessible location.  The browser sends it automatically on every request
to the backend origin.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import get_settings

settings = get_settings()

# ── Password hashing ──────────────────────────────────────────────────────────

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plaintext: str) -> str:
    """Return a bcrypt hash of *plaintext*.  Never store the plaintext."""
    return _pwd_context.hash(plaintext)


def verify_password(plaintext: str, hashed: str) -> bool:
    """Return True if *plaintext* matches *hashed*."""
    return _pwd_context.verify(plaintext, hashed)


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_access_token(user_id: int, role: str) -> str:
    """
    Create a signed JWT access token.

    Parameters
    ----------
    user_id : Primary key of the user.
    role    : Role string (e.g. "user", "analyst", "admin").

    Returns
    -------
    Encoded JWT string.  The secret is read from settings.secret_key which
    MUST be overridden via the SECRET_KEY environment variable in production.
    """
    now    = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.access_token_expire_minutes)

    payload = {
        "sub":  str(user_id),
        "role": role,
        "type": "access",
        "iat":  int(now.timestamp()),
        "exp":  expire,
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    """
    Decode and verify a JWT access token.

    Raises
    ------
    jose.JWTError if the token is expired, tampered, or malformed.
    """
    return jwt.decode(
        token,
        settings.secret_key,
        algorithms=[settings.jwt_algorithm],
    )

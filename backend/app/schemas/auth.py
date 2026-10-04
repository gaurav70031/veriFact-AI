"""
Pydantic schemas for authentication endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator, ConfigDict


# ── Request schemas ───────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email:    EmailStr = Field(..., description="Valid email address.")
    username: str      = Field(..., min_length=3, max_length=50, description="3–50 chars.")
    password: str      = Field(..., min_length=8, max_length=128,
                               description="At least 8 characters.")
    full_name: Optional[str] = Field(None, max_length=200)

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        import re
        if not re.match(r"^[a-zA-Z0-9_.-]+$", v):
            raise ValueError(
                "Username may only contain letters, numbers, underscores, "
                "dots, and hyphens."
            )
        return v.strip()

    @field_validator("password")
    @classmethod
    def password_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Password must not be blank.")
        return v   # do NOT strip — spaces in passwords are valid

    model_config = ConfigDict(str_strip_whitespace=True)


class LoginRequest(BaseModel):
    email:    EmailStr = Field(..., description="Registered email address.")
    password: str      = Field(..., min_length=1, description="Account password.")

    model_config = ConfigDict(str_strip_whitespace=True)


# ── Response schemas ──────────────────────────────────────────────────────────

class UserOut(BaseModel):
    """Public user profile — never includes hashed_password."""

    id:        int
    email:     str
    username:  str
    full_name: Optional[str]
    role:      str
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AuthResponse(BaseModel):
    """Returned by /login and /register on success."""

    user:    UserOut
    message: str = "Authentication successful."

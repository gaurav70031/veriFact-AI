"""
Declarative base for all SQLAlchemy ORM models.

Import Base here and all models register against it automatically.
Alembic's env.py imports Base.metadata to detect schema changes.
"""

from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped
from sqlalchemy import DateTime, func
from datetime import datetime


class Base(DeclarativeBase):
    """
    Shared base class.  Every ORM model inherits from this.
    Provides a common `created_at` mixin via TimestampMixin below.
    """
    pass


class TimestampMixin:
    """Adds created_at / updated_at columns to any model that inherits it."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

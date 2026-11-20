"""Add BERT and RoBERTa to model_algorithm enum.

Revision ID: 0003
Revises: 0002
Create Date: 2024-06-01 00:00:00.000000

NOTE: ALTER TYPE ... ADD VALUE cannot run inside a transaction on PostgreSQL.
We use the AUTOCOMMIT pattern described in:
  https://alembic.sqlalchemy.org/en/latest/cookbook.html#adding-new-enum-values-without-transactions
"""

from typing import Sequence, Union
from alembic import op

revision:      str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on:    Union[str, Sequence[str], None] = None


def _add_enum_value(conn, enum_type: str, value: str) -> None:
    """Add a value to a PostgreSQL enum, skipping if already present."""
    # Check if value already exists before adding
    result = conn.execute(
        f"SELECT 1 FROM pg_enum e "
        f"JOIN pg_type t ON e.enumtypid = t.oid "
        f"WHERE t.typname = '{enum_type}' AND e.enumlabel = '{value}'"
    )
    if result.fetchone() is None:
        conn.execute(f"ALTER TYPE {enum_type} ADD VALUE '{value}'")


def upgrade() -> None:
    # Must run outside a transaction block on PostgreSQL
    bind = op.get_bind()

    # Get raw DBAPI connection
    if hasattr(bind, "connection"):
        raw = bind.connection
    else:
        raw = bind.engine.raw_connection()

    original_isolation = raw.isolation_level
    raw.set_isolation_level(0)  # AUTOCOMMIT

    try:
        _add_enum_value(raw, "model_algorithm", "bert")
        _add_enum_value(raw, "model_algorithm", "roberta")
    finally:
        raw.set_isolation_level(original_isolation)


def downgrade() -> None:
    # PostgreSQL does not support removing enum values without recreating the type.
    # This migration is intentionally irreversible.
    pass

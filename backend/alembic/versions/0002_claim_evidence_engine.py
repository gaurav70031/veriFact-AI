"""Add evidence assessment enum, evidence_verdict + evidence_explanation columns,
   NOT_RELEVANT to evidence_relationship, comparison_score to evidence_sources.

Revision ID: 0002
Revises: 0001
Create Date: 2024-01-02 00:00:00.000000
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on:    Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── 1. Add NOT_RELEVANT to evidence_relationship enum ────────────────────
    # PostgreSQL requires COMMIT before ALTER TYPE ADD VALUE in a transaction,
    # so we execute it outside a transaction block.
    op.execute("ALTER TYPE evidence_relationship ADD VALUE IF NOT EXISTS 'not_relevant'")

    # ── 2. Create new evidence_assessment enum ────────────────────────────────
    sa.Enum(
        "LIKELY_CREDIBLE",
        "LIKELY_MISLEADING",
        "CONTRADICTED",
        "UNVERIFIED",
        "INSUFFICIENT_EVIDENCE",
        name="evidence_assessment",
    ).create(op.get_bind(), checkfirst=True)

    # ── 3. Add evidence_verdict column to claims ──────────────────────────────
    op.add_column(
        "claims",
        sa.Column(
            "evidence_verdict",
            postgresql.ENUM(
                "LIKELY_CREDIBLE", "LIKELY_MISLEADING", "CONTRADICTED",
                "UNVERIFIED", "INSUFFICIENT_EVIDENCE",
                name="evidence_assessment",
                create_type=False,
            ),
            nullable=True,
        ),
    )

    # ── 4. Add evidence_explanation column to claims ──────────────────────────
    op.add_column(
        "claims",
        sa.Column("evidence_explanation", sa.Text(), nullable=True),
    )

    # ── 5. Add comparison_score column to evidence_sources ────────────────────
    op.add_column(
        "evidence_sources",
        sa.Column(
            "comparison_score",
            sa.Float(),
            nullable=False,
            server_default="0.0",
        ),
    )
    op.create_check_constraint(
        "ck_evidence_comparison_range",
        "evidence_sources",
        "comparison_score >= 0.0 AND comparison_score <= 1.0",
    )

    # ── 6. Index on new evidence_verdict column ───────────────────────────────
    op.create_index("ix_claims_evidence_verdict", "claims", ["evidence_verdict"])


def downgrade() -> None:
    op.drop_index("ix_claims_evidence_verdict", table_name="claims")

    op.drop_constraint("ck_evidence_comparison_range", "evidence_sources")
    op.drop_column("evidence_sources", "comparison_score")

    op.drop_column("claims", "evidence_explanation")
    op.drop_column("claims", "evidence_verdict")

    sa.Enum(name="evidence_assessment").drop(op.get_bind(), checkfirst=True)

    # NOTE: PostgreSQL does not support removing values from an ENUM type.
    # The 'not_relevant' value added to evidence_relationship cannot be
    # automatically removed on downgrade.  Downgrade to 0001 only removes
    # the new columns; the enum value must be cleaned up manually if needed.

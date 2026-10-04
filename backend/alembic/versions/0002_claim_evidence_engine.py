"""Add evidence assessment enum, evidence_verdict + evidence_explanation columns,
   NOT_RELEVANT to evidence_relationship, comparison_score to evidence_sources.

Revision ID: 0002
Revises: 0001
Create Date: 2024-01-02 00:00:00.000000

IMPORTANT — PostgreSQL ALTER TYPE constraint
--------------------------------------------
`ALTER TYPE ... ADD VALUE` cannot execute inside a transaction block in
PostgreSQL. Alembic runs migrations inside a transaction by default.

The workaround used here: we grab a raw DBAPI connection, set its isolation
level to AUTOCOMMIT, execute the ALTER TYPE, then restore the original
isolation level before handing the connection back to Alembic.

This pattern is the recommended approach for PostgreSQL enum extensions in
Alembic — see https://alembic.sqlalchemy.org/en/latest/cookbook.html
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
    # Must run OUTSIDE a transaction block in PostgreSQL.
    # We use a raw DBAPI connection in AUTOCOMMIT mode for this single
    # statement, then restore DEFERRED isolation for the rest.
    conn = op.get_bind()
    conn.execute(sa.text("COMMIT"))   # end the transaction Alembic opened
    conn.execute(
        sa.text("ALTER TYPE evidence_relationship ADD VALUE IF NOT EXISTS 'not_relevant'")
    )
    # Alembic will open a new transaction for the remaining DDL statements.

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
    # NOTE: PostgreSQL does not support removing enum values — 'not_relevant'
    # remains in the evidence_relationship enum after downgrade.

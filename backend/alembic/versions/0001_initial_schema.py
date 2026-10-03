"""Initial schema — all tables, ENUMs, indexes, constraints, trigger

Revision ID: 0001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_enum(name: str, *values: str) -> postgresql.ENUM:
    """Create a PostgreSQL ENUM type that is schema-managed (not inline)."""
    return postgresql.ENUM(*values, name=name, create_type=False)


def upgrade() -> None:
    # ── Extensions ──────────────────────────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # ── ENUM types ──────────────────────────────────────────────────────────
    sa.Enum("user", "analyst", "admin", name="user_role").create(op.get_bind(), checkfirst=True)
    sa.Enum("text", "url", "query", name="input_type").create(op.get_bind(), checkfirst=True)
    sa.Enum("pending", "processing", "completed", "failed",
            name="analysis_status").create(op.get_bind(), checkfirst=True)
    sa.Enum("FAKE", "REAL", "UNVERIFIED", "MIXED",
            name="final_verdict").create(op.get_bind(), checkfirst=True)
    sa.Enum("FAKE", "REAL", "UNVERIFIED", "MIXED",
            name="claim_verdict").create(op.get_bind(), checkfirst=True)
    sa.Enum("FAKE", "REAL", name="prediction_label").create(op.get_bind(), checkfirst=True)
    sa.Enum("traditional", "transformer", "ensemble",
            name="model_type").create(op.get_bind(), checkfirst=True)
    sa.Enum("logistic_regression", "linear_svm", "naive_bayes", "distilbert", "ensemble",
            name="model_algorithm").create(op.get_bind(), checkfirst=True)
    sa.Enum("news_api", "rss_feed", "web_search", "official",
            "fact_check", "academic", "user_url",
            name="source_type").create(op.get_bind(), checkfirst=True)
    sa.Enum("supporting", "contradicting", "inconclusive",
            name="evidence_relationship").create(op.get_bind(), checkfirst=True)
    sa.Enum("success", "failed", "timeout", "skipped",
            name="run_status").create(op.get_bind(), checkfirst=True)

    # ── TABLE: users ─────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("username", sa.String(100), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(200), nullable=True),
        sa.Column("role", _create_enum("user_role"), nullable=False, server_default="user"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("is_verified", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email", name="uq_users_email"),
        sa.UniqueConstraint("username", name="uq_users_username"),
    )
    op.create_index("ix_users_email_active", "users", ["email", "is_active"])
    op.create_index("ix_users_username", "users", ["username"])

    # ── TABLE: model_versions ────────────────────────────────────────────────
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("model_type", _create_enum("model_type"), nullable=False),
        sa.Column("algorithm", _create_enum("model_algorithm"), nullable=False),
        sa.Column("artifact_path", sa.String(500), nullable=False),
        sa.Column("vectorizer_path", sa.String(500), nullable=True),
        sa.Column("dataset_name", sa.String(200), nullable=True),
        sa.Column("training_samples", sa.Integer(), nullable=True),
        sa.Column("test_samples", sa.Integer(), nullable=True),
        sa.Column("accuracy", sa.Float(), nullable=True),
        sa.Column("precision", sa.Float(), nullable=True),
        sa.Column("recall", sa.Float(), nullable=True),
        sa.Column("f1_score", sa.Float(), nullable=True),
        sa.Column("roc_auc", sa.Float(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_name", "version", name="uq_model_name_version"),
        sa.CheckConstraint("accuracy  IS NULL OR (accuracy  BETWEEN 0.0 AND 1.0)",
                           name="ck_mv_accuracy"),
        sa.CheckConstraint("precision IS NULL OR (precision BETWEEN 0.0 AND 1.0)",
                           name="ck_mv_precision"),
        sa.CheckConstraint("recall    IS NULL OR (recall    BETWEEN 0.0 AND 1.0)",
                           name="ck_mv_recall"),
        sa.CheckConstraint("f1_score  IS NULL OR (f1_score  BETWEEN 0.0 AND 1.0)",
                           name="ck_mv_f1"),
        sa.CheckConstraint("roc_auc   IS NULL OR (roc_auc   BETWEEN 0.0 AND 1.0)",
                           name="ck_mv_roc_auc"),
    )
    op.create_index("ix_model_versions_active", "model_versions", ["is_active"])
    op.create_index("ix_model_versions_algorithm", "model_versions", ["algorithm"])

    # ── TABLE: analyses ──────────────────────────────────────────────────────
    op.create_table(
        "analyses",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("input_type", _create_enum("input_type"), nullable=False),
        sa.Column("original_input", sa.Text(), nullable=False),
        sa.Column("source_url", sa.String(2000), nullable=True),
        sa.Column("article_title", sa.String(500), nullable=True),
        sa.Column("article_text", sa.Text(), nullable=True),
        sa.Column("status", _create_enum("analysis_status"),
                  nullable=False, server_default="pending"),
        sa.Column("final_verdict", _create_enum("final_verdict"), nullable=True),
        sa.Column("final_confidence", sa.Float(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("processing_time_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.CheckConstraint(
            "final_confidence IS NULL OR (final_confidence BETWEEN 0.0 AND 1.0)",
            name="ck_analysis_confidence",
        ),
        sa.CheckConstraint(
            "processing_time_ms IS NULL OR processing_time_ms >= 0",
            name="ck_analysis_processing_time",
        ),
    )
    op.create_index("ix_analyses_user_id", "analyses", ["user_id"])
    op.create_index("ix_analyses_user_created", "analyses", ["user_id", "created_at"])
    op.create_index("ix_analyses_status_created", "analyses", ["status", "created_at"])
    op.create_index("ix_analyses_verdict", "analyses", ["final_verdict"])
    op.execute(
        "CREATE INDEX ix_analyses_input_trgm ON analyses "
        "USING GIN (original_input gin_trgm_ops)"
    )

    # ── TABLE: claims ────────────────────────────────────────────────────────
    op.create_table(
        "claims",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("analysis_id", sa.Integer(), nullable=False),
        sa.Column("claim_text", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("verdict", _create_enum("claim_verdict"), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("explanation", sa.Text(), nullable=True),
        sa.Column("top_tokens_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["analysis_id"], ["analyses.id"], ondelete="CASCADE"),
        sa.CheckConstraint(
            "confidence IS NULL OR (confidence BETWEEN 0.0 AND 1.0)",
            name="ck_claim_confidence_range",
        ),
        sa.CheckConstraint("position >= 1", name="ck_claim_position_positive"),
    )
    op.create_index("ix_claims_analysis_id", "claims", ["analysis_id"])
    op.create_index("ix_claims_analysis_position", "claims", ["analysis_id", "position"])
    op.execute(
        "CREATE INDEX ix_claims_text_trgm ON claims "
        "USING GIN (claim_text gin_trgm_ops)"
    )

    # ── TABLE: predictions ───────────────────────────────────────────────────
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("analysis_id", sa.Integer(), nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=False),
        sa.Column("label", _create_enum("prediction_label"), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("fake_probability", sa.Float(), nullable=False),
        sa.Column("real_probability", sa.Float(), nullable=False),
        sa.Column("explanation_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["analysis_id"], ["analyses.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"],
                                ondelete="RESTRICT"),
        sa.CheckConstraint("confidence       BETWEEN 0.0 AND 1.0",
                           name="ck_pred_confidence"),
        sa.CheckConstraint("fake_probability BETWEEN 0.0 AND 1.0",
                           name="ck_pred_fake_prob"),
        sa.CheckConstraint("real_probability BETWEEN 0.0 AND 1.0",
                           name="ck_pred_real_prob"),
    )
    op.create_index("ix_predictions_analysis", "predictions", ["analysis_id"])
    op.create_index("ix_predictions_model_version", "predictions", ["model_version_id"])
    op.create_index("ix_predictions_label", "predictions", ["label"])
    op.create_index("ix_predictions_created", "predictions", ["created_at"])

    # ── TABLE: evidence_sources ──────────────────────────────────────────────
    op.create_table(
        "evidence_sources",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("claim_id", sa.Integer(), nullable=False),
        sa.Column("source_name", sa.String(200), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("url", sa.String(2000), nullable=False),
        sa.Column("snippet", sa.String(500), nullable=True),
        sa.Column("source_type", _create_enum("source_type"), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("relevance_score", sa.Float(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("relationship_to_claim", _create_enum("evidence_relationship"),
                  nullable=False, server_default="inconclusive"),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["claim_id"], ["claims.id"], ondelete="CASCADE"),
        sa.CheckConstraint("relevance_score BETWEEN 0.0 AND 1.0",
                           name="ck_evidence_relevance_range"),
        sa.CheckConstraint("rank >= 1", name="ck_evidence_rank_positive"),
    )
    op.create_index("ix_evidence_claim_rank", "evidence_sources", ["claim_id", "rank"])
    op.create_index("ix_evidence_source_type", "evidence_sources", ["source_type"])
    op.create_index("ix_evidence_relationship", "evidence_sources",
                    ["relationship_to_claim"])
    op.create_index("ix_evidence_relevance", "evidence_sources", ["relevance_score"])

    # ── TABLE: model_runs ────────────────────────────────────────────────────
    op.create_table(
        "model_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("analysis_id", sa.Integer(), nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=False),
        sa.Column("status", _create_enum("run_status"),
                  nullable=False, server_default="success"),
        sa.Column("inference_time_ms", sa.Integer(), nullable=True),
        sa.Column("input_length", sa.Integer(), nullable=True),
        sa.Column("succeeded", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["analysis_id"], ["analyses.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"],
                                ondelete="RESTRICT"),
        sa.CheckConstraint(
            "inference_time_ms IS NULL OR inference_time_ms >= 0",
            name="ck_run_inference_time",
        ),
        sa.CheckConstraint(
            "input_length IS NULL OR input_length >= 0",
            name="ck_run_input_length",
        ),
    )
    op.create_index("ix_model_runs_analysis", "model_runs", ["analysis_id"])
    op.create_index("ix_model_runs_status", "model_runs", ["status"])
    op.create_index("ix_model_runs_model_version", "model_runs", ["model_version_id"])
    op.create_index("ix_model_runs_created", "model_runs", ["created_at"])

    # ── updated_at trigger ───────────────────────────────────────────────────
    op.execute("""
        CREATE OR REPLACE FUNCTION trigger_set_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """)

    for table in [
        "users", "analyses", "claims",
        "model_versions", "predictions",
        "evidence_sources", "model_runs",
    ]:
        op.execute(f"""
            CREATE TRIGGER trg_{table}_updated_at
            BEFORE UPDATE ON {table}
            FOR EACH ROW EXECUTE FUNCTION trigger_set_updated_at()
        """)


def downgrade() -> None:
    # Drop triggers first
    for table in [
        "model_runs", "evidence_sources", "predictions",
        "model_versions", "claims", "analyses", "users",
    ]:
        op.execute(f"DROP TRIGGER IF EXISTS trg_{table}_updated_at ON {table}")

    op.execute("DROP FUNCTION IF EXISTS trigger_set_updated_at()")

    # Drop tables in reverse FK dependency order
    op.drop_table("model_runs")
    op.drop_table("evidence_sources")
    op.drop_table("predictions")
    op.drop_table("claims")
    op.drop_table("analyses")
    op.drop_table("model_versions")
    op.drop_table("users")

    # Drop ENUMs
    for enum_name in [
        "run_status", "evidence_relationship", "source_type",
        "model_algorithm", "model_type", "prediction_label",
        "claim_verdict", "final_verdict", "analysis_status",
        "input_type", "user_role",
    ]:
        sa.Enum(name=enum_name).drop(op.get_bind(), checkfirst=True)

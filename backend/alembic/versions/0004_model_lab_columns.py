"""Add Model Laboratory columns to model_versions.

Adds: random_seed, hyperparameters_json, train_time_s, inference_time_ms,
      model_size_bytes, confusion_matrix_json, training_history_json,
      per_class_metrics_json

Revision ID: 0004
Revises: 0003
Create Date: 2024-07-01 00:00:00.000000
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision:      str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on:    Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("model_versions", sa.Column("random_seed",             sa.Integer(),  nullable=True))
    op.add_column("model_versions", sa.Column("hyperparameters_json",    sa.Text(),     nullable=True))
    op.add_column("model_versions", sa.Column("train_time_s",            sa.Float(),    nullable=True))
    op.add_column("model_versions", sa.Column("inference_time_ms",       sa.Float(),    nullable=True))
    op.add_column("model_versions", sa.Column("model_size_bytes",        sa.Integer(),  nullable=True))
    op.add_column("model_versions", sa.Column("confusion_matrix_json",   sa.Text(),     nullable=True))
    op.add_column("model_versions", sa.Column("training_history_json",   sa.Text(),     nullable=True))
    op.add_column("model_versions", sa.Column("per_class_metrics_json",  sa.Text(),     nullable=True))


def downgrade() -> None:
    for col in [
        "random_seed", "hyperparameters_json", "train_time_s",
        "inference_time_ms", "model_size_bytes", "confusion_matrix_json",
        "training_history_json", "per_class_metrics_json",
    ]:
        op.drop_column("model_versions", col)

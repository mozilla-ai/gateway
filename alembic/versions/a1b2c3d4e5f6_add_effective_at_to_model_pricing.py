"""add effective_at to model_pricing

Revision ID: a1b2c3d4e5f6
Revises: 967575f779b7
Create Date: 2026-04-13 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "967575f779b7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add effective_at column and composite primary key."""

    op.add_column(
        "model_pricing",
        sa.Column("effective_at", sa.DateTime(timezone=True), nullable=True),
    )

    conn = op.get_bind()
    is_sqlite = conn.dialect.name == "sqlite"
    conn.execute(sa.text("UPDATE model_pricing SET effective_at = created_at WHERE effective_at IS NULL"))

    with op.batch_alter_table("model_pricing") as batch_op:
        batch_op.alter_column(
            "effective_at",
            existing_type=sa.DateTime(timezone=True),
            nullable=False,
        )
        if not is_sqlite:
            batch_op.drop_constraint("model_pricing_pkey", type_="primary")
        batch_op.create_primary_key("model_pricing_pkey", ["model_key", "effective_at"])


def downgrade() -> None:
    """Revert to single-column primary key on model_key."""

    conn = op.get_bind()
    is_sqlite = conn.dialect.name == "sqlite"
    conn.execute(
        sa.text(
            """
            DELETE FROM model_pricing
            WHERE effective_at < (
                SELECT MAX(mp2.effective_at)
                FROM model_pricing AS mp2
                WHERE mp2.model_key = model_pricing.model_key
            )
            """
        )
    )

    with op.batch_alter_table("model_pricing") as batch_op:
        if not is_sqlite:
            batch_op.drop_constraint("model_pricing_pkey", type_="primary")
        batch_op.create_primary_key("model_pricing_pkey", ["model_key"])
        batch_op.drop_column("effective_at")

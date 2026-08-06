"""billing & quota: usage_records / quotas / provider_configs

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-06 09:30:00.000000
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from omnistream.db.types import JSONB, ULIDType, UTCDateTime

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "usage_records",
        sa.Column("id", ULIDType(), nullable=False),
        sa.Column("tenant_id", ULIDType(), nullable=False),
        sa.Column("session_id", ULIDType(), nullable=True),
        sa.Column("turn_id", ULIDType(), nullable=True),
        sa.Column("metric", sa.String(length=64), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False, server_default="0"),
        sa.Column("unit", sa.String(length=32), nullable=False, server_default="unit"),
        sa.Column("unit_price", sa.Float(), nullable=False, server_default="0"),
        sa.Column("amount", sa.Float(), nullable=False, server_default="0"),
        sa.Column("currency", sa.String(length=8), nullable=False, server_default="USD"),
        sa.Column("bucket_hour", UTCDateTime(), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=True),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column("is_rollup", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("idempotency_key", sa.String(length=200), nullable=True),
        sa.Column("extra", JSONB(), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_usage_records"),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name="fk_usage_records_tenant_id_tenants", ondelete="CASCADE"
        ),
    )
    op.create_index("ix_usage_records_tenant_id", "usage_records", ["tenant_id"])
    op.create_index("ix_usage_records_session_id", "usage_records", ["session_id"])
    op.create_index("ix_usage_records_metric", "usage_records", ["metric"])
    op.create_index("ix_usage_records_bucket_hour", "usage_records", ["bucket_hour"])
    op.create_index("ix_usage_records_created_at", "usage_records", ["created_at"])
    op.create_index("ix_usage_records_idempotency_key", "usage_records", ["idempotency_key"])
    op.create_index(
        "ix_usage_records_tenant_bucket_metric",
        "usage_records",
        ["tenant_id", "bucket_hour", "metric"],
    )
    op.create_index("ix_usage_records_tenant_rollup", "usage_records", ["tenant_id", "is_rollup"])

    op.create_table(
        "quotas",
        sa.Column("id", ULIDType(), nullable=False),
        sa.Column("tenant_id", ULIDType(), nullable=False),
        sa.Column("policy", JSONB(), nullable=False),
        sa.Column("used", JSONB(), nullable=False),
        sa.Column("period_start", UTCDateTime(), nullable=True),
        sa.Column("period_end", UTCDateTime(), nullable=True),
        sa.Column("period", sa.String(length=16), nullable=False, server_default="monthly"),
        sa.Column("enforced", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("on_exceed", sa.String(length=16), nullable=False, server_default="block"),
        sa.Column("notes", sa.String(length=1000), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_quotas"),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name="fk_quotas_tenant_id_tenants", ondelete="CASCADE"
        ),
        sa.UniqueConstraint("tenant_id", name="uq_quotas_tenant_id"),
    )
    op.create_index("ix_quotas_tenant_id", "quotas", ["tenant_id"])
    op.create_index("ix_quotas_created_at", "quotas", ["created_at"])
    op.create_index("ix_quotas_period_end", "quotas", ["period_end"])

    op.create_table(
        "provider_configs",
        sa.Column("id", ULIDType(), nullable=False),
        sa.Column("tenant_id", ULIDType(), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=200), nullable=True),
        sa.Column("credentials_enc", sa.LargeBinary(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="100"),
        sa.Column("options", JSONB(), nullable=False),
        sa.Column("health", sa.String(length=16), nullable=False, server_default="unknown"),
        sa.Column("last_error", sa.String(length=1000), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_provider_configs"),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name="fk_provider_configs_tenant_id_tenants",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("tenant_id", "provider", name="uq_provider_configs_tenant_id_provider"),
    )
    op.create_index("ix_provider_configs_tenant_id", "provider_configs", ["tenant_id"])
    op.create_index("ix_provider_configs_provider", "provider_configs", ["provider"])
    op.create_index("ix_provider_configs_created_at", "provider_configs", ["created_at"])
    op.create_index(
        "ix_provider_configs_enabled_priority", "provider_configs", ["enabled", "priority"]
    )


def downgrade() -> None:
    op.drop_table("provider_configs")
    op.drop_table("quotas")
    op.drop_table("usage_records")

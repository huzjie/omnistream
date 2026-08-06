"""governance: recordings / quality_scores / audit_logs

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-06 09:20:00.000000
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from omnistream.db.types import JSONB, ULIDType, UTCDateTime

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "recordings",
        sa.Column("id", ULIDType(), nullable=False),
        sa.Column("tenant_id", ULIDType(), nullable=False),
        sa.Column("session_id", ULIDType(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="recording"),
        sa.Column("storage_backend", sa.String(length=16), nullable=False, server_default="local"),
        sa.Column("audio_key", sa.String(length=500), nullable=True),
        sa.Column("video_key", sa.String(length=500), nullable=True),
        sa.Column("events_key", sa.String(length=500), nullable=True),
        sa.Column("manifest_key", sa.String(length=500), nullable=True),
        sa.Column("transcript_key", sa.String(length=500), nullable=True),
        sa.Column("export_key", sa.String(length=500), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("size_bytes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("audio_format", sa.String(length=16), nullable=False, server_default="wav"),
        sa.Column("sample_rate", sa.Integer(), nullable=False, server_default="16000"),
        sa.Column("channels", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("keyframe_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("event_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("encrypted", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("checksum", sa.String(length=128), nullable=True),
        sa.Column("retention_until", UTCDateTime(), nullable=True),
        sa.Column("purged_at", UTCDateTime(), nullable=True),
        sa.Column("error", sa.String(length=1000), nullable=True),
        sa.Column("extra", JSONB(), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_recordings"),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name="fk_recordings_tenant_id_tenants", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["session_id"], ["sessions.id"], name="fk_recordings_session_id_sessions", ondelete="CASCADE"
        ),
    )
    op.create_index("ix_recordings_tenant_id", "recordings", ["tenant_id"])
    op.create_index("ix_recordings_session_id", "recordings", ["session_id"])
    op.create_index("ix_recordings_retention_until", "recordings", ["retention_until"])
    op.create_index("ix_recordings_created_at", "recordings", ["created_at"])
    op.create_index("ix_recordings_tenant_created", "recordings", ["tenant_id", "created_at"])
    op.create_index("ix_recordings_status_retention", "recordings", ["status", "retention_until"])

    op.create_table(
        "quality_scores",
        sa.Column("id", ULIDType(), nullable=False),
        sa.Column("tenant_id", ULIDType(), nullable=False),
        sa.Column("session_id", ULIDType(), nullable=False),
        sa.Column("turn_id", ULIDType(), nullable=True),
        sa.Column("rubric", sa.String(length=64), nullable=False, server_default="duplex_v1"),
        sa.Column("scores", JSONB(), nullable=False),
        sa.Column("total", sa.Float(), nullable=False, server_default="0"),
        sa.Column("reviewer", sa.String(length=32), nullable=False, server_default="rule"),
        sa.Column("reviewer_model", sa.String(length=128), nullable=True),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("flags", JSONB(), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_quality_scores"),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name="fk_quality_scores_tenant_id_tenants", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.id"],
            name="fk_quality_scores_session_id_sessions",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_quality_scores_tenant_id", "quality_scores", ["tenant_id"])
    op.create_index("ix_quality_scores_session_id", "quality_scores", ["session_id"])
    op.create_index("ix_quality_scores_turn_id", "quality_scores", ["turn_id"])
    op.create_index("ix_quality_scores_created_at", "quality_scores", ["created_at"])
    op.create_index("ix_quality_scores_tenant_created", "quality_scores", ["tenant_id", "created_at"])
    op.create_index("ix_quality_scores_rubric_total", "quality_scores", ["rubric", "total"])

    # audit_logs 刻意不建 tenant 外键：允许 ten_system 这类平台级事件
    op.create_table(
        "audit_logs",
        sa.Column("id", ULIDType(), nullable=False),
        sa.Column("tenant_id", ULIDType(), nullable=False),
        sa.Column("seq", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("actor_id", sa.String(length=64), nullable=True),
        sa.Column("actor_type", sa.String(length=32), nullable=False, server_default="system"),
        sa.Column("actor_label", sa.String(length=320), nullable=True),
        sa.Column("action", sa.String(length=128), nullable=False),
        sa.Column("resource_type", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("resource_id", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="success"),
        sa.Column("ip", sa.String(length=64), nullable=True),
        sa.Column("user_agent", sa.String(length=500), nullable=True),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("diff", JSONB(), nullable=False),
        sa.Column("prev_hash", sa.String(length=64), nullable=False, server_default="0" * 64),
        sa.Column("hash", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_audit_logs"),
    )
    op.create_index("ix_audit_logs_tenant_id", "audit_logs", ["tenant_id"])
    op.create_index("ix_audit_logs_actor_id", "audit_logs", ["actor_id"])
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"])
    op.create_index("ix_audit_logs_resource_id", "audit_logs", ["resource_id"])
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])
    op.create_index("ix_audit_logs_tenant_created", "audit_logs", ["tenant_id", "created_at"])
    op.create_index("ix_audit_logs_tenant_seq", "audit_logs", ["tenant_id", "seq"])
    op.create_index("ix_audit_logs_action_created", "audit_logs", ["action", "created_at"])


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("quality_scores")
    op.drop_table("recordings")

"""initial schema: tenants / users / api_keys / sessions / turns / transcripts / vision_events / tool_calls

Revision ID: 0001
Revises:
Create Date: 2026-08-06 09:00:00.000000
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from omnistream.db.types import JSONB, ULIDType, UTCDateTime

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tenants",
        sa.Column("id", ULIDType(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("slug", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="active"),
        sa.Column("plan", sa.String(length=64), nullable=False, server_default="free"),
        sa.Column("contact_email", sa.String(length=320), nullable=True),
        sa.Column("description", sa.String(length=1000), nullable=True),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("settings", JSONB(), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_tenants"),
        sa.UniqueConstraint("slug", name="uq_tenants_slug"),
    )
    op.create_index("ix_tenants_slug", "tenants", ["slug"])
    op.create_index("ix_tenants_created_at", "tenants", ["created_at"])
    op.create_index("ix_tenants_deleted_at", "tenants", ["deleted_at"])
    op.create_index("ix_tenants_status_created", "tenants", ["status", "created_at"])

    op.create_table(
        "users",
        sa.Column("id", ULIDType(), nullable=False),
        sa.Column("tenant_id", ULIDType(), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=512), nullable=True),
        sa.Column("display_name", sa.String(length=200), nullable=False, server_default=""),
        sa.Column("role", sa.String(length=32), nullable=False, server_default="viewer"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="active"),
        sa.Column("is_service_account", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("last_login_at", UTCDateTime(), nullable=True),
        sa.Column("failed_login_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("locked_until", UTCDateTime(), nullable=True),
        sa.Column("preferences", JSONB(), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name="fk_users_tenant_id_tenants", ondelete="CASCADE"
        ),
        sa.UniqueConstraint("tenant_id", "email", name="uq_users_tenant_id_email"),
    )
    op.create_index("ix_users_tenant_id", "users", ["tenant_id"])
    op.create_index("ix_users_created_at", "users", ["created_at"])
    op.create_index("ix_users_deleted_at", "users", ["deleted_at"])
    op.create_index("ix_users_tenant_role", "users", ["tenant_id", "role"])

    op.create_table(
        "api_keys",
        sa.Column("id", ULIDType(), nullable=False),
        sa.Column("tenant_id", ULIDType(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False, server_default="default"),
        sa.Column("prefix", sa.String(length=64), nullable=False),
        sa.Column("key_hash", sa.String(length=512), nullable=False),
        sa.Column("scopes", JSONB(), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False, server_default="developer"),
        sa.Column("environment", sa.String(length=16), nullable=False, server_default="live"),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("expires_at", UTCDateTime(), nullable=True),
        sa.Column("last_used_at", UTCDateTime(), nullable=True),
        sa.Column("last_used_ip", sa.String(length=64), nullable=True),
        sa.Column("usage_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("revoked_at", UTCDateTime(), nullable=True),
        sa.Column("revoked_reason", sa.String(length=500), nullable=True),
        sa.Column("created_by", sa.String(length=32), nullable=True),
        sa.Column("description", sa.String(length=1000), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_api_keys"),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name="fk_api_keys_tenant_id_tenants", ondelete="CASCADE"
        ),
        sa.UniqueConstraint("prefix", name="uq_api_keys_prefix"),
    )
    op.create_index("ix_api_keys_prefix", "api_keys", ["prefix"])
    op.create_index("ix_api_keys_tenant_id", "api_keys", ["tenant_id"])
    op.create_index("ix_api_keys_expires_at", "api_keys", ["expires_at"])
    op.create_index("ix_api_keys_created_at", "api_keys", ["created_at"])
    op.create_index("ix_api_keys_deleted_at", "api_keys", ["deleted_at"])
    op.create_index("ix_api_keys_tenant_enabled", "api_keys", ["tenant_id", "enabled"])

    op.create_table(
        "sessions",
        sa.Column("id", ULIDType(), nullable=False),
        sa.Column("tenant_id", ULIDType(), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False, server_default="mock"),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column("state", sa.String(length=32), nullable=False, server_default="created"),
        sa.Column("modalities", JSONB(), nullable=False),
        sa.Column("config", JSONB(), nullable=False),
        sa.Column("stats", JSONB(), nullable=False),
        sa.Column("metadata", JSONB(), nullable=False),
        sa.Column("started_at", UTCDateTime(), nullable=True),
        sa.Column("ended_at", UTCDateTime(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("stop_reason", sa.String(length=64), nullable=True),
        sa.Column("end_user_id", sa.String(length=200), nullable=True),
        sa.Column("api_key_id", sa.String(length=32), nullable=True),
        sa.Column("user_id", sa.String(length=32), nullable=True),
        sa.Column("client_ip", sa.String(length=64), nullable=True),
        sa.Column("user_agent", sa.String(length=500), nullable=True),
        sa.Column("record_enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("redact_pii", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("turn_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("barge_in_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("audio_in_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("audio_out_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("video_frames", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("avg_ttfa_ms", sa.Float(), nullable=False, server_default="0"),
        sa.Column("error_code", sa.String(length=32), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_sessions"),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name="fk_sessions_tenant_id_tenants", ondelete="CASCADE"
        ),
    )
    op.create_index("ix_sessions_tenant_id", "sessions", ["tenant_id"])
    op.create_index("ix_sessions_state", "sessions", ["state"])
    op.create_index("ix_sessions_created_at", "sessions", ["created_at"])
    op.create_index("ix_sessions_end_user_id", "sessions", ["end_user_id"])
    op.create_index("ix_sessions_tenant_created", "sessions", ["tenant_id", "created_at"])
    op.create_index("ix_sessions_tenant_state", "sessions", ["tenant_id", "state"])
    op.create_index("ix_sessions_provider", "sessions", ["provider"])

    op.create_table(
        "turns",
        sa.Column("id", ULIDType(), nullable=False),
        sa.Column("tenant_id", ULIDType(), nullable=False),
        sa.Column("session_id", ULIDType(), nullable=False),
        sa.Column("index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("role", sa.String(length=16), nullable=False, server_default="user"),
        sa.Column("state", sa.String(length=32), nullable=False, server_default="idle"),
        sa.Column("text", sa.Text(), nullable=False, server_default=""),
        sa.Column("audio_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("started_at", UTCDateTime(), nullable=True),
        sa.Column("ended_at", UTCDateTime(), nullable=True),
        sa.Column("stop_reason", sa.String(length=32), nullable=True),
        sa.Column("interrupted_at_ms", sa.Integer(), nullable=True),
        sa.Column("metrics", JSONB(), nullable=False),
        sa.Column("ttft_ms", sa.Float(), nullable=True),
        sa.Column("ttfa_ms", sa.Float(), nullable=True),
        sa.Column("e2e_ms", sa.Float(), nullable=True),
        sa.Column("barge_in_ms", sa.Float(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tool_call_ids", JSONB(), nullable=False),
        sa.Column("memory_injected_ids", JSONB(), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_turns"),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name="fk_turns_tenant_id_tenants", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["session_id"], ["sessions.id"], name="fk_turns_session_id_sessions", ondelete="CASCADE"
        ),
        sa.UniqueConstraint("session_id", "index", name="uq_turns_session_id_index"),
    )
    op.create_index("ix_turns_tenant_id", "turns", ["tenant_id"])
    op.create_index("ix_turns_session_id", "turns", ["session_id"])
    op.create_index("ix_turns_created_at", "turns", ["created_at"])
    op.create_index("ix_turns_tenant_created", "turns", ["tenant_id", "created_at"])

    op.create_table(
        "transcripts",
        sa.Column("id", ULIDType(), nullable=False),
        sa.Column("tenant_id", ULIDType(), nullable=False),
        sa.Column("turn_id", ULIDType(), nullable=False),
        sa.Column("session_id", ULIDType(), nullable=False),
        sa.Column("speaker", sa.String(length=32), nullable=False, server_default="user"),
        sa.Column("text", sa.Text(), nullable=False, server_default=""),
        sa.Column("redacted_text", sa.Text(), nullable=True),
        sa.Column("language", sa.String(length=16), nullable=False, server_default="zh-CN"),
        sa.Column("words", JSONB(), nullable=False),
        sa.Column("start_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("end_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1"),
        sa.Column("is_final", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("source", sa.String(length=32), nullable=False, server_default="realtime"),
        sa.Column("engine", sa.String(length=64), nullable=True),
        sa.Column("pii_hits", JSONB(), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_transcripts"),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name="fk_transcripts_tenant_id_tenants", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["turn_id"], ["turns.id"], name="fk_transcripts_turn_id_turns", ondelete="CASCADE"
        ),
    )
    op.create_index("ix_transcripts_tenant_id", "transcripts", ["tenant_id"])
    op.create_index("ix_transcripts_turn_id", "transcripts", ["turn_id"])
    op.create_index("ix_transcripts_session_id", "transcripts", ["session_id"])
    op.create_index("ix_transcripts_created_at", "transcripts", ["created_at"])
    op.create_index("ix_transcripts_session_start", "transcripts", ["session_id", "start_ms"])
    op.create_index("ix_transcripts_tenant_created", "transcripts", ["tenant_id", "created_at"])

    op.create_table(
        "vision_events",
        sa.Column("id", ULIDType(), nullable=False),
        sa.Column("tenant_id", ULIDType(), nullable=False),
        sa.Column("session_id", ULIDType(), nullable=False),
        sa.Column("turn_id", ULIDType(), nullable=True),
        sa.Column("pts_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("entities", JSONB(), nullable=False),
        sa.Column("frame_key", sa.String(length=500), nullable=True),
        sa.Column("keyframe_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("scene_change", sa.Float(), nullable=False, server_default="0"),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=False, server_default="0"),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_vision_events"),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name="fk_vision_events_tenant_id_tenants", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.id"],
            name="fk_vision_events_session_id_sessions",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_vision_events_tenant_id", "vision_events", ["tenant_id"])
    op.create_index("ix_vision_events_session_id", "vision_events", ["session_id"])
    op.create_index("ix_vision_events_turn_id", "vision_events", ["turn_id"])
    op.create_index("ix_vision_events_created_at", "vision_events", ["created_at"])
    op.create_index("ix_vision_events_session_pts", "vision_events", ["session_id", "pts_ms"])

    op.create_table(
        "tool_calls",
        sa.Column("id", ULIDType(), nullable=False),
        sa.Column("tenant_id", ULIDType(), nullable=False),
        sa.Column("turn_id", ULIDType(), nullable=False),
        sa.Column("session_id", ULIDType(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("arguments", JSONB(), nullable=False),
        sa.Column("result", JSONB(), nullable=True),
        sa.Column("state", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("ok", sa.Boolean(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("error_code", sa.String(length=32), nullable=True),
        sa.Column("elapsed_ms", sa.Float(), nullable=False, server_default="0"),
        sa.Column("started_at", UTCDateTime(), nullable=True),
        sa.Column("finished_at", UTCDateTime(), nullable=True),
        sa.Column("requires_confirmation", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("confirmed_by", sa.String(length=32), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_tool_calls"),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name="fk_tool_calls_tenant_id_tenants", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["turn_id"], ["turns.id"], name="fk_tool_calls_turn_id_turns", ondelete="CASCADE"
        ),
    )
    op.create_index("ix_tool_calls_tenant_id", "tool_calls", ["tenant_id"])
    op.create_index("ix_tool_calls_turn_id", "tool_calls", ["turn_id"])
    op.create_index("ix_tool_calls_session_id", "tool_calls", ["session_id"])
    op.create_index("ix_tool_calls_name", "tool_calls", ["name"])
    op.create_index("ix_tool_calls_created_at", "tool_calls", ["created_at"])
    op.create_index("ix_tool_calls_tenant_name", "tool_calls", ["tenant_id", "name"])
    op.create_index("ix_tool_calls_tenant_created", "tool_calls", ["tenant_id", "created_at"])


def downgrade() -> None:
    op.drop_table("tool_calls")
    op.drop_table("vision_events")
    op.drop_table("transcripts")
    op.drop_table("turns")
    op.drop_table("sessions")
    op.drop_table("api_keys")
    op.drop_table("users")
    op.drop_table("tenants")

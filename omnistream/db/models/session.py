"""``sessions`` 表：实时会话主表。

注意 §7.8：Pydantic 侧同名模型为 ``omnistream.schemas.session.Session``，
导入本模块时请用别名 ``from omnistream.db.models.session import Session as SessionORM``。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, Float, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from omnistream.db.base import Base
from omnistream.db.mixins import TenantMixin, TimestampMixin, ULIDPkMixin
from omnistream.db.types import JSONB, UTCDateTime

#: 视为"仍在计费/占用并发"的状态
LIVE_STATES: frozenset[str] = frozenset({"created", "active", "ending"})


class Session(ULIDPkMixin, TenantMixin, TimestampMixin, Base):
    __tablename__ = "sessions"
    __id_prefix__ = "sess_"

    provider: Mapped[str] = mapped_column(String(64), nullable=False, default="mock")
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    state: Mapped[str] = mapped_column(String(32), nullable=False, default="created", index=True)
    modalities: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=lambda: ["text", "audio"])
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    stats: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)

    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    duration_ms: Mapped[int] = mapped_column(nullable=False, default=0)
    stop_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)

    end_user_id: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    api_key_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    user_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    client_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(500), nullable=True)

    record_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    redact_pii: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    turn_count: Mapped[int] = mapped_column(nullable=False, default=0)
    barge_in_count: Mapped[int] = mapped_column(nullable=False, default=0)
    input_tokens: Mapped[int] = mapped_column(nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(nullable=False, default=0)
    audio_in_ms: Mapped[int] = mapped_column(nullable=False, default=0)
    audio_out_ms: Mapped[int] = mapped_column(nullable=False, default=0)
    video_frames: Mapped[int] = mapped_column(nullable=False, default=0)
    avg_ttfa_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    error_code: Mapped[str | None] = mapped_column(String(32), nullable=True)

    __table_args__ = (
        Index("ix_sessions_tenant_created", "tenant_id", "created_at"),
        Index("ix_sessions_tenant_state", "tenant_id", "state"),
        Index("ix_sessions_provider", "provider"),
    )

    @property
    def is_live(self) -> bool:
        return self.state in LIVE_STATES


__all__ = ["LIVE_STATES", "Session"]

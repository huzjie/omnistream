"""``recordings`` 表：会话录制资产。"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import Boolean, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from omnistream.db.base import Base
from omnistream.db.mixins import TenantMixin, TimestampMixin, ULIDPkMixin
from omnistream.db.types import JSONB, ULIDType, UTCDateTime


class RecordingStatus(StrEnum):
    RECORDING = "recording"
    FINALIZING = "finalizing"
    READY = "ready"
    FAILED = "failed"
    PURGED = "purged"


class Recording(ULIDPkMixin, TenantMixin, TimestampMixin, Base):
    __tablename__ = "recordings"
    __id_prefix__ = "rec_"

    session_id: Mapped[str] = mapped_column(
        ULIDType, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=RecordingStatus.RECORDING.value)
    storage_backend: Mapped[str] = mapped_column(String(16), nullable=False, default="local")

    audio_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    video_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    events_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    manifest_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    transcript_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    export_key: Mapped[str | None] = mapped_column(String(500), nullable=True)

    duration_ms: Mapped[int] = mapped_column(nullable=False, default=0)
    size_bytes: Mapped[int] = mapped_column(nullable=False, default=0)
    audio_format: Mapped[str] = mapped_column(String(16), nullable=False, default="wav")
    sample_rate: Mapped[int] = mapped_column(nullable=False, default=16000)
    channels: Mapped[int] = mapped_column(nullable=False, default=1)
    keyframe_count: Mapped[int] = mapped_column(nullable=False, default=0)
    event_count: Mapped[int] = mapped_column(nullable=False, default=0)

    encrypted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    retention_until: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True, index=True)
    purged_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    error: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    extra: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    __table_args__ = (
        Index("ix_recordings_tenant_created", "tenant_id", "created_at"),
        Index("ix_recordings_status_retention", "status", "retention_until"),
    )

    @property
    def is_ready(self) -> bool:
        return self.status == RecordingStatus.READY.value


__all__ = ["Recording", "RecordingStatus"]

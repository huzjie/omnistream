"""``transcripts`` 表：转写落库（含词级时间戳）。"""
from __future__ import annotations

from typing import Any

from sqlalchemy import Boolean, Float, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from omnistream.db.base import Base
from omnistream.db.mixins import TenantMixin, TimestampMixin, ULIDPkMixin
from omnistream.db.types import JSONB, ULIDType


class Transcript(ULIDPkMixin, TenantMixin, TimestampMixin, Base):
    __tablename__ = "transcripts"
    __id_prefix__ = "tsc_"

    turn_id: Mapped[str] = mapped_column(
        ULIDType, ForeignKey("turns.id", ondelete="CASCADE"), nullable=False, index=True
    )
    session_id: Mapped[str] = mapped_column(ULIDType, nullable=False, index=True)
    speaker: Mapped[str] = mapped_column(String(32), nullable=False, default="user")
    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    #: PII 脱敏后的文本；未脱敏时为 None
    redacted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str] = mapped_column(String(16), nullable=False, default="zh-CN")
    #: [{"w": "你好", "s": 120, "e": 380, "c": 0.93}, ...]
    words: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    start_ms: Mapped[int] = mapped_column(nullable=False, default=0)
    end_ms: Mapped[int] = mapped_column(nullable=False, default=0)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    is_final: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    #: realtime | offline_rescore | manual
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="realtime")
    engine: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pii_hits: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)

    __table_args__ = (
        Index("ix_transcripts_session_start", "session_id", "start_ms"),
        Index("ix_transcripts_tenant_created", "tenant_id", "created_at"),
    )

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def display_text(self) -> str:
        """对外展示优先用脱敏文本。"""
        return self.redacted_text if self.redacted_text is not None else self.text


__all__ = ["Transcript"]

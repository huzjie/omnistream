"""``turns`` 表：轮次与延迟指标。"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Float, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from omnistream.db.base import Base
from omnistream.db.mixins import TenantMixin, TimestampMixin, ULIDPkMixin
from omnistream.db.types import JSONB, ULIDType, UTCDateTime


class Turn(ULIDPkMixin, TenantMixin, TimestampMixin, Base):
    """一次"用户说 → 助手答"的完整往返（或其中一半）。"""

    __tablename__ = "turns"
    __id_prefix__ = "turn_"

    session_id: Mapped[str] = mapped_column(
        ULIDType, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    index: Mapped[int] = mapped_column(nullable=False, default=0)
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="user")
    state: Mapped[str] = mapped_column(String(32), nullable=False, default="idle")
    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    audio_ms: Mapped[int] = mapped_column(nullable=False, default=0)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    stop_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    interrupted_at_ms: Mapped[int | None] = mapped_column(nullable=True)

    #: 完整 TurnMetrics 快照（omnistream.schemas.turn.TurnMetrics 的 JSON 形态）
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    #: 冗余出的关键 SLO 列，便于直接聚合，无需展开 JSONB
    ttft_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    ttfa_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    e2e_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    barge_in_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    input_tokens: Mapped[int] = mapped_column(nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(nullable=False, default=0)

    tool_call_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    memory_injected_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)

    __table_args__ = (
        UniqueConstraint("session_id", "index", name="uq_turns_session_id_index"),
        Index("ix_turns_tenant_created", "tenant_id", "created_at"),
    )

    def sync_metrics_columns(self) -> None:
        """把 metrics JSONB 中的关键项同步到冗余列（写入前调用）。"""
        metrics = self.metrics or {}
        for key in ("ttft_ms", "ttfa_ms", "e2e_ms", "barge_in_ms"):
            value = metrics.get(key)
            setattr(self, key, float(value) if isinstance(value, (int, float)) else None)
        self.input_tokens = int(metrics.get("input_tokens") or 0)
        self.output_tokens = int(metrics.get("output_tokens") or 0)


__all__ = ["Turn"]

"""``quality_scores`` 表：会话/轮次维度质检分。"""
from __future__ import annotations

from typing import Any

from sqlalchemy import Float, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from omnistream.db.base import Base
from omnistream.db.mixins import TenantMixin, TimestampMixin, ULIDPkMixin
from omnistream.db.types import JSONB, ULIDType


class QualityScore(ULIDPkMixin, TenantMixin, TimestampMixin, Base):
    __tablename__ = "quality_scores"
    __id_prefix__ = "qsc_"

    session_id: Mapped[str] = mapped_column(
        ULIDType, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    turn_id: Mapped[str | None] = mapped_column(ULIDType, nullable=True, index=True)
    #: 评分标准集名称，如 ``duplex_v1``
    rubric: Mapped[str] = mapped_column(String(64), nullable=False, default="duplex_v1")
    #: {"barge_in_grace": 0.9, "silence_ratio": 0.7, "compliance": 1.0, ...}
    scores: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    total: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    #: rule | llm | human
    reviewer: Mapped[str] = mapped_column(String(32), nullable=False, default="rule")
    reviewer_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: 触发的违规/告警项
    flags: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)

    __table_args__ = (
        Index("ix_quality_scores_tenant_created", "tenant_id", "created_at"),
        Index("ix_quality_scores_rubric_total", "rubric", "total"),
    )


__all__ = ["QualityScore"]

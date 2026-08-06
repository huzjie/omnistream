"""``vision_events`` 表：视觉观察事件。"""
from __future__ import annotations

from typing import Any

from sqlalchemy import Float, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from omnistream.db.base import Base
from omnistream.db.mixins import TenantMixin, TimestampMixin, ULIDPkMixin
from omnistream.db.types import JSONB, ULIDType


class VisionEvent(ULIDPkMixin, TenantMixin, TimestampMixin, Base):
    __tablename__ = "vision_events"
    __id_prefix__ = "vis_"

    session_id: Mapped[str] = mapped_column(
        ULIDType, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    turn_id: Mapped[str | None] = mapped_column(ULIDType, nullable=True, index=True)
    pts_ms: Mapped[int] = mapped_column(nullable=False, default=0)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    #: [{"label": "laptop", "conf": 0.88, "bbox": [x, y, w, h]}, ...]
    entities: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    #: 对象存储中的关键帧 key（JPEG）
    frame_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    keyframe_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    scene_change: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    __table_args__ = (
        Index("ix_vision_events_session_pts", "session_id", "pts_ms"),
    )


__all__ = ["VisionEvent"]

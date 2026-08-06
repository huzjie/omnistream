"""``tool_calls`` 表：实时工具调用记录。"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Float, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from omnistream.db.base import Base
from omnistream.db.mixins import TenantMixin, TimestampMixin, ULIDPkMixin
from omnistream.db.types import JSONB, ULIDType, UTCDateTime


class ToolCall(ULIDPkMixin, TenantMixin, TimestampMixin, Base):
    __tablename__ = "tool_calls"
    __id_prefix__ = "call_"

    turn_id: Mapped[str] = mapped_column(
        ULIDType, ForeignKey("turns.id", ondelete="CASCADE"), nullable=False, index=True
    )
    session_id: Mapped[str] = mapped_column(ULIDType, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    arguments: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    #: 取值见 omnistream.protocol.enums.ToolCallState
    state: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    ok: Mapped[bool | None] = mapped_column(nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    elapsed_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    requires_confirmation: Mapped[bool] = mapped_column(nullable=False, default=False)
    confirmed_by: Mapped[str | None] = mapped_column(String(32), nullable=True)

    __table_args__ = (
        Index("ix_tool_calls_tenant_name", "tenant_id", "name"),
        Index("ix_tool_calls_tenant_created", "tenant_id", "created_at"),
    )


__all__ = ["ToolCall"]

"""``memory_items`` 表：主动记忆主表（含向量列）。"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Float, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from omnistream.db.base import Base
from omnistream.db.mixins import SoftDeleteMixin, TenantMixin, TimestampMixin, ULIDPkMixin
from omnistream.db.types import DEFAULT_VECTOR_DIM, JSONB, ULIDType, UTCDateTime, VectorType


class MemoryItem(ULIDPkMixin, TenantMixin, TimestampMixin, SoftDeleteMixin, Base):
    """一条记忆。``layer`` 取值见 ``omnistream.protocol.enums.MemoryLayer``。"""

    __tablename__ = "memory_items"
    __id_prefix__ = "mem_"

    layer: Mapped[str] = mapped_column(String(32), nullable=False, default="session", index=True)
    #: private_state 层的键，同租户同键写覆盖
    key: Mapped[str | None] = mapped_column(String(200), nullable=True)
    subject: Mapped[str] = mapped_column(String(64), nullable=False, default="user")
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    embedding: Mapped[list[float] | None] = mapped_column(
        VectorType(DEFAULT_VECTOR_DIM), nullable=True
    )
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.8)
    importance: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    hit_count: Mapped[int] = mapped_column(nullable=False, default=0)
    last_hit_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True, index=True)
    source_session_id: Mapped[str | None] = mapped_column(ULIDType, nullable=True, index=True)
    end_user_id: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    tags: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    #: 被压缩合并后指向的父记忆（memory_compaction 任务写入）
    superseded_by: Mapped[str | None] = mapped_column(ULIDType, nullable=True)

    __table_args__ = (
        Index("ix_memory_items_tenant_layer", "tenant_id", "layer"),
        Index("ix_memory_items_tenant_key", "tenant_id", "key"),
        Index("ix_memory_items_tenant_subject", "tenant_id", "subject"),
    )

    def is_expired(self, now: datetime) -> bool:
        return self.expires_at is not None and self.expires_at <= now

    def decayed_importance(self, now: datetime, half_life_days: float = 30.0) -> float:
        """按半衰期做重要度衰减，命中次数提供正向补偿。"""
        age_days = max(0.0, (now - self.created_at).total_seconds() / 86400.0)
        decay = 0.5 ** (age_days / max(half_life_days, 0.1))
        boost = min(0.3, 0.03 * float(self.hit_count))
        return max(0.0, min(1.0, self.importance * decay + boost))


__all__ = ["MemoryItem"]

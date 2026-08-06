"""``quotas`` 表：租户级配额策略与当前周期用量快照。"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from omnistream.db.base import Base
from omnistream.db.mixins import TenantMixin, TimestampMixin, ULIDPkMixin
from omnistream.db.types import JSONB, UTCDateTime

#: 默认配额策略。0 或负数表示不限制。
DEFAULT_QUOTA_POLICY: dict[str, Any] = {
    "max_concurrent_sessions": 20,
    "max_sessions_per_day": 5000,
    "max_session_minutes": 60,
    "monthly_audio_minutes": 10000,
    "daily_llm_tokens": 5_000_000,
    "monthly_storage_gb": 50,
    "max_api_keys": 20,
    "max_users": 50,
    "rate_limit_rps": 20.0,
    "rate_limit_burst": 40,
    "ws_connections_per_key": 50,
}


class Quota(ULIDPkMixin, TenantMixin, TimestampMixin, Base):
    __tablename__ = "quotas"
    __id_prefix__ = "quo_"

    policy: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=lambda: dict(DEFAULT_QUOTA_POLICY)
    )
    #: 当前周期累计用量：{"audio_minutes": 12.5, "llm_tokens": 88123, ...}
    used: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    period_start: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    period_end: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    #: monthly | daily
    period: Mapped[str] = mapped_column(String(16), nullable=False, default="monthly")
    enforced: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    #: 超额时的行为：block | degrade | notify
    on_exceed: Mapped[str] = mapped_column(String(16), nullable=False, default="block")
    notes: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    __table_args__ = (
        UniqueConstraint("tenant_id", name="uq_quotas_tenant_id"),
        Index("ix_quotas_period_end", "period_end"),
    )

    def limit_of(self, key: str) -> float:
        raw = (self.policy or {}).get(key, DEFAULT_QUOTA_POLICY.get(key, 0))
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    def used_of(self, key: str) -> float:
        try:
            return float((self.used or {}).get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0

    def add_used(self, key: str, delta: float) -> float:
        current = dict(self.used or {})
        value = float(current.get(key, 0.0) or 0.0) + float(delta)
        current[key] = value
        self.used = current
        return value


__all__ = ["DEFAULT_QUOTA_POLICY", "Quota"]

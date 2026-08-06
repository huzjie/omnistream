"""``api_keys`` 表：机器到机器鉴权凭据。

明文密钥形如 ``oms_live_<32 位随机>``，**只在签发时返回一次**，
库里只存 ``prefix``（用于定位）与 ``key_hash``（用于校验）。
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from omnistream.db.base import Base
from omnistream.db.mixins import SoftDeleteMixin, TenantMixin, TimestampMixin, ULIDPkMixin
from omnistream.db.types import JSONB, UTCDateTime


class ApiKey(ULIDPkMixin, TenantMixin, TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "api_keys"
    __id_prefix__ = "ak_"

    name: Mapped[str] = mapped_column(String(200), nullable=False, default="default")
    #: 明文密钥的可见前缀，如 ``oms_live_A1B2C3D4``（前 8 位随机部分）
    prefix: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    key_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    scopes: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="developer")
    environment: Mapped[str] = mapped_column(String(16), nullable=False, default="live")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True, index=True)
    last_used_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    last_used_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    usage_count: Mapped[int] = mapped_column(nullable=False, default=0)
    revoked_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    revoked_reason: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    description: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    __table_args__ = (
        Index("ix_api_keys_tenant_enabled", "tenant_id", "enabled"),
    )

    def is_expired(self, now: datetime) -> bool:
        return self.expires_at is not None and self.expires_at <= now

    def is_revoked(self) -> bool:
        return self.revoked_at is not None or not self.enabled or self.deleted_at is not None

    def is_usable(self, now: datetime) -> bool:
        return not self.is_revoked() and not self.is_expired(now)


__all__ = ["ApiKey"]

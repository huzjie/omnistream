"""``users`` 表：控制台用户。"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import Boolean, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from omnistream.db.base import Base
from omnistream.db.mixins import SoftDeleteMixin, TenantMixin, TimestampMixin, ULIDPkMixin
from omnistream.db.types import JSONB, UTCDateTime


class UserStatus(StrEnum):
    ACTIVE = "active"
    INVITED = "invited"
    DISABLED = "disabled"


class User(ULIDPkMixin, TenantMixin, TimestampMixin, SoftDeleteMixin, Base):
    """租户下的控制台成员。角色语义见 ``omnistream.platform.rbac.roles``。"""

    __tablename__ = "users"
    __id_prefix__ = "usr_"

    email: Mapped[str] = mapped_column(String(320), nullable=False)
    password_hash: Mapped[str | None] = mapped_column(String(512), nullable=True)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="viewer")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=UserStatus.ACTIVE.value)
    is_service_account: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    last_login_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    failed_login_count: Mapped[int] = mapped_column(nullable=False, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    preferences: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    __table_args__ = (
        UniqueConstraint("tenant_id", "email", name="uq_users_tenant_id_email"),
        Index("ix_users_tenant_role", "tenant_id", "role"),
    )

    @property
    def active(self) -> bool:
        return self.status == UserStatus.ACTIVE.value and self.deleted_at is None

    def is_locked(self, now: datetime) -> bool:
        return self.locked_until is not None and self.locked_until > now


__all__ = ["User", "UserStatus"]

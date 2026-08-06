"""``tenants`` 表：多租户根实体。"""
from __future__ import annotations

from enum import StrEnum
from typing import Any

from sqlalchemy import Boolean, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from omnistream.db.base import Base
from omnistream.db.mixins import SoftDeleteMixin, TimestampMixin, ULIDPkMixin
from omnistream.db.types import JSONB


class TenantStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ARCHIVED = "archived"


DEFAULT_TENANT_SETTINGS: dict[str, Any] = {
    "webhooks": {"endpoints": []},
    "privacy": {"redact_pii": True, "sensitive_words": []},
    "retention": {"recording_days": 30, "transcript_days": 180, "audit_days": 365},
    "branding": {"display_name": "", "logo_url": ""},
}


class Tenant(ULIDPkMixin, TimestampMixin, SoftDeleteMixin, Base):
    """租户。除本表外所有业务表都必须带 ``tenant_id``。"""

    __tablename__ = "tenants"
    __id_prefix__ = "ten_"

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=TenantStatus.ACTIVE.value)
    plan: Mapped[str] = mapped_column(String(64), nullable=False, default="free")
    contact_email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    description: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    settings: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=lambda: dict(DEFAULT_TENANT_SETTINGS)
    )

    __table_args__ = (
        Index("ix_tenants_status_created", "status", "created_at"),
    )

    @property
    def active(self) -> bool:
        return self.status == TenantStatus.ACTIVE.value and self.deleted_at is None

    def setting_path(self, path: str, default: Any = None) -> Any:  # noqa: ANN401
        """按点分路径读取 settings JSONB。"""
        node: Any = self.settings or {}
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_setting_path(self, path: str, value: Any) -> None:  # noqa: ANN401
        """按点分路径写入 settings JSONB（整体替换以触发 ORM 变更检测）。"""
        root: dict[str, Any] = dict(self.settings or {})
        node = root
        parts = path.split(".")
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
            node[part] = dict(child)
            node = node[part]
        node[parts[-1]] = value
        self.settings = root


__all__ = ["DEFAULT_TENANT_SETTINGS", "Tenant", "TenantStatus"]

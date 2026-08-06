"""租户领域模型。所有业务数据以 tenant_id 为硬隔离边界。"""
from __future__ import annotations

from datetime import datetime

from pydantic import Field, field_validator

from omnistream.core.clock import utcnow
from omnistream.schemas.base import OmniModel

_SLUG_OK = set("abcdefghijklmnopqrstuvwxyz0123456789-")


class TenantSettings(OmniModel):
    """租户级默认值与开关，会话创建时作为 SessionConfig 的兜底。"""

    default_provider: str = "mock"
    default_model: str | None = None
    default_voice: str = "default"
    default_language: str = "zh-CN"
    allowed_providers: list[str] = Field(default_factory=list,
                                         description="为空表示允许全部已启用 provider")
    record_by_default: bool = False
    redact_pii: bool = True
    #: 录制与转写的保留天数，0 表示不保留
    retention_days: int = Field(default=30, ge=0, le=3650)
    memory_enabled: bool = True
    max_concurrent_sessions: int = Field(default=20, ge=1)
    webhook_url: str | None = None
    branding: dict[str, str] = Field(default_factory=dict)


class Tenant(OmniModel):
    id: str
    slug: str = Field(description="URL 安全短名，全局唯一")
    name: str
    status: str = Field(default="active", description="active|suspended|deleted")
    settings: TenantSettings = Field(default_factory=TenantSettings)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("slug")
    @classmethod
    def _slug(cls, v: str) -> str:
        low = v.strip().lower()
        if not low or len(low) > 48:
            raise ValueError("slug 长度必须在 1~48 之间")
        if set(low) - _SLUG_OK:
            raise ValueError(f"slug 含非法字符: {v!r}（仅允许小写字母、数字、连字符）")
        if low.startswith("-") or low.endswith("-"):
            raise ValueError("slug 不能以连字符开头或结尾")
        return low

    @property
    def active(self) -> bool:
        return self.status == "active"


class TenantCreate(OmniModel):
    slug: str
    name: str
    settings: TenantSettings = Field(default_factory=TenantSettings)
    metadata: dict[str, str] = Field(default_factory=dict)


class TenantUpdate(OmniModel):
    name: str | None = None
    status: str | None = None
    settings: TenantSettings | None = None
    metadata: dict[str, str] | None = None

    def apply_to(self, tenant: Tenant) -> Tenant:
        patch = self.model_dump(exclude_none=True)
        patch["updated_at"] = utcnow()
        return tenant.model_copy(update=patch, deep=True)

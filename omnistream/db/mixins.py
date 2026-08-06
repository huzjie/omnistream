"""ORM 模型混入：主键 / 时间戳 / 租户 / 软删除。

约定（§3.12 尾注）：
* 全表继承 ``TimestampMixin``；
* 除 ``tenants`` 外全部继承 ``TenantMixin``，且必须经 ``TenantScopedRepository`` 访问。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.orm import Mapped, declared_attr, mapped_column

from omnistream.db.types import ULIDType, UTCDateTime, generate_id, utc_now


class ULIDPkMixin:
    """带前缀 ULID 主键。子类通过 ``__id_prefix__`` 声明前缀。"""

    __id_prefix__: ClassVar[str] = "obj_"

    @declared_attr
    @classmethod
    def id(cls) -> Mapped[str]:
        prefix = getattr(cls, "__id_prefix__", "obj_")
        return mapped_column(
            ULIDType,
            primary_key=True,
            default=lambda: generate_id(prefix),
            doc="带前缀 ULID 主键，字典序即时间序",
        )

    @classmethod
    def new_id(cls) -> str:
        return generate_id(getattr(cls, "__id_prefix__", "obj_"))


class TimestampMixin:
    """创建/更新时间戳，全部 UTC tz-aware。"""

    @declared_attr
    @classmethod
    def created_at(cls) -> Mapped[datetime]:
        return mapped_column(UTCDateTime, default=utc_now, nullable=False, index=True)

    @declared_attr
    @classmethod
    def updated_at(cls) -> Mapped[datetime]:
        return mapped_column(UTCDateTime, default=utc_now, onupdate=utc_now, nullable=False)

    def touch(self) -> None:
        self.updated_at = utc_now()  # type: ignore[assignment]


class TenantMixin:
    """租户归属列。

    子类可用类变量微调：
    * ``__tenant_fk__``      是否建外键（audit_logs 允许系统级记录，故关掉）
    * ``__tenant_nullable__`` 是否允许为空
    """

    __tenant_fk__: ClassVar[bool] = True
    __tenant_nullable__: ClassVar[bool] = False

    @declared_attr
    @classmethod
    def tenant_id(cls) -> Mapped[str]:
        nullable = bool(getattr(cls, "__tenant_nullable__", False))
        if getattr(cls, "__tenant_fk__", True):
            return mapped_column(
                ULIDType,
                ForeignKey("tenants.id", ondelete="CASCADE"),
                nullable=nullable,
                index=True,
            )
        return mapped_column(ULIDType, nullable=nullable, index=True)


class SoftDeleteMixin:
    """软删除。仓储层默认过滤 ``deleted_at IS NOT NULL`` 的行。"""

    @declared_attr
    @classmethod
    def deleted_at(cls) -> Mapped[datetime | None]:
        return mapped_column(UTCDateTime, nullable=True, default=None, index=True)

    @property
    def is_deleted(self) -> bool:
        return getattr(self, "deleted_at", None) is not None

    def soft_delete(self) -> None:
        self.deleted_at = utc_now()  # type: ignore[assignment]

    def restore(self) -> None:
        self.deleted_at = None  # type: ignore[assignment]


class LabelMixin:
    """可选的人类可读标签，方便控制台展示。"""

    @declared_attr
    @classmethod
    def label(cls) -> Mapped[str | None]:
        return mapped_column(String(200), nullable=True, default=None)


def tenant_time_index(table_name: str, extra: str | None = None) -> Index:
    """构造 ``(tenant_id, created_at DESC)`` 复合索引。"""
    name = f"ix_{table_name}_tenant_created"
    if extra:
        name = f"{name}_{extra}"
    return Index(name, "tenant_id", "created_at")


def as_public_dict(obj: Any, *, drop: set[str] | None = None) -> dict[str, Any]:
    """把 ORM 对象转成 API 友好的 dict（datetime → ISO8601 由 FastAPI 处理）。"""
    hidden = {"password_hash", "key_hash", "credentials_enc"} | (drop or set())
    data: dict[str, Any] = {}
    for column in obj.__table__.columns:
        key = obj.__mapper__.get_property_by_column(column).key
        if key in hidden:
            continue
        data[key] = getattr(obj, key, None)
    return data


__all__ = [
    "LabelMixin",
    "SoftDeleteMixin",
    "TenantMixin",
    "TimestampMixin",
    "ULIDPkMixin",
    "as_public_dict",
    "tenant_time_index",
]

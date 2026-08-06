"""SQLAlchemy 2.0 声明式基类与约束命名约定。

命名约定必须稳定，否则 alembic autogenerate 会在不同环境下产生随机约束名，
导致迁移脚本无法在 SQLite 与 PostgreSQL 之间复用。
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

#: 约束命名模板（供 alembic autogenerate 生成稳定名称）
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata_obj = MetaData(naming_convention=NAMING_CONVENTION)


class Base(DeclarativeBase):
    """所有 ORM 模型的根基类。"""

    metadata = metadata_obj

    #: ULID 主键前缀，由 ULIDPkMixin 消费（见 omnistream/db/mixins.py）
    __id_prefix__: str = "obj_"

    def to_dict(self, *, exclude: set[str] | None = None) -> dict[str, Any]:
        """把 ORM 行转换为原生 dict（datetime 保持为 datetime 对象）。"""
        skip = exclude or set()
        out: dict[str, Any] = {}
        for column in self.__table__.columns:
            attr = self.__mapper__.get_property_by_column(column).key
            if attr in skip:
                continue
            out[attr] = getattr(self, attr, None)
        return out

    def apply(self, patch: dict[str, Any]) -> None:
        """按字段名批量赋值，忽略未知字段与主键。"""
        for key, value in patch.items():
            if key in {"id", "created_at", "tenant_id"}:
                continue
            if hasattr(self, key):
                setattr(self, key, value)

    def __repr__(self) -> str:
        pk = getattr(self, "id", None)
        return f"<{type(self).__name__} id={pk!r}>"


__all__ = ["Base", "NAMING_CONVENTION", "metadata_obj"]

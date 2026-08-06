"""``audit_logs`` 表：不可变追加 + 哈希链防篡改。

哈希链算法见 ``omnistream.governance.audit.logger``：
``hash = sha256(prev_hash || tenant_id || seq || action || resource || canonical(diff) || created_at)``
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import BigInteger, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from omnistream.db.base import Base
from omnistream.db.mixins import TenantMixin, TimestampMixin, ULIDPkMixin
from omnistream.db.types import JSONB

#: 系统级（无具体租户）审计事件使用的哨兵值
SYSTEM_TENANT = "ten_system"


class AuditLog(ULIDPkMixin, TenantMixin, TimestampMixin, Base):
    """审计日志。禁止 UPDATE / DELETE，只允许 INSERT 与读取。"""

    __tablename__ = "audit_logs"
    __id_prefix__ = "aud_"
    #: 允许 ten_system 这类不在 tenants 表里的哨兵值，故不建外键
    __tenant_fk__ = False

    #: 同租户内单调递增序号，哈希链的顺序依据
    seq: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    actor_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    #: user | api_key | system | worker
    actor_type: Mapped[str] = mapped_column(String(32), nullable=False, default="system")
    actor_label: Mapped[str | None] = mapped_column(String(320), nullable=True)
    action: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    resource_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    #: success | denied | error
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="success")
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(500), nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    diff: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    prev_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="0" * 64)
    hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")

    __table_args__ = (
        Index("ix_audit_logs_tenant_created", "tenant_id", "created_at"),
        Index("ix_audit_logs_tenant_seq", "tenant_id", "seq"),
        Index("ix_audit_logs_action_created", "action", "created_at"),
    )


__all__ = ["SYSTEM_TENANT", "AuditLog"]

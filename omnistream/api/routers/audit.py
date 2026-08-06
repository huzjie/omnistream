"""审计路由：分页查询审计日志（按动作 / 资源 / 时间过滤）。

读操作需要 ``audit:read``。审计日志只读不可改，这里纯粹是查询接口。
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select

from omnistream.api.deps import CurrentPrincipal, DbSession, Pagination, require_perm
from omnistream.db.models.audit_log import AuditLog
from omnistream.schemas.auth import Permission
from omnistream.schemas.common import Ok, Page

router = APIRouter()
ROUTER_ORDER = 80
ROUTER_TAGS = ["audit"]


def _to_dict(log: AuditLog) -> dict[str, Any]:
    return {
        "id": log.id,
        "seq": log.seq,
        "actor_id": log.actor_id,
        "actor_type": log.actor_type,
        "actor_label": log.actor_label,
        "action": log.action,
        "resource_type": log.resource_type,
        "resource_id": log.resource_id,
        "status": log.status,
        "ip": log.ip,
        "user_agent": log.user_agent,
        "request_id": log.request_id,
        "message": log.message,
        "diff": log.diff,
        "created_at": log.created_at.isoformat() if log.created_at else None,
    }


@router.get("", response_model=Ok[Page[dict]])
async def list_audit(
    principal: Annotated[Any, Depends(require_perm(Permission.AUDIT_READ))],
    session: DbSession,
    page: Pagination,
    action: str | None = Query(default=None),
    resource_id: str | None = Query(default=None),
    start: datetime | None = Query(default=None),
    end: datetime | None = Query(default=None),
) -> Ok[Page[dict]]:
    from sqlalchemy import func

    stmt = select(AuditLog).where(AuditLog.tenant_id == principal.tenant_id)
    if action:
        stmt = stmt.where(AuditLog.action == action)
    if resource_id:
        stmt = stmt.where(AuditLog.resource_id == resource_id)
    if start is not None:
        stmt = stmt.where(AuditLog.created_at >= start)
    if end is not None:
        stmt = stmt.where(AuditLog.created_at < end)

    total = await session.execute(
        func.count().select_from(AuditLog).where(AuditLog.tenant_id == principal.tenant_id)
    )
    rows = await session.execute(
        stmt.order_by(AuditLog.created_at.desc()).limit(page.limit).offset(page.offset)
    )
    items = [_to_dict(r) for r in rows.scalars().all()]
    return Ok.of(Page.of(items, total=int(total.scalar_one() or 0)))

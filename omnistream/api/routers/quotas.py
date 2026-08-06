"""配额路由：查询当前配额与用量，调整配额策略。

读取走 :func:`omnistream.platform.quota.resolve_policy`（三层策略叠加 + 周期翻页自动重置）；
调整写 ``quotas.policy`` JSONB。写操作需要 ``tenant:write``。
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from omnistream.api.deps import CurrentPrincipal, DbSession, require_perm
from omnistream.core.errors import NotFoundError, ValidationError
from omnistream.db.models.quota import Quota, DEFAULT_QUOTA_POLICY
from omnistream.platform.quota import ensure_quota_row, resolve_policy
from omnistream.schemas.auth import Permission
from omnistream.schemas.common import Ok

router = APIRouter()
ROUTER_ORDER = 50
ROUTER_TAGS = ["quotas"]


class QuotaPolicyPatch(BaseModel):
    policy: dict[str, float] = {}
    on_exceed: str | None = None


@router.get("", response_model=Ok[dict])
async def get_quota(
    principal: Annotated[Any, Depends(require_perm(Permission.TENANT_READ))],
    session: DbSession,
) -> Ok[dict]:
    """返回当前生效配额策略与周期用量。"""
    policy = await resolve_policy(session, principal.tenant_id)
    return Ok.of(policy.to_dict())


@router.patch("", response_model=Ok[dict])
async def update_quota(
    body: QuotaPolicyPatch,
    principal: Annotated[Any, Depends(require_perm(Permission.TENANT_WRITE))],
    session: DbSession,
) -> Ok[dict]:
    """调整配额策略（仅接受合法策略键）。"""
    for key in body.policy:
        if key not in DEFAULT_QUOTA_POLICY:
            raise ValidationError(f"未知配额键: {key}", detail={"key": key})
    row = await ensure_quota_row(session, principal.tenant_id)
    merged = dict(row.policy or {})
    for key, value in body.policy.items():
        merged[key] = float(value)
    row.policy = merged
    if body.on_exceed is not None:
        row.on_exceed = body.on_exceed
    await session.flush()
    policy = await resolve_policy(session, principal.tenant_id)
    return Ok.of(policy.to_dict())

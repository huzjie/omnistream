"""API Key 路由：签发 / 列举 / 详情 / 吊销 / 轮换。

明文密钥只在签发 / 轮换瞬间通过 ``ApiKeyIssued.plaintext`` 返回一次，库里只存前缀与哈希。
写操作需要 ``apikey:write``。
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from omnistream.api.deps import CurrentPrincipal, DbSession, Pagination, require_perm
from omnistream.config.settings import get_settings
from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import NotFoundError
from omnistream.db.models.api_key import ApiKey
from omnistream.platform.auth import ApiKeyService
from omnistream.schemas.auth import ApiKey as ApiKeySchema
from omnistream.schemas.auth import ApiKeyCreate, ApiKeyIssued, Permission
from omnistream.schemas.common import Ok, Page

router = APIRouter()
ROUTER_ORDER = 40
ROUTER_TAGS = ["api-keys"]


class RevokeBody(BaseModel):
    reason: str | None = None


class RotateBody(BaseModel):
    ttl_days: int | None = None


def _to_schema(key: ApiKey) -> ApiKeySchema:
    return ApiKeySchema.model_validate(key, from_attributes=True)


@router.get("", response_model=Ok[Page[ApiKeySchema]])
async def list_keys(
    principal: Annotated[Any, Depends(require_perm(Permission.APIKEY_READ))],
    session: DbSession,
    page: Pagination,
) -> Ok[Page[ApiKeySchema]]:
    from sqlalchemy import func

    service = ApiKeyService(session)
    total = await session.execute(
        func.count().select_from(ApiKey).where(ApiKey.tenant_id == principal.tenant_id)
    )
    keys = await service.list_for_tenant(principal.tenant_id)
    paged = keys[page.offset : page.offset + page.limit]
    return Ok.of(Page.of([_to_schema(k) for k in paged], total=int(total.scalar_one() or 0)))


@router.post("", response_model=Ok[ApiKeyIssued])
async def issue_key(
    body: ApiKeyCreate,
    principal: Annotated[Any, Depends(require_perm(Permission.APIKEY_WRITE))],
    session: DbSession,
) -> Ok[ApiKeyIssued]:
    security = get_settings().security
    service = ApiKeyService(session)
    issued = await service.issue(
        principal.tenant_id,
        name=body.name,
        role=body.role,
        scopes=list(body.scopes),
        ttl_days=body.ttl_days,
        created_by=principal.id,
        security=security,
    )
    return Ok.of(ApiKeyIssued(key=_to_schema(issued.key), plaintext=issued.plaintext))


@router.get("/{key_id}", response_model=Ok[ApiKeySchema])
async def get_key(
    key_id: str,
    principal: Annotated[Any, Depends(require_perm(Permission.APIKEY_READ))],
    session: DbSession,
) -> Ok[ApiKeySchema]:
    service = ApiKeyService(session)
    key = await service.get(key_id, principal.tenant_id)
    if key is None:
        raise NotFoundError("密钥不存在", code=ErrorCode.NOT_FOUND)
    return Ok.of(_to_schema(key))


@router.post("/{key_id}/revoke", response_model=Ok[ApiKeySchema])
async def revoke_key(
    key_id: str,
    body: RevokeBody | None = None,
    principal: Annotated[Any, Depends(require_perm(Permission.APIKEY_WRITE))] = None,
    session: DbSession | None = None,
) -> Ok[ApiKeySchema]:
    service = ApiKeyService(session)
    ok = await service.revoke(key_id, principal.tenant_id, reason=(body.reason if body else None))
    if not ok:
        raise NotFoundError("密钥不存在", code=ErrorCode.NOT_FOUND)
    key = await service.get(key_id, principal.tenant_id)
    assert key is not None
    return Ok.of(_to_schema(key))


@router.post("/{key_id}/rotate", response_model=Ok[ApiKeyIssued])
async def rotate_key(
    key_id: str,
    body: RotateBody | None = None,
    principal: Annotated[Any, Depends(require_perm(Permission.APIKEY_WRITE))] = None,
    session: DbSession | None = None,
) -> Ok[ApiKeyIssued]:
    security = get_settings().security
    service = ApiKeyService(session)
    issued = await service.rotate(
        key_id, principal.tenant_id, security=security, ttl_days=(body.ttl_days if body else None)
    )
    if issued is None:
        raise NotFoundError("密钥不存在", code=ErrorCode.NOT_FOUND)
    return Ok.of(ApiKeyIssued(key=_to_schema(issued.key), plaintext=issued.plaintext))

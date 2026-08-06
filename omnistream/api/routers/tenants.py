"""租户路由：CRUD 与设置更新。

仅 OWNER / ADMIN（``tenant:write``）可写。所有查询按调用主体租户隔离——OWNER 也只能管理
自己所在的租户（平台多租户硬隔离；跨租户运营操作由独立控制台完成）。
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select

from omnistream.api.deps import CurrentPrincipal, DbSession, require_perm
from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import ConflictError, NotFoundError, ValidationError
from omnistream.db.models.tenant import Tenant, TenantStatus
from omnistream.platform.tenancy import validate_slug
from omnistream.schemas.auth import Permission
from omnistream.schemas.common import Ok, Page
from omnistream.schemas.tenant import Tenant as TenantSchema, TenantCreate, TenantUpdate

router = APIRouter()
ROUTER_ORDER = 20
ROUTER_TAGS = ["tenants"]


class TenantSettingsPatch(BaseModel):
    """局部更新租户 settings（点分路径合并）。"""

    path: str
    value: Any


def _to_schema(tenant: Tenant) -> TenantSchema:
    return TenantSchema.model_validate(tenant, from_attributes=True)


@router.get("", response_model=Ok[TenantSchema])
async def get_current_tenant(
    principal: Annotated[Any, Depends(require_perm(Permission.TENANT_READ))],
    session: DbSession,
) -> Ok[TenantSchema]:
    """返回当前调用主体所属租户。"""
    tenant = (
        await session.execute(select(Tenant).where(Tenant.id == principal.tenant_id))
    ).scalar_one_or_none()
    if tenant is None:
        raise NotFoundError("租户不存在", code=ErrorCode.NOT_FOUND)
    return Ok.of(_to_schema(tenant))


@router.post("", response_model=Ok[TenantSchema])
async def create_tenant(
    body: TenantCreate,
    principal: Annotated[Any, Depends(require_perm(Permission.TENANT_WRITE))],
    session: DbSession,
) -> Ok[TenantSchema]:
    """创建新租户（仅具 ``tenant:write`` 的主体）。"""
    from omnistream.core.ids import new_id

    if not validate_slug(body.slug):
        raise ValidationError("slug 非法", code=ErrorCode.INVALID_REQUEST)
    existing = (
        await session.execute(select(Tenant).where(Tenant.slug == body.slug))
    ).scalar_one_or_none()
    if existing is not None:
        raise ConflictError("slug 已存在", code=ErrorCode.CONFLICT, detail={"slug": body.slug})

    tenant = Tenant(
        id=new_id("tenant"),
        slug=body.slug,
        name=body.name,
        status=TenantStatus.ACTIVE.value,
        settings=dict(body.settings.model_dump()),
    )
    session.add(tenant)
    await session.flush()
    return Ok.of(_to_schema(tenant))


@router.patch("/settings", response_model=Ok[TenantSchema])
async def patch_settings(
    body: TenantSettingsPatch,
    principal: Annotated[Any, Depends(require_perm(Permission.TENANT_WRITE))],
    session: DbSession,
) -> Ok[TenantSchema]:
    """按点分路径局部更新当前租户设置（如 ``billing.price_book``）。"""
    tenant = (
        await session.execute(select(Tenant).where(Tenant.id == principal.tenant_id))
    ).scalar_one_or_none()
    if tenant is None:
        raise NotFoundError("租户不存在", code=ErrorCode.NOT_FOUND)
    tenant.set_setting_path(body.path, body.value)
    await session.flush()
    return Ok.of(_to_schema(tenant))


@router.post("/suspend", response_model=Ok[TenantSchema])
async def suspend(
    principal: Annotated[Any, Depends(require_perm(Permission.TENANT_WRITE))],
    session: DbSession,
) -> Ok[TenantSchema]:
    """停用当前租户。"""
    tenant = (
        await session.execute(select(Tenant).where(Tenant.id == principal.tenant_id))
    ).scalar_one_or_none()
    if tenant is None:
        raise NotFoundError("租户不存在", code=ErrorCode.NOT_FOUND)
    tenant.status = TenantStatus.SUSPENDED.value
    await session.flush()
    return Ok.of(_to_schema(tenant))

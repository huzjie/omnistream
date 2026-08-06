"""成员管理路由：列表 / 创建 / 详情 / 改角色 / 停用 / 删除。

写操作需要 ``tenant:write``（或对应角色管理权限）。所有数据按调用主体租户隔离。
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from omnistream.api.deps import CurrentPrincipal, DbSession, Pagination, require_perm
from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import ConflictError, NotFoundError, ValidationError
from omnistream.core.ids import new_id
from omnistream.db.models.user import User, UserStatus
from omnistream.platform.auth.password import hash_password
from omnistream.schemas.auth import Permission, Role
from omnistream.schemas.common import Ok, Page

router = APIRouter()
ROUTER_ORDER = 30
ROUTER_TAGS = ["users"]


class UserCreate(BaseModel):
    email: str
    display_name: str = ""
    role: Role = Role.VIEWER
    password: str = Field(min_length=8)
    is_service_account: bool = False


class UserUpdate(BaseModel):
    display_name: str | None = None
    role: Role | None = None
    status: str | None = None
    password: str | None = Field(default=None, min_length=8)


def _user_to_dict(user: User) -> dict[str, Any]:
    return {
        "id": user.id,
        "email": user.email,
        "display_name": user.display_name,
        "role": user.role,
        "status": user.status,
        "is_service_account": user.is_service_account,
        "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


@router.get("", response_model=Ok[Page[dict]])
async def list_users(
    principal: Annotated[Any, Depends(require_perm(Permission.TENANT_READ))],
    session: DbSession,
    page: Pagination,
) -> Ok[Page[dict]]:
    from sqlalchemy import func

    total = await session.execute(
        func.count().select_from(User).where(User.tenant_id == principal.tenant_id)
    )
    rows = await session.execute(
        select(User)
        .where(User.tenant_id == principal.tenant_id)
        .order_by(User.created_at.desc())
        .limit(page.limit)
        .offset(page.offset)
    )
    items = [_user_to_dict(u) for u in rows.scalars().all()]
    return Ok.of(Page.of(items, total=int(total.scalar_one() or 0)))


@router.post("", response_model=Ok[dict])
async def create_user(
    body: UserCreate,
    principal: Annotated[Any, Depends(require_perm(Permission.TENANT_WRITE))],
    session: DbSession,
) -> Ok[dict]:
    existing = await session.execute(
        select(User).where(User.tenant_id == principal.tenant_id, User.email == body.email)
    )
    if existing.scalar_one_or_none() is not None:
        raise ConflictError("邮箱已存在", code=ErrorCode.CONFLICT, detail={"email": body.email})
    user = User(
        id=new_id("user"),
        tenant_id=principal.tenant_id,
        email=body.email,
        display_name=body.display_name or body.email.split("@")[0],
        role=str(body.role),
        password_hash=hash_password(body.password),
        is_service_account=body.is_service_account,
        status=UserStatus.ACTIVE.value,
    )
    session.add(user)
    await session.flush()
    return Ok.of(_user_to_dict(user))


@router.get("/{user_id}", response_model=Ok[dict])
async def get_user(
    user_id: str,
    principal: Annotated[Any, Depends(require_perm(Permission.TENANT_READ))],
    session: DbSession,
) -> Ok[dict]:
    user = await session.execute(
        select(User).where(User.id == user_id, User.tenant_id == principal.tenant_id)
    )
    found = user.scalar_one_or_none()
    if found is None:
        raise NotFoundError("用户不存在", code=ErrorCode.NOT_FOUND)
    return Ok.of(_user_to_dict(found))


@router.patch("/{user_id}", response_model=Ok[dict])
async def update_user(
    user_id: str,
    body: UserUpdate,
    principal: Annotated[Any, Depends(require_perm(Permission.TENANT_WRITE))],
    session: DbSession,
) -> Ok[dict]:
    user = await session.execute(
        select(User).where(User.id == user_id, User.tenant_id == principal.tenant_id)
    )
    found = user.scalar_one_or_none()
    if found is None:
        raise NotFoundError("用户不存在", code=ErrorCode.NOT_FOUND)
    if body.display_name is not None:
        found.display_name = body.display_name
    if body.role is not None:
        found.role = str(body.role)
    if body.status is not None:
        found.status = body.status
    if body.password is not None:
        found.password_hash = hash_password(body.password)
    await session.flush()
    return Ok.of(_user_to_dict(found))


@router.delete("/{user_id}", response_model=Ok[None])
async def delete_user(
    user_id: str,
    principal: Annotated[Any, Depends(require_perm(Permission.TENANT_WRITE))],
    session: DbSession,
) -> Ok[None]:
    user = await session.execute(
        select(User).where(User.id == user_id, User.tenant_id == principal.tenant_id)
    )
    found = user.scalar_one_or_none()
    if found is None:
        raise NotFoundError("用户不存在", code=ErrorCode.NOT_FOUND)
    await session.delete(found)
    await session.flush()
    return Ok.of(None)

"""统一鉴权主体工厂。

不同凭据来源（JWT claims / API Key ORM / User ORM / 系统任务）都被规约成同一个
:class:`omnistream.schemas.auth.Principal`，下游（RBAC / 配额 / 审计 / 计费）只认这一个类型。

设计要点：

* 本模块**不重新定义** Principal，直接复用 ``schemas.auth.Principal``（单一事实来源）；
* 工厂函数接受 ORM 对象或 claims，统一做「脏数据兜底」（非法 role / scope 降级而非抛错）；
* 系统主体用于后台任务（Celery / 巡检脚本），拥有全部权限。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from omnistream.schemas.auth import JWTClaims, Permission, Principal, Role

if TYPE_CHECKING:
    from omnistream.db.models.api_key import ApiKey
    from omnistream.db.models.user import User

#: 平台级操作使用的哨兵租户（审计 / worker 可写入此租户而不依赖 tenants 行）
SYSTEM_TENANT_ID = "ten_system"


def _safe_role(value: object) -> Role:
    """把任意值解析为合法 :class:`Role`；非法值降级为 ``SERVICE``。"""
    try:
        return Role(str(value))
    except (ValueError, TypeError):
        return Role.SERVICE


def _safe_scopes(values: object) -> list[Permission]:
    """把字符串 / Permission 列表解析为合法 :class:`Permission` 列表，跳过非法项。"""
    out: list[Permission] = []
    if not isinstance(values, (list, tuple, set, frozenset)):
        return out
    for raw in values:
        try:
            out.append(Permission(str(raw)))
        except (ValueError, TypeError):
            continue
    return out


def principal_from_claims(claims: JWTClaims) -> Principal:
    """由已验签的 JWT claims 构建主体。供 ``api/deps.py`` 调用。"""
    return Principal(
        id=claims.sub,
        tenant_id=claims.tid,
        kind="user",
        role=claims.role,
        scopes=list(claims.scopes),
        session_id=claims.sid,
    )


def principal_from_api_key(key: "ApiKey", *, tenant_id: str | None = None) -> Principal:
    """由 ``ApiKey`` ORM 对象构建主体。"""
    return Principal(
        id=key.id,
        tenant_id=tenant_id or key.tenant_id,
        kind="apikey",
        role=_safe_role(key.role),
        scopes=_safe_scopes(key.scopes),
        api_key_id=key.id,
    )


def principal_from_user(user: "User", *, tenant_id: str | None = None) -> Principal:
    """由 ``User`` ORM 对象构建主体。"""
    return Principal(
        id=user.id,
        tenant_id=tenant_id or user.tenant_id,
        kind="user",
        role=_safe_role(user.role),
        email=user.email,
    )


def system_principal(tenant_id: str = SYSTEM_TENANT_ID) -> Principal:
    """后台任务（Celery / 定时器）使用的全权限系统主体。"""
    return Principal.internal(tenant_id)


__all__ = [
    "SYSTEM_TENANT_ID",
    "principal_from_api_key",
    "principal_from_claims",
    "principal_from_user",
    "system_principal",
]

"""权限执行（enforcement）。

围绕 :class:`omnistream.schemas.auth.Principal` 提供：

* 命令式检查 :func:`enforce` / :func:`enforce_any` / :func:`enforce_all`
  —— 无权限时抛 :class:`~omnistream.core.errors.PermissionError_`；
* 租户 / 资源归属检查 :func:`enforce_tenant` / :func:`enforce_resource_owner`；
* FastAPI 依赖工厂 :func:`require` / :func:`require_any` / :func:`require_role`。

``require_*`` 返回的是普通 async callable，调用方用 ``Depends(require(x))`` 包裹即可，
本模块不再直接依赖 FastAPI，便于在 worker / CLI 等非 Web 场景复用。
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import AuthError, PermissionError_

if TYPE_CHECKING:
    from omnistream.schemas.auth import Permission, Principal, Role

__all__ = [
    "enforce",
    "enforce_all",
    "enforce_any",
    "enforce_resource_owner",
    "enforce_tenant",
    "require",
    "require_any",
    "require_role",
]


def enforce(principal: "Principal", perm: "Permission | str") -> None:
    """无权限抛 ``FORBIDDEN``。"""
    from omnistream.schemas.auth import Permission as _P

    need = _P(perm) if not isinstance(perm, _P) else perm
    if not principal.has(need):
        raise PermissionError_(
            "权限不足",
            code=ErrorCode.FORBIDDEN,
            detail={"required": str(need), "granted": sorted(str(p) for p in principal.permissions)},
        )


def enforce_any(principal: "Principal", *perms: "Permission | str") -> None:
    """具备任意一个即放行；全无则抛 ``FORBIDDEN``。"""
    from omnistream.schemas.auth import Permission as _P

    needs = [(_P(p) if not isinstance(p, _P) else p) for p in perms]
    if not needs:
        return
    if not any(principal.has(p) for p in needs):
        raise PermissionError_(
            "权限不足",
            code=ErrorCode.FORBIDDEN,
            detail={"required_any": [str(p) for p in needs]},
        )


def enforce_all(principal: "Principal", *perms: "Permission | str") -> None:
    """必须全部具备；缺一则抛 ``FORBIDDEN``。"""
    from omnistream.schemas.auth import Permission as _P

    needs = [(_P(p) if not isinstance(p, _P) else p) for p in perms]
    missing = [str(p) for p in needs if not principal.has(p)]
    if missing:
        raise PermissionError_(
            "权限不足",
            code=ErrorCode.FORBIDDEN,
            detail={"missing": missing, "granted": sorted(str(p) for p in principal.permissions)},
        )


def enforce_tenant(principal: "Principal", tenant_id: str) -> None:
    """跨租户操作拦截。"""
    if not tenant_id or principal.tenant_id != tenant_id:
        raise PermissionError_(
            "无权访问该租户资源",
            code=ErrorCode.FORBIDDEN,
            detail={"expected_tenant": tenant_id, "actual_tenant": principal.tenant_id},
        )


def enforce_resource_owner(
    principal: "Principal",
    *,
    owner_id: str,
    role_floor: "Role | None" = None,
) -> None:
    """资源级归属检查：是本人，或角色不低于 ``role_floor`` 的管理员，放行。

    ``role_floor`` 不传时只要非本人就拦截（严格归属）。
    """
    from omnistream.platform.rbac.roles import role_at_least

    if owner_id and principal.id == owner_id:
        return
    if role_floor is not None and role_at_least(principal.role, role_floor):
        return
    raise PermissionError_(
        "只能操作本人资源",
        code=ErrorCode.FORBIDDEN,
        detail={"owner_id": owner_id},
    )


def require(perm: "Permission | str") -> Callable[["Principal"], "Principal"]:
    """依赖工厂：调用方必须具备 ``perm``。"""
    from omnistream.schemas.auth import Permission as _P

    need = _P(perm) if not isinstance(perm, _P) else perm

    async def _dependency(principal: "Principal") -> "Principal":
        enforce(principal, need)
        return principal

    _dependency.__name__ = f"require_{need.name.lower()}"
    return _dependency


def require_any(*perms: "Permission | str") -> Callable[["Principal"], "Principal"]:
    """依赖工厂：具备任意一个即可。"""
    from omnistream.schemas.auth import Permission as _P

    needs = [(_P(p) if not isinstance(p, _P) else p) for p in perms]

    async def _dependency(principal: "Principal") -> "Principal":
        enforce_any(principal, *needs)
        return principal

    _dependency.__name__ = "require_any_" + "_".join(p.name.lower() for p in needs)
    return _dependency


def require_role(minimum: "Role") -> Callable[["Principal"], "Principal"]:
    """依赖工厂：角色等级不低于 ``minimum``。"""
    from omnistream.platform.rbac.roles import role_at_least

    async def _dependency(principal: "Principal") -> "Principal":
        if not role_at_least(principal.role, minimum):
            raise PermissionError_(
                "角色权限不足",
                code=ErrorCode.FORBIDDEN,
                detail={"required_role": str(minimum), "actual_role": str(principal.role)},
            )
        return principal

    _dependency.__name__ = f"require_role_{minimum.name.lower()}"
    return _dependency


def _unused_auth_error() -> None:  # pragma: no cover - 仅用于显式保留导入
    raise AuthError("unused")

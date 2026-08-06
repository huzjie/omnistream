"""角色定义与权限解析。

角色权限矩阵以 :data:`omnistream.schemas.auth.ROLE_PERMISSIONS` 为准（单一事实来源）。
本模块在其基础上补充：

* 角色等级（``role_rank`` / ``role_at_least``）—— 用于「角色不低于 X」式校验；
* ``permissions_for`` —— 把 API Key 的显式 ``scopes`` 与角色权限取**交集**
  （Key 只能收窄角色权限，不能放大）；

所有对外暴露的权限集合统一为 :class:`~omnistream.schemas.auth.Permission` 的 frozenset。
"""
from __future__ import annotations

from typing import Iterable, Mapping

from omnistream.schemas.auth import Permission, Role, role_permissions

from omnistream.platform.rbac.permissions import expand_wildcards, parse

__all__ = [
    "DEFAULT_ROLE",
    "ROLE_DESCRIPTIONS",
    "ROLE_ORDER",
    "Role",
    "is_builtin_role",
    "permissions_for",
    "role_at_least",
    "role_rank",
]

#: 角色等级（数值越大权限越高）。仅用于比较，不决定具体权限点。
ROLE_ORDER: dict[Role, int] = {
    Role.VIEWER: 10,
    Role.SERVICE: 20,
    Role.ANALYST: 30,
    Role.DEVELOPER: 50,
    Role.ADMIN: 80,
    Role.OWNER: 100,
}

#: 新成员默认角色
DEFAULT_ROLE: Role = Role.VIEWER

#: 角色中文说明（控制台展示）
ROLE_DESCRIPTIONS: dict[Role, str] = {
    Role.OWNER: "租户所有者，拥有全部权限",
    Role.ADMIN: "管理员，可管理成员、密钥与配置（不含销毁租户等极少数操作）",
    Role.DEVELOPER: "开发者，可创建会话、调用工具、查看录制",
    Role.ANALYST: "分析师，可查看会话、质检与账单，可提交质检",
    Role.VIEWER: "只读访客，仅查看会话与指标",
    Role.SERVICE: "机器身份（API Key 签发），权限由签发时授权的 scopes 决定",
}


def is_builtin_role(role: "Role | str") -> bool:
    """判断是否为系统内置角色。"""
    try:
        return Role(role) in ROLE_ORDER
    except (ValueError, TypeError):
        return False


def role_rank(role: "Role | str") -> int:
    """返回角色等级数值；未知角色为 0。"""
    try:
        return ROLE_ORDER.get(Role(role), 0)
    except (ValueError, TypeError):
        return 0


def role_at_least(role: "Role | str", minimum: "Role | str") -> bool:
    """判断 ``role`` 等级是否不低于 ``minimum``。"""
    return role_rank(role) >= role_rank(minimum)


def permissions_for(
    role: "Role | str",
    *,
    tenant_settings: Mapping[str, object] | None = None,
    extra_scopes: Iterable["Permission | str"] | None = None,
    expand: bool = False,
) -> frozenset[Permission]:
    """解析某角色的最终生效权限集合。

    * 基础权限来自 :func:`role_permissions(role)`；
    * 若传入 ``extra_scopes``（API Key 的显式授权），则与基础权限取**交集**
      —— Key 只能收窄，不能放大；
    * ``expand=True`` 时把 ``admin:*`` 展开成全集；
    * ``tenant_settings`` 预留给未来「租户级权限模板」，当前不影响结果。
    """
    try:
        resolved_role = Role(role)
    except (ValueError, TypeError):
        resolved_role = Role.SERVICE

    base = role_permissions(resolved_role)

    if extra_scopes is not None:
        scopes: set[Permission] = set()
        for raw in extra_scopes:
            parsed = parse(str(raw)) if not isinstance(raw, Permission) else raw
            if parsed is not None:
                scopes.add(parsed)
        # 交集：显式授权只能收窄角色权限
        effective = base & scopes
    else:
        effective = set(base)

    if expand:
        return expand_wildcards(effective)
    return frozenset(effective)

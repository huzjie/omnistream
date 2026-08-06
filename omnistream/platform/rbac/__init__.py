"""RBAC 子包：权限表、角色映射与判定器。"""
from __future__ import annotations

from omnistream.platform.rbac.enforcer import (
    enforce,
    enforce_all,
    enforce_any,
    enforce_resource_owner,
    enforce_tenant,
    require,
    require_any,
    require_role,
)
from omnistream.platform.rbac.permissions import (
    ALL_PERMISSIONS,
    PERMISSION_GROUPS,
    PERMISSION_LABELS,
    Permission,
    WILDCARD,
    expand_wildcards,
    match_permission,
    normalize,
    parse,
    resource_of,
)
from omnistream.platform.rbac.roles import (
    DEFAULT_ROLE,
    ROLE_DESCRIPTIONS,
    ROLE_ORDER,
    Role,
    is_builtin_role,
    permissions_for,
    role_at_least,
    role_rank,
)

__all__ = [
    "ALL_PERMISSIONS",
    "DEFAULT_ROLE",
    "PERMISSION_GROUPS",
    "PERMISSION_LABELS",
    "ROLE_DESCRIPTIONS",
    "ROLE_ORDER",
    "Permission",
    "Role",
    "WILDCARD",
    "enforce",
    "enforce_all",
    "enforce_any",
    "enforce_resource_owner",
    "enforce_tenant",
    "expand_wildcards",
    "is_builtin_role",
    "match_permission",
    "normalize",
    "parse",
    "permissions_for",
    "require",
    "require_any",
    "require_role",
    "resource_of",
    "role_at_least",
    "role_rank",
]

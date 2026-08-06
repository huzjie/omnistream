"""权限点定义与匹配工具。

本模块是平台层对 :mod:`omnistream.schemas.auth.Permission` 的再导出与配套工具。
所有权限点以 ``<resource>:<action>`` 形式表达，``admin:*`` 为超级通配符。

下游（RBAC / 配额 / 审计 / 计费 / API 路由）只消费 :class:`Permission`，
不要自行定义新的权限字符串，避免与 ``schemas.auth`` 漂移。
"""
from __future__ import annotations

from typing import Iterable

from omnistream.schemas.auth import Permission

__all__ = [
    "ALL_PERMISSIONS",
    "PERMISSION_GROUPS",
    "PERMISSION_LABELS",
    "Permission",
    "WILDCARD",
    "expand_wildcards",
    "match_permission",
    "normalize",
    "parse",
    "resource_of",
]

#: 超级通配符（拥有它即拥有全部权限）
WILDCARD: Permission = Permission.ADMIN_ALL

#: 全部权限点集合
ALL_PERMISSIONS: frozenset[Permission] = frozenset(Permission)

#: 人类可读标签（控制台展示 / 审计说明）
PERMISSION_LABELS: dict[Permission, str] = {
    Permission.SESSION_READ: "查看会话",
    Permission.SESSION_WRITE: "创建 / 结束会话",
    Permission.SESSION_CONNECT: "建立会话连接",
    Permission.MEMORY_READ: "查看记忆",
    Permission.MEMORY_WRITE: "写入记忆",
    Permission.RECORDING_READ: "查看录制",
    Permission.RECORDING_DELETE: "删除录制",
    Permission.PROVIDER_READ: "查看 Provider",
    Permission.PROVIDER_WRITE: "配置 Provider",
    Permission.TENANT_READ: "查看租户",
    Permission.TENANT_WRITE: "编辑租户",
    Permission.APIKEY_READ: "查看密钥",
    Permission.APIKEY_WRITE: "签发 / 吊销密钥",
    Permission.BILLING_READ: "查看账单",
    Permission.AUDIT_READ: "查看审计",
    Permission.QUALITY_READ: "查看质检",
    Permission.QUALITY_WRITE: "提交质检",
    Permission.METRICS_READ: "查看指标",
    Permission.ADMIN_ALL: "全部权限",
}

#: 按资源分组（控制台分组展示与前端菜单）
PERMISSION_GROUPS: dict[str, list[Permission]] = {
    "session": [Permission.SESSION_READ, Permission.SESSION_WRITE, Permission.SESSION_CONNECT],
    "memory": [Permission.MEMORY_READ, Permission.MEMORY_WRITE],
    "recording": [Permission.RECORDING_READ, Permission.RECORDING_DELETE],
    "provider": [Permission.PROVIDER_READ, Permission.PROVIDER_WRITE],
    "tenant": [Permission.TENANT_READ, Permission.TENANT_WRITE],
    "apikey": [Permission.APIKEY_READ, Permission.APIKEY_WRITE],
    "billing": [Permission.BILLING_READ],
    "audit": [Permission.AUDIT_READ],
    "quality": [Permission.QUALITY_READ, Permission.QUALITY_WRITE],
    "metrics": [Permission.METRICS_READ],
    "admin": [Permission.ADMIN_ALL],
}


def normalize(value: "Permission | str") -> Permission:
    """归一化任意输入为 :class:`Permission`。

    接受 :class:`Permission` 或字符串（大小写不敏感）。非法值抛 ``ValueError``。
    """
    if isinstance(value, Permission):
        return value
    try:
        return Permission(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError(f"未知权限点: {value!r}") from exc


def parse(value: str) -> Permission | None:
    """宽松解析：非法字符串返回 ``None``（用于读取第三方 / 历史数据）。"""
    try:
        return Permission(value.strip().lower())
    except ValueError:
        return None


def resource_of(perm: Permission) -> str:
    """返回权限点所属的资源域，如 ``session:read`` -> ``session``。"""
    return str(perm).split(":", 1)[0]


def match_permission(granted: Iterable[Permission], required: Permission) -> bool:
    """判断 ``granted`` 集合是否覆盖 ``required``。

    ``admin:*`` 视为覆盖一切。
    """
    granted_set = set(granted)
    if WILDCARD in granted_set:
        return True
    return required in granted_set


def expand_wildcards(granted: Iterable[Permission]) -> frozenset[Permission]:
    """把 ``admin:*`` 展开成全集，其余原样返回。"""
    granted_set = set(granted)
    if WILDCARD in granted_set:
        return ALL_PERMISSIONS
    return frozenset(granted_set)

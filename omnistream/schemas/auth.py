"""认证与授权领域模型：角色、权限、API Key、JWT 声明、调用主体。"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from omnistream.core.clock import utcnow
from omnistream.schemas.base import OmniModel


class Role(StrEnum):
    """租户内角色。权限自上而下包含。"""

    OWNER = "owner"
    ADMIN = "admin"
    DEVELOPER = "developer"
    ANALYST = "analyst"
    VIEWER = "viewer"
    SERVICE = "service"          # 机器身份（API Key 签发）


class Permission(StrEnum):
    """细粒度权限点。API Key scopes 与角色映射都用它。"""

    SESSION_READ = "session:read"
    SESSION_WRITE = "session:write"
    SESSION_CONNECT = "session:connect"
    MEMORY_READ = "memory:read"
    MEMORY_WRITE = "memory:write"
    RECORDING_READ = "recording:read"
    RECORDING_DELETE = "recording:delete"
    PROVIDER_READ = "provider:read"
    PROVIDER_WRITE = "provider:write"
    TENANT_READ = "tenant:read"
    TENANT_WRITE = "tenant:write"
    APIKEY_READ = "apikey:read"
    APIKEY_WRITE = "apikey:write"
    BILLING_READ = "billing:read"
    AUDIT_READ = "audit:read"
    QUALITY_READ = "quality:read"
    QUALITY_WRITE = "quality:write"
    METRICS_READ = "metrics:read"
    ADMIN_ALL = "admin:*"


#: 角色 → 权限集合。查询时用 ``role_permissions()``。
ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.OWNER: frozenset(Permission),
    Role.ADMIN: frozenset(p for p in Permission if p is not Permission.ADMIN_ALL),
    Role.DEVELOPER: frozenset({
        Permission.SESSION_READ, Permission.SESSION_WRITE, Permission.SESSION_CONNECT,
        Permission.MEMORY_READ, Permission.MEMORY_WRITE,
        Permission.RECORDING_READ, Permission.PROVIDER_READ,
        Permission.APIKEY_READ, Permission.METRICS_READ, Permission.QUALITY_READ,
    }),
    Role.ANALYST: frozenset({
        Permission.SESSION_READ, Permission.RECORDING_READ, Permission.METRICS_READ,
        Permission.QUALITY_READ, Permission.QUALITY_WRITE, Permission.BILLING_READ,
        Permission.AUDIT_READ, Permission.MEMORY_READ,
    }),
    Role.VIEWER: frozenset({
        Permission.SESSION_READ, Permission.METRICS_READ, Permission.PROVIDER_READ,
    }),
    Role.SERVICE: frozenset({
        Permission.SESSION_READ, Permission.SESSION_WRITE, Permission.SESSION_CONNECT,
        Permission.MEMORY_READ, Permission.MEMORY_WRITE, Permission.PROVIDER_READ,
    }),
}


def role_permissions(role: Role) -> frozenset[Permission]:
    """返回角色对应的权限集合，未知角色返回空集。"""
    return ROLE_PERMISSIONS.get(role, frozenset())


class ApiKey(OmniModel):
    """API Key 元数据。明文仅在签发瞬间存在于 ``ApiKeyIssued.plaintext``。"""

    id: str
    tenant_id: str
    name: str = ""
    prefix: str = Field(description="明文前 12 位，用于快速定位与展示")
    scopes: list[Permission] = Field(default_factory=list)
    role: Role = Role.SERVICE
    enabled: bool = True
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = None
    last_used_at: datetime | None = None
    created_by: str | None = None

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and utcnow() >= self.expires_at

    @property
    def usable(self) -> bool:
        return self.enabled and not self.expired

    @property
    def masked(self) -> str:
        return f"{self.prefix}{'*' * 20}"


class ApiKeyCreate(OmniModel):
    name: str = ""
    role: Role = Role.SERVICE
    scopes: list[Permission] = Field(default_factory=list)
    ttl_days: int | None = Field(default=None, ge=1, le=3650)


class ApiKeyIssued(OmniModel):
    """签发响应。``plaintext`` 只出现这一次，之后不可再取回。"""

    key: ApiKey
    plaintext: str = Field(repr=False)


class JWTClaims(OmniModel):
    """控制台登录态 JWT 的载荷。"""

    sub: str = Field(description="用户 ID")
    tid: str = Field(description="租户 ID")
    role: Role = Role.VIEWER
    scopes: list[Permission] = Field(default_factory=list)
    iss: str = "omnistream"
    aud: str = "omnistream-console"
    iat: int = 0
    exp: int = 0
    jti: str = ""
    #: 会话专用 token 时携带，限制该 token 只能连这一个会话
    sid: str | None = None

    @property
    def expired(self) -> bool:
        return self.exp > 0 and utcnow().timestamp() >= self.exp


class Principal(OmniModel):
    """一次请求的调用主体。所有依赖注入与鉴权都围绕它。"""

    id: str
    tenant_id: str
    kind: str = Field(default="user", description="user|apikey|internal")
    role: Role = Role.VIEWER
    scopes: list[Permission] = Field(default_factory=list)
    email: str | None = None
    session_id: str | None = None
    api_key_id: str | None = None

    @property
    def permissions(self) -> frozenset[Permission]:
        """角色权限与显式 scopes 的并集。"""
        return role_permissions(self.role) | frozenset(self.scopes)

    def has(self, perm: Permission) -> bool:
        perms = self.permissions
        return Permission.ADMIN_ALL in perms or perm in perms

    def has_all(self, *perms: Permission) -> bool:
        return all(self.has(p) for p in perms)

    @classmethod
    def internal(cls, tenant_id: str) -> "Principal":
        """内部任务（worker/定时器）使用的全权限主体。"""
        return cls(id="internal", tenant_id=tenant_id, kind="internal",
                   role=Role.OWNER, scopes=[Permission.ADMIN_ALL])


class LoginRequest(OmniModel):
    email: str
    password: str = Field(repr=False)
    tenant_slug: str | None = None


class TokenPair(OmniModel):
    access_token: str = Field(repr=False)
    refresh_token: str = Field(repr=False)
    token_type: str = "Bearer"
    expires_in: int = 3600
    principal: Principal | None = None

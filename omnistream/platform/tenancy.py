"""租户解析、租户上下文与 Redis 命名空间隔离。

解析优先级（越靠前优先级越高）：

1. 已认证主体携带的 ``tenant_id``（API Key / JWT）—— 最可信；
2. 显式请求头 ``X-Tenant-Id`` 或 ``X-Tenant-Slug``（需与主体一致，或主体具备平台超管权限）；
3. 子域名 ``<slug>.example.com``；
4. 单租户部署时的默认租户 ``ten_default``。

Redis 命名空间统一走 :func:`namespaced`，形如
``oms:{env}:{tenant_id}:{scope}:{key}``，从 key 结构上杜绝跨租户串数据。
Redis 缺失时全部退化为进程内实现，服务照常启动。
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import AuthError, NotFoundError, ValidationError
from omnistream.db.engine import setting
from omnistream.db.models.tenant import Tenant, TenantStatus

if TYPE_CHECKING:  # 仅类型标注，运行期不导入，避免 auth <-> tenancy 循环
    from omnistream.platform.auth.principal import Principal

__all__ = [
    "DEFAULT_TENANT_ID",
    "DEFAULT_TENANT_SLUG",
    "MemoryKV",
    "TenantContext",
    "TenantResolver",
    "close_redis",
    "current_tenant_id",
    "ensure_default_tenant",
    "get_redis",
    "memory_kv",
    "namespaced",
    "platform_secret",
    "redis_available",
    "slug_from_host",
]

DEFAULT_TENANT_ID = "ten_default"
DEFAULT_TENANT_SLUG = "default"

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")
_RESERVED_SLUGS = frozenset({"www", "api", "app", "admin", "console", "static", "cdn", "localhost"})

_redis_client: Any | None = None
_redis_lock = asyncio.Lock()
_redis_disabled = False


# --------------------------------------------------------------------------- #
# 平台密钥
# --------------------------------------------------------------------------- #

def platform_secret() -> str:
    """平台对称密钥。

    读取顺序：``security.secret_key`` > ``security.jwt_secret`` > 开发兜底值。
    生产环境必须通过 ``OMS_SECURITY__SECRET_KEY`` 注入，否则日志会给出告警。
    """
    value = str(setting("security.secret_key", "") or setting("security.jwt_secret", "") or "")
    if value:
        return value
    return "omnistream-dev-insecure-secret-change-me"


def _env_name() -> str:
    return str(setting("env", "dev") or "dev")


# --------------------------------------------------------------------------- #
# Redis 命名空间与降级 KV
# --------------------------------------------------------------------------- #

def namespaced(tenant_id: str, scope: str, key: str = "") -> str:
    """构造租户隔离的缓存键。"""
    tid = tenant_id or "ten_anonymous"
    parts = ["oms", _env_name(), tid, scope]
    if key:
        parts.append(key)
    return ":".join(parts)


class MemoryKV:
    """进程内 KV，带 TTL 与原子自增，用作 Redis 的零依赖替身。

    单进程部署、Mock 模式与本地开发完全够用；多副本部署请配置 Redis。
    """

    __slots__ = ("_data", "_lock")

    def __init__(self) -> None:
        self._data: dict[str, tuple[float, Any]] = {}
        self._lock = asyncio.Lock()

    def _purge(self, now: float) -> None:
        expired = [k for k, (exp, _) in self._data.items() if exp and exp <= now]
        for key in expired:
            self._data.pop(key, None)

    async def get(self, key: str) -> Any:
        now = time.time()
        async with self._lock:
            self._purge(now)
            entry = self._data.get(key)
            return entry[1] if entry else None

    async def set(self, key: str, value: Any, ttl: float = 0.0) -> None:
        async with self._lock:
            self._data[key] = (time.time() + ttl if ttl > 0 else 0.0, value)

    async def delete(self, key: str) -> bool:
        async with self._lock:
            return self._data.pop(key, None) is not None

    async def incr(self, key: str, amount: float = 1.0, ttl: float = 0.0) -> float:
        async with self._lock:
            now = time.time()
            self._purge(now)
            exp, current = self._data.get(key, (0.0, 0.0))
            total = float(current or 0.0) + amount
            new_exp = exp if exp else (now + ttl if ttl > 0 else 0.0)
            self._data[key] = (new_exp, total)
            return total

    async def exists(self, key: str) -> bool:
        return await self.get(key) is not None

    async def keys(self, prefix: str) -> list[str]:
        now = time.time()
        async with self._lock:
            self._purge(now)
            return [k for k in self._data if k.startswith(prefix)]

    async def clear(self, prefix: str = "") -> int:
        async with self._lock:
            targets = [k for k in self._data if not prefix or k.startswith(prefix)]
            for key in targets:
                self._data.pop(key, None)
            return len(targets)


#: 全局降级 KV 实例
memory_kv = MemoryKV()


def redis_url() -> str:
    """Redis 连接串，未配置返回空字符串。"""
    return str(setting("storage.redis_url", "") or "")


async def get_redis() -> Any | None:
    """返回 ``redis.asyncio.Redis`` 客户端；未配置或不可用时返回 ``None``。

    只探测一次，失败后永久降级到 :data:`memory_kv`，避免每次请求都超时等待。
    """
    global _redis_client, _redis_disabled

    if _redis_disabled:
        return None
    if _redis_client is not None:
        return _redis_client

    url = redis_url()
    if not url:
        _redis_disabled = True
        return None

    async with _redis_lock:
        if _redis_client is not None:
            return _redis_client
        if _redis_disabled:
            return None
        try:
            from redis.asyncio import Redis  # type: ignore[import-not-found]

            client = Redis.from_url(url, encoding="utf-8", decode_responses=True)
            await client.ping()
        except Exception:  # noqa: BLE001 - 任何异常都降级，不能拖垮启动
            _redis_disabled = True
            return None
        _redis_client = client
        return _redis_client


async def redis_available() -> bool:
    """Redis 是否可用。"""
    return await get_redis() is not None


async def close_redis() -> None:
    """关闭 Redis 连接（lifespan 收尾用）。"""
    global _redis_client, _redis_disabled
    client = _redis_client
    _redis_client = None
    _redis_disabled = False
    if client is not None:
        try:
            await client.aclose()
        except AttributeError:  # 老版本 redis-py
            try:
                await client.close()
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# 租户上下文
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class TenantContext:
    """请求级租户上下文。"""

    tenant_id: str
    slug: str = ""
    name: str = ""
    plan: str = "free"
    status: str = str(TenantStatus.ACTIVE)
    settings: Mapping[str, Any] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.status == str(TenantStatus.ACTIVE)

    def key(self, scope: str, key: str = "") -> str:
        """该租户下的命名空间缓存键。"""
        return namespaced(self.tenant_id, scope, key)

    def option(self, path: str, default: Any = None) -> Any:  # noqa: ANN401
        """按点分路径读取租户设置。"""
        node: Any = self.settings
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node if node is not None else default

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "slug": self.slug,
            "name": self.name,
            "plan": self.plan,
            "status": self.status,
        }

    @classmethod
    def from_model(cls, tenant: Tenant) -> "TenantContext":
        return cls(
            tenant_id=str(tenant.id),
            slug=str(tenant.slug or ""),
            name=str(tenant.name or ""),
            plan=str(tenant.plan or "free"),
            status=str(tenant.status or TenantStatus.ACTIVE),
            settings=dict(tenant.settings or {}),
        )


def current_tenant_id() -> str:
    """从 contextvar 读取当前租户（结构化日志与后台任务用）。"""
    try:
        from omnistream.core.context import get_tenant_id

        return str(get_tenant_id() or "")
    except Exception:  # noqa: BLE001
        return ""


def slug_from_host(host: str) -> str:
    """从 Host 头解析子域名 slug；无法解析返回空串。"""
    if not host:
        return ""
    hostname = host.split(":", 1)[0].strip().lower()
    if not hostname or hostname.replace(".", "").isdigit():
        return ""
    labels = hostname.split(".")
    if len(labels) < 3:
        return ""
    candidate = labels[0]
    if candidate in _RESERVED_SLUGS or not _SLUG_RE.match(candidate):
        return ""
    return candidate


def validate_slug(slug: str) -> str:
    """校验并归一化租户 slug。"""
    value = (slug or "").strip().lower()
    if not _SLUG_RE.match(value):
        raise ValidationError(
            "slug 必须是 2-63 位小写字母/数字/连字符，且以字母数字开头",
            code=ErrorCode.INVALID_REQUEST,
            detail={"slug": slug},
        )
    if value in _RESERVED_SLUGS:
        raise ValidationError("该 slug 为保留字", code=ErrorCode.INVALID_REQUEST, detail={"slug": value})
    return value


class TenantResolver:
    """租户解析器。所有方法都直连数据库，调用方负责缓存。"""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def by_id(self, tenant_id: str) -> Tenant | None:
        if not tenant_id:
            return None
        stmt = select(Tenant).where(Tenant.id == tenant_id, Tenant.deleted_at.is_(None))
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def by_slug(self, slug: str) -> Tenant | None:
        if not slug:
            return None
        stmt = select(Tenant).where(
            Tenant.slug == slug.strip().lower(), Tenant.deleted_at.is_(None)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def require(self, tenant_id: str) -> Tenant:
        tenant = await self.by_id(tenant_id)
        if tenant is None:
            raise NotFoundError(
                f"租户不存在: {tenant_id}", code=ErrorCode.NOT_FOUND, detail={"tenant_id": tenant_id}
            )
        return tenant

    @staticmethod
    def ensure_active(tenant: Tenant) -> Tenant:
        """租户被停用/删除时拒绝服务。"""
        status = str(tenant.status or "")
        if status != str(TenantStatus.ACTIVE):
            raise AuthError(
                f"租户状态不可用: {status}",
                code=ErrorCode.TENANT_SUSPENDED,
                detail={"tenant_id": str(tenant.id), "status": status},
            )
        return tenant

    async def resolve(
        self,
        *,
        principal: "Principal | None" = None,
        header_tenant_id: str = "",
        header_tenant_slug: str = "",
        host: str = "",
        allow_default: bool = True,
    ) -> TenantContext:
        """按优先级解析租户并校验一致性。"""
        candidate_id = str(getattr(principal, "tenant_id", "") or "")

        requested_id = (header_tenant_id or "").strip()
        requested_slug = (header_tenant_slug or "").strip().lower() or slug_from_host(host)

        tenant: Tenant | None = None
        if candidate_id:
            tenant = await self.by_id(candidate_id)
        if tenant is None and requested_id:
            tenant = await self.by_id(requested_id)
        if tenant is None and requested_slug:
            tenant = await self.by_slug(requested_slug)
        if tenant is None and allow_default:
            tenant = await self.by_id(DEFAULT_TENANT_ID) or await self.by_slug(DEFAULT_TENANT_SLUG)

        if tenant is None:
            raise NotFoundError(
                "无法解析租户，请携带 X-Tenant-Id 或使用有效的 API Key",
                code=ErrorCode.NOT_FOUND,
            )

        self.ensure_active(tenant)
        resolved_id = str(tenant.id)

        # 主体已绑定租户时，禁止通过请求头切换到别的租户（平台超管除外）
        if candidate_id and resolved_id != candidate_id:
            is_super = bool(principal is not None and principal.has("system:admin"))
            if not is_super:
                raise AuthError(
                    "跨租户访问被拒绝",
                    code=ErrorCode.FORBIDDEN,
                    detail={"principal_tenant": candidate_id, "target_tenant": resolved_id},
                )

        return TenantContext.from_model(tenant)


async def ensure_default_tenant(session: AsyncSession) -> Tenant:
    """确保默认租户存在（单机/演示部署的零配置启动路径）。"""
    resolver = TenantResolver(session)
    tenant = await resolver.by_id(DEFAULT_TENANT_ID)
    if tenant is not None:
        return tenant
    tenant = await resolver.by_slug(DEFAULT_TENANT_SLUG)
    if tenant is not None:
        return tenant

    tenant = Tenant(
        id=DEFAULT_TENANT_ID,
        name="Default Tenant",
        slug=DEFAULT_TENANT_SLUG,
        status=str(TenantStatus.ACTIVE),
        plan=str(setting("security.default_plan", "free")),
        is_default=True,
        description="单机部署自动创建的默认租户",
        settings={},
    )
    session.add(tenant)
    await session.flush()
    return tenant

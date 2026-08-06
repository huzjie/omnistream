"""API Key 签发、存储与校验。

明文密钥形如 ``oms_live_<40 位随机>``，**只在签发时返回一次**，库里只存：

* ``prefix`` —— 明文随机部分的前 8 位，用于按前缀快速定位（唯一索引）；
* ``key_hash`` —— 对完整明文做 HMAC-SHA256（以 ``security.api_key_salt`` 为 key）后的摘要。

校验时按 prefix 取出记录，再用恒定时间比较 HMAC，杜绝时序侧信道。未配置外部依赖
（如 Redis）时，本模块不依赖任何可选包，纯标准库即可跑通。
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from sqlalchemy import func, select

from omnistream.core.clock import utcnow
from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import AuthError, ConflictError, NotFoundError, ValidationError
from omnistream.db.models.api_key import ApiKey
from omnistream.db.session import session_scope
from omnistream.schemas.auth import Permission, Principal, Role
from omnistream.platform.auth.principal import principal_from_api_key

if TYPE_CHECKING:
    from omnistream.config.security import SecuritySettings

__all__ = [
    "API_KEY_ALPHABET",
    "ApiKeyService",
    "GeneratedApiKey",
    "IssuedApiKey",
    "extract_prefix",
    "generate_api_key",
    "hash_api_key",
    "looks_like_api_key",
    "mask_api_key",
    "verify_api_key",
    "verify_api_key_hash",
]

#: 不含易混淆字符（0/O/1/l/I）的 Base58 风格字母表
API_KEY_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_PREFIX_LEN = 8
_BODY_LEN = 40


@dataclass(slots=True)
class GeneratedApiKey:
    """一次签发生成的完整密钥。``plaintext`` 仅此一次可见。"""

    plaintext: str
    prefix: str
    body: str
    hash: str
    environment: str = "live"


@dataclass(slots=True)
class IssuedApiKey:
    """签发结果：ORM 记录 + 一次性明文。"""

    key: ApiKey
    plaintext: str


def _gen_body() -> str:
    return "".join(secrets.choice(API_KEY_ALPHABET) for _ in range(_BODY_LEN))


def generate_api_key(
    environment: str = "live",
    *,
    security: "SecuritySettings | None" = None,
) -> GeneratedApiKey:
    """生成一个新的 API Key 明文（尚未落库）。"""
    from omnistream.core.constants import API_KEY_PLAINTEXT_PREFIX

    body = _gen_body()
    plaintext = f"{API_KEY_PLAINTEXT_PREFIX}{body}"
    prefix = body[:_PREFIX_LEN]
    key_hash = hash_api_key(plaintext, security) if security is not None else ""
    return GeneratedApiKey(
        plaintext=plaintext,
        prefix=prefix,
        body=body,
        hash=key_hash,
        environment=environment,
    )


def hash_api_key(plaintext: str, security: "SecuritySettings") -> str:
    """对明文密钥做 HMAC-SHA256，返回 ``hmac-sha256$<hex>``。"""
    secret = (security.api_key_salt or "").encode("utf-8")
    digest = hmac.new(secret, plaintext.encode("utf-8"), hashlib.sha256).digest()
    return "hmac-sha256$" + digest.hex()


def verify_api_key_hash(plaintext: str, stored_hash: str) -> bool:
    """恒定时间校验明文与存储哈希是否一致。"""
    if not stored_hash or not plaintext:
        # 仍做一次 HMAC 以缓解「用户不存在」时间差探测
        hmac.new(b"x", b"x", hashlib.sha256).digest()
        return False
    if not stored_hash.startswith("hmac-sha256$"):
        return False
    expected = stored_hash.split("$", 1)[1]
    actual = hmac.new(b"x", plaintext.encode("utf-8"), hashlib.sha256).digest().hex()
    return hmac.compare_digest(actual, expected)


def extract_prefix(raw: str) -> str | None:
    """从明文密钥提取用于定位的 prefix。"""
    from omnistream.core.constants import API_KEY_PLAINTEXT_PREFIX

    if not raw.startswith(API_KEY_PLAINTEXT_PREFIX):
        return None
    body = raw[len(API_KEY_PLAINTEXT_PREFIX):]
    if len(body) < _PREFIX_LEN:
        return None
    return body[:_PREFIX_LEN]


def looks_like_api_key(raw: str) -> bool:
    """快速判断字符串是否形如 API Key（用于路由鉴权分支）。"""
    from omnistream.core.constants import API_KEY_PLAINTEXT_PREFIX

    if not raw or not raw.startswith(API_KEY_PLAINTEXT_PREFIX):
        return False
    return len(raw) > len(API_KEY_PLAINTEXT_PREFIX) + _PREFIX_LEN


def mask_api_key(raw: str) -> str:
    """把明文密钥做成可展示的掩码，避免日志泄露。"""
    from omnistream.core.constants import API_KEY_PLAINTEXT_PREFIX

    if not looks_like_api_key(raw):
        return "***"
    body = raw[len(API_KEY_PLAINTEXT_PREFIX):]
    return f"{API_KEY_PLAINTEXT_PREFIX}{body[:4]}{'*' * (len(body) - 8)}{body[-4:]}"


async def verify_api_key(raw: str, security: "SecuritySettings") -> Principal | None:
    """校验明文密钥并返回主体；失败（格式错 / 不存在 / 已吊销 / 哈希不符）返回 ``None``。

    供 :mod:`omnistream.api.deps` 调用。成功会顺带更新 ``last_used_at`` 与 ``usage_count``。
    """
    raw = (raw or "").strip()
    if not looks_like_api_key(raw):
        return None
    prefix = extract_prefix(raw)
    assert prefix is not None

    async with session_scope() as session:
        key = (
            await session.execute(select(ApiKey).where(ApiKey.prefix == prefix))
        ).scalar_one_or_none()
        if key is None:
            return None
        if not key.is_usable(utcnow()):
            return None
        if not verify_api_key_hash(raw, key.key_hash):
            return None
        key.last_used_at = utcnow()
        key.usage_count = int(key.usage_count or 0) + 1
        await session.flush()
        return principal_from_api_key(key)


class ApiKeyService:
    """API Key 的增删改查（供 ``api/routers/api_keys.py`` 使用）。"""

    def __init__(self, session: object) -> None:
        # session 类型在运行时为 AsyncSession；此处仅做鸭子类型约定
        self.session = session

    async def issue(
        self,
        tenant_id: str,
        *,
        name: str = "",
        role: "Role | str" = Role.SERVICE,
        scopes: "list[Permission | str] | None" = None,
        environment: str = "live",
        ttl_days: int | None = None,
        created_by: str | None = None,
        security: "SecuritySettings",
        max_keys: int = 200,
    ) -> IssuedApiKey:
        """签发一个新 Key。"""
        from omnistream.core.ids import new_api_key_id

        role_value = _safe_role_value(role)
        scope_values = [str(s) for s in (_safe_scopes_list(scopes))]

        active = await self.count_active(tenant_id)
        if active >= max_keys:
            raise ConflictError(
                "密钥数量已达上限",
                code=ErrorCode.CONFLICT,
                detail={"limit": max_keys, "active": active},
            )

        generated = generate_api_key(environment, security=security)
        now = utcnow()
        key = ApiKey(
            id=new_api_key_id(),
            tenant_id=tenant_id,
            name=name or "default",
            prefix=generated.prefix,
            key_hash=generated.hash,
            scopes=scope_values,
            role=role_value,
            environment=environment,
            enabled=True,
            expires_at=(now + timedelta(days=ttl_days)) if ttl_days else None,
            created_by=created_by,
        )
        self.session.add(key)  # type: ignore[attr-defined]
        await self.session.flush()  # type: ignore[attr-defined]
        return IssuedApiKey(key=key, plaintext=generated.plaintext)

    async def list_for_tenant(self, tenant_id: str) -> list[ApiKey]:
        result = await self.session.execute(  # type: ignore[attr-defined]
            select(ApiKey)
            .where(ApiKey.tenant_id == tenant_id)
            .order_by(ApiKey.created_at.desc())
        )
        return list(result.scalars().all())

    async def count_active(self, tenant_id: str) -> int:
        result = await self.session.execute(  # type: ignore[attr-defined]
            select(func.count())
            .select_from(ApiKey)
            .where(ApiKey.tenant_id == tenant_id, ApiKey.enabled.is_(True), ApiKey.revoked_at.is_(None))
        )
        return int(result.scalar_one() or 0)

    async def get(self, key_id: str, tenant_id: str) -> ApiKey | None:
        return (
            await self.session.execute(  # type: ignore[attr-defined]
                select(ApiKey).where(ApiKey.id == key_id, ApiKey.tenant_id == tenant_id)
            )
        ).scalar_one_or_none()

    async def revoke(
        self, key_id: str, tenant_id: str, *, reason: str | None = None
    ) -> bool:
        key = await self.get(key_id, tenant_id)
        if key is None:
            return False
        key.revoked_at = utcnow()
        key.revoked_reason = reason
        key.enabled = False
        await self.session.flush()  # type: ignore[attr-defined]
        return True

    async def update(
        self,
        key_id: str,
        tenant_id: str,
        *,
        name: str | None = None,
        enabled: bool | None = None,
    ) -> ApiKey | None:
        key = await self.get(key_id, tenant_id)
        if key is None:
            return None
        if name is not None:
            key.name = name
        if enabled is not None:
            key.enabled = enabled
            if enabled:
                key.revoked_at = None
        await self.session.flush()  # type: ignore[attr-defined]
        return key

    async def rotate(
        self,
        key_id: str,
        tenant_id: str,
        *,
        security: "SecuritySettings",
        ttl_days: int | None = None,
    ) -> IssuedApiKey | None:
        """吊销旧 Key 并签发同权限的新 Key。"""
        old = await self.get(key_id, tenant_id)
        if old is None:
            return None
        issued = await self.issue(
            tenant_id,
            name=old.name,
            role=Role(old.role),
            scopes=[Permission(s) for s in old.scopes],
            environment=old.environment,
            ttl_days=ttl_days,
            created_by=old.created_by,
            security=security,
        )
        old.revoked_at = utcnow()
        old.revoked_reason = "rotated"
        old.enabled = False
        await self.session.flush()  # type: ignore[attr-defined]
        return issued


def _safe_role_value(role: "Role | str") -> str:
    try:
        return str(Role(role))
    except (ValueError, TypeError):
        return str(Role.SERVICE)


def _safe_scopes_list(scopes: object) -> list[str]:
    out: list[str] = []
    if not isinstance(scopes, (list, tuple, set, frozenset)):
        return out
    for raw in scopes:
        try:
            out.append(str(Permission(raw)))
        except (ValueError, TypeError):
            continue
    return out

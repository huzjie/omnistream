"""认证聚合层。

把 API Key / JWT / 密码 三类凭据的验证统一封装为可直接被路由层调用的函数：

* :func:`authenticate_user` —— 控制台邮箱 + 密码登录（含锁定策略、密码透明升级）；
* :func:`tokens_for_principal` —— 为主体签发 access/refresh 令牌对；
* :func:`authenticate_request` —— 由请求头解析主体（供 WS / 非 FastAPI 入口复用）；
* :func:`describe_auth_backends` —— 自检用。

具体的 FastAPI 依赖（``get_principal`` / ``require_perm``）由 ``omnistream.api.deps`` 提供，
本模块只负责「凭据 -> Principal / Token」这一层逻辑，不耦合 Web 框架。
"""
from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from omnistream.core.clock import utcnow
from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import AuthError
from omnistream.schemas.auth import Permission, Principal, Role, TokenPair

from .api_key import (
    ApiKeyService,
    generate_api_key,
    hash_api_key,
    looks_like_api_key,
    mask_api_key,
    verify_api_key,
    verify_api_key_hash,
)
from .jwt import (
    ACCESS_TOKEN_TYPE,
    JwtConfig,
    REFRESH_TOKEN_TYPE,
    decode_access_token,
    decode_token,
    encode_token,
    is_revoked,
    issue_access_token,
    issue_refresh_token,
    issue_token_pair,
    jwt_config,
    revoke_all_for_subject,
    revoke_token,
)
from .password import (
    check_strength,
    generate_password,
    hash_password,
    needs_rehash,
    verify_password,
)
from .principal import (
    SYSTEM_TENANT_ID,
    principal_from_api_key,
    principal_from_claims,
    principal_from_user,
    system_principal,
)

if TYPE_CHECKING:
    from omnistream.config.security import SecuritySettings

__all__ = [
    "ACCESS_TOKEN_TYPE",
    "ApiKeyService",
    "JwtConfig",
    "MAX_LOGIN_FAILURES",
    "LOCKOUT_MINUTES",
    "REFRESH_TOKEN_TYPE",
    "SYSTEM_TENANT_ID",
    "TokenPair",
    "authenticate_request",
    "authenticate_user",
    "decode_access_token",
    "decode_token",
    "describe_auth_backends",
    "encode_token",
    "generate_api_key",
    "generate_password",
    "hash_api_key",
    "hash_password",
    "is_revoked",
    "issue_access_token",
    "issue_refresh_token",
    "issue_token_pair",
    "jwt_config",
    "looks_like_api_key",
    "mask_api_key",
    "needs_rehash",
    "principal_from_api_key",
    "principal_from_claims",
    "principal_from_user",
    "revoke_all_for_subject",
    "revoke_token",
    "system_principal",
    "tokens_for_principal",
    "verify_api_key",
    "verify_api_key_hash",
    "verify_password",
]

#: 连续登录失败达到此次数后锁定账号
MAX_LOGIN_FAILURES = 8
#: 账号锁定时长（分钟）
LOCKOUT_MINUTES = 15


async def authenticate_user(
    session: Any,
    *,
    email: str,
    password: str,
    tenant_slug: str | None = None,
    security: "SecuritySettings",
    client_ip: str | None = None,
    user_agent: str | None = None,
) -> tuple[Any, Principal]:
    """邮箱 + 密码登录。成功返回 ``(User ORM, Principal)``，失败抛 ``AuthError``。

    实现要点：

    * 优先按 ``tenant_slug`` 定位租户，否则落到默认租户；
    * 租户禁用 / 用户禁用 / 账号锁定 一律 401（不区分原因，避免信息泄露）；
    * 失败累加 ``failed_login_count``，达到阈值锁定 ``LOCKOUT_MINUTES`` 分钟；
    * 成功重置失败计数，并在算法升级时透明刷新密码哈希。
    """
    from omnistream.db.models.tenant import Tenant
    from omnistream.db.models.user import User

    now = utcnow()
    tenant = None
    if tenant_slug:
        tenant = (
            await session.execute(select(Tenant).where(Tenant.slug == tenant_slug))
        ).scalar_one_or_none()
    if tenant is None:
        tenant = (
            await session.execute(select(Tenant).where(Tenant.is_default.is_(True)))
        ).scalar_one_or_none()
    if tenant is None or not tenant.active:
        raise AuthError("租户不可用", code=ErrorCode.TENANT_SUSPENDED)

    user = (
        await session.execute(
            select(User).where(User.tenant_id == tenant.id, User.email == email)
        )
    ).scalar_one_or_none()
    if user is None or not user.active:
        raise AuthError("账号或密码错误", code=ErrorCode.UNAUTHENTICATED)
    if user.is_locked(now):
        raise AuthError(
            "账号已锁定，请稍后再试",
            code=ErrorCode.UNAUTHENTICATED,
            detail={"locked_until": user.locked_until.isoformat() if user.locked_until else None},
        )

    if not verify_password(password, user.password_hash):
        user.failed_login_count = int(user.failed_login_count or 0) + 1
        if user.failed_login_count >= MAX_LOGIN_FAILURES:
            user.locked_until = now + timedelta(minutes=LOCKOUT_MINUTES)
        await session.flush()
        raise AuthError("账号或密码错误", code=ErrorCode.UNAUTHENTICATED)

    # 登录成功：重置失败计数 + 透明升级哈希
    user.failed_login_count = 0
    user.locked_until = None
    user.last_login_at = now
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(password)
    await session.flush()

    principal = principal_from_user(user, tenant_id=tenant.id)
    return user, principal


async def tokens_for_principal(principal: Principal, *, security: "SecuritySettings") -> TokenPair:
    """为主体签发 access/refresh 令牌对。"""
    return issue_token_pair(
        sub=principal.id,
        tid=principal.tenant_id,
        role=principal.role,
        scopes=list(principal.scopes),
        security=security,
    )


async def authenticate_request(
    session: Any,
    *,
    authorization: str | None = None,
    api_key_header: str | None = None,
    client_ip: str | None = None,
    user_agent: str | None = None,
    security: "SecuritySettings | None" = None,
    required: bool = False,
) -> Principal | None:
    """由请求头解析主体。供 WS / 非 FastAPI 入口复用。

    识别顺序：``X-Api-Key`` -> ``Authorization: Bearer``（按前缀区分 Key / JWT）。
    """
    from omnistream.core.constants import API_KEY_PLAINTEXT_PREFIX

    raw: str | None = None
    kind = "apikey"
    if api_key_header:
        raw = api_key_header.strip()
        kind = "apikey"
    elif authorization:
        value = authorization.strip()
        if value.lower().startswith("bearer "):
            token = value[len("bearer "):].strip()
            if token.startswith(API_KEY_PLAINTEXT_PREFIX):
                raw, kind = token, "apikey"
            else:
                raw, kind = token, "jwt"
        elif value.startswith(API_KEY_PLAINTEXT_PREFIX):
            raw, kind = value, "apikey"

    if raw is None:
        if required:
            raise AuthError("需要身份认证", code=ErrorCode.UNAUTHENTICATED)
        return None

    if security is None:
        raise AuthError("未提供安全配置", code=ErrorCode.CONFIG_MISSING)

    if kind == "jwt":
        claims = decode_access_token(raw, security)
        if await is_revoked(claims.jti):
            raise AuthError("令牌已吊销", code=ErrorCode.TOKEN_EXPIRED)
        return principal_from_claims(claims)

    principal = await verify_api_key(raw, security)
    if principal is None:
        raise AuthError("API Key 无效或已吊销", code=ErrorCode.INVALID_API_KEY)
    return principal


def describe_auth_backends() -> list[str]:
    """返回当前启用的认证后端（自检 / 文档用）。"""
    return ["api_key", "jwt"]

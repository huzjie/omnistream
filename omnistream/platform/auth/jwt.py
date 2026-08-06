"""JWT 签发与校验（零第三方依赖）。

默认实现使用 HS256（``hmac`` + ``base64`` 标准库即可），生产可切 RS256
（需 ``cryptography``，可选）。对外契约与 :mod:`omnistream.api.deps` 对齐：

* :func:`decode_access_token(token, security) -> JWTClaims` —— 鉴权中间件调用；
* :func:`issue_token_pair(sub, tid, role, scopes, security) -> TokenPair` —— 登录签发。

黑名单（jti 吊销）使用进程内存储，多实例部署时建议改用 Redis（见 :mod:`omnistream.platform.tenancy`）。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import AuthError, ConfigError
from omnistream.schemas.auth import JWTClaims, Permission, Principal, Role, TokenPair

if TYPE_CHECKING:
    from omnistream.config.security import SecuritySettings

__all__ = [
    "ACCESS_TOKEN_TYPE",
    "JwtConfig",
    "REFRESH_TOKEN_TYPE",
    "decode_access_token",
    "decode_token",
    "encode_token",
    "is_revoked",
    "issue_access_token",
    "issue_refresh_token",
    "issue_token_pair",
    "jwt_config",
    "revoke_all_for_subject",
    "revoke_token",
]

ACCESS_TOKEN_TYPE = "access"
REFRESH_TOKEN_TYPE = "refresh"

# --------------------------------------------------------------------------- 黑名单
# 进程内存储：key = jti，value = 过期时间戳（Unix 秒）。多实例需换 Redis。
_BLACKLIST: dict[str, float] = {}
_BLACKLIST_LOCK: Any = None  # 延迟初始化为 asyncio.Lock


def _lock() -> Any:
    global _BLACKLIST_LOCK
    if _BLACKLIST_LOCK is None:
        import asyncio

        try:
            _BLACKLIST_LOCK = asyncio.Lock()
        except RuntimeError:  # 无事件循环（极少见）
            _BLACKLIST_LOCK = _DummyLock()
    return _BLACKLIST_LOCK


class _DummyLock:
    """无事件循环时的退化锁，保证单线程安全即可。"""

    async def __aenter__(self) -> "_DummyLock":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


@dataclass(slots=True)
class JwtConfig:
    """JWT 相关参数（由 :class:`SecuritySettings` 构建）。"""

    secret: str
    algorithm: str = "HS256"
    issuer: str = "omnistream"
    audience: str = "omnistream-console"
    access_ttl_s: int = 3600
    refresh_ttl_s: int = 604_800

    def validate(self) -> None:
        if self.algorithm.startswith("HS"):
            if not self.secret or len(self.secret) < 16:
                raise ConfigError(
                    "JWT 密钥过短，生产环境必须通过环境变量覆盖 jwt_secret",
                    code=ErrorCode.CONFIG_INVALID,
                )
        else:
            raise ConfigError(
                f"OSS-lite 仅支持 HS256/HS512，当前算法 {self.algorithm} 需要 cryptography 依赖",
                code=ErrorCode.CONFIG_INVALID,
            )


def jwt_config(security: "SecuritySettings") -> JwtConfig:
    """由安全配置构建 :class:`JwtConfig`。"""
    return JwtConfig(
        secret=security.jwt_secret,
        algorithm=security.jwt_algorithm,
        issuer=security.jwt_issuer,
        audience=security.jwt_audience,
        access_ttl_s=security.access_token_ttl_s,
        refresh_ttl_s=security.refresh_token_ttl_s,
    )


# --------------------------------------------------------------------------- 编解码
def _b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def _hash_for(algorithm: str) -> Any:
    return {
        "HS256": hashlib.sha256,
        "HS384": hashlib.sha384,
        "HS512": hashlib.sha512,
    }.get(algorithm, hashlib.sha256)


def _sign(header_b64: str, payload_b64: str, config: JwtConfig) -> str:
    if not config.algorithm.startswith("HS"):
        raise ConfigError(
            f"不支持的 JWT 算法: {config.algorithm}",
            code=ErrorCode.CONFIG_INVALID,
        )
    msg = f"{header_b64}.{payload_b64}".encode("utf-8")
    digest = hmac.new(config.secret.encode("utf-8"), msg, _hash_for(config.algorithm)).digest()
    return _b64e(digest)


def encode_token(claims: dict[str, Any], config: JwtConfig) -> str:
    """编码 claims 为 JWT 字符串（不自动加 iat/exp，由调用方决定）。"""
    config.validate()
    header = {"alg": config.algorithm, "typ": "JWT"}
    header_b64 = _b64e(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64e(json.dumps(claims, separators=(",", ":")).encode("utf-8"))
    sig = _sign(header_b64, payload_b64, config)
    return f"{header_b64}.{payload_b64}.{sig}"


def decode_token(token: str, config: JwtConfig) -> dict[str, Any]:
    """底层解码 + 验签，返回 claims dict。任何异常统一为 ``AuthError``。"""
    if not token or not isinstance(token, str):
        raise AuthError("缺少令牌", code=ErrorCode.UNAUTHENTICATED)
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthError("令牌格式错误", code=ErrorCode.UNAUTHENTICATED)
    header_b64, payload_b64, sig = parts
    expected = _sign(header_b64, payload_b64, config)
    if not hmac.compare_digest(sig, expected):
        raise AuthError("令牌签名无效", code=ErrorCode.UNAUTHENTICATED)
    try:
        header = json.loads(_b64d(header_b64))
        claims = json.loads(_b64d(payload_b64))
    except (ValueError, json.JSONDecodeError) as exc:
        raise AuthError("令牌载荷损坏", code=ErrorCode.UNAUTHENTICATED) from exc
    # 防御 alg 混淆 / 无签名攻击
    if str(header.get("alg")) != config.algorithm:
        raise AuthError("令牌算法不匹配", code=ErrorCode.UNAUTHENTICATED)
    return claims


def _claims_to_model(claims: dict[str, Any]) -> JWTClaims:
    scopes_raw = claims.get("scopes") or []
    scopes = [Permission(s) for s in scopes_raw if _is_permission(s)]
    return JWTClaims(
        sub=str(claims.get("sub", "")),
        tid=str(claims.get("tid", "")),
        role=_safe_role(claims.get("role", "viewer")),
        scopes=scopes,
        iss=str(claims.get("iss", "")),
        aud=str(claims.get("aud", "")),
        iat=int(claims.get("iat", 0) or 0),
        exp=int(claims.get("exp", 0) or 0),
        jti=str(claims.get("jti", "")),
        sid=claims.get("sid"),
    )


def _is_permission(value: object) -> bool:
    try:
        Permission(str(value))
        return True
    except (ValueError, TypeError):
        return False


def _safe_role(value: object) -> Role:
    try:
        return Role(str(value))
    except (ValueError, TypeError):
        return Role.VIEWER


def decode_access_token(token: str, security: "SecuritySettings") -> JWTClaims:
    """解码访问令牌为 :class:`JWTClaims`，校验签名 / 发行方 / 受众 / 过期。

    供 :mod:`omnistream.api.deps` 调用。
    """
    config = jwt_config(security)
    claims = decode_token(token, config)
    if claims.get("iss") != config.issuer:
        raise AuthError("令牌发行方不匹配", code=ErrorCode.UNAUTHENTICATED)
    if claims.get("aud") != config.audience:
        raise AuthError("令牌受众不匹配", code=ErrorCode.UNAUTHENTICATED)
    exp = int(claims.get("exp", 0) or 0)
    if exp and exp < int(time.time()):
        raise AuthError("令牌已过期", code=ErrorCode.TOKEN_EXPIRED)
    return _claims_to_model(claims)


def _build_claims(
    *,
    sub: str,
    tid: str,
    role: "Role | str",
    scopes: "list[Permission | str] | None",
    config: JwtConfig,
    typ: str,
    ttl: int,
    sid: str | None = None,
) -> dict[str, Any]:
    now = int(time.time())
    role_value = _safe_role(role).value
    scope_values = [str(s) for s in scopes if _is_permission(s)]
    return {
        "sub": sub,
        "tid": tid,
        "role": role_value,
        "scopes": scope_values,
        "iss": config.issuer,
        "aud": config.audience,
        "iat": now,
        "exp": now + ttl,
        "jti": uuid.uuid4().hex,
        "typ": typ,
        "sid": sid,
    }


def issue_access_token(
    *,
    sub: str,
    tid: str,
    role: "Role | str",
    scopes: "list[Permission | str] | None" = None,
    security: "SecuritySettings",
    sid: str | None = None,
    ttl: int | None = None,
) -> tuple[str, JWTClaims]:
    """签发访问令牌（默认 1h）。返回 ``(token, claims)``。"""
    config = jwt_config(security)
    claims = _build_claims(
        sub=sub, tid=tid, role=role, scopes=scopes, config=config,
        typ=ACCESS_TOKEN_TYPE, ttl=ttl or config.access_ttl_s, sid=sid,
    )
    token = encode_token(claims, config)
    return token, _claims_to_model(claims)


def issue_refresh_token(
    *,
    sub: str,
    tid: str,
    role: "Role | str",
    scopes: "list[Permission | str] | None" = None,
    security: "SecuritySettings",
    ttl: int | None = None,
) -> tuple[str, JWTClaims]:
    """签发刷新令牌（默认 7d）。"""
    config = jwt_config(security)
    claims = _build_claims(
        sub=sub, tid=tid, role=role, scopes=scopes, config=config,
        typ=REFRESH_TOKEN_TYPE, ttl=ttl or config.refresh_ttl_s,
    )
    token = encode_token(claims, config)
    return token, _claims_to_model(claims)


def issue_token_pair(
    sub: str,
    tid: str,
    role: "Role | str",
    scopes: "list[Permission | str] | None" = None,
    *,
    security: "SecuritySettings",
    sid: str | None = None,
) -> TokenPair:
    """签发 access + refresh 令牌对。"""
    config = jwt_config(security)
    access, _ = issue_access_token(
        sub=sub, tid=tid, role=role, scopes=scopes, security=security, sid=sid
    )
    refresh, _ = issue_refresh_token(
        sub=sub, tid=tid, role=role, scopes=scopes, security=security
    )
    principal = Principal(
        id=sub,
        tenant_id=tid,
        kind="user",
        role=_safe_role(role),
        scopes=[Permission(s) for s in scopes if _is_permission(s)],
    )
    return TokenPair(
        access_token=access,
        refresh_token=refresh,
        expires_in=config.access_ttl_s,
        principal=principal,
    )


# --------------------------------------------------------------------------- 吊销
async def revoke_token(jti: str, *, ttl_s: int | None = None) -> None:
    """把某个 jti 加入黑名单直到其自然过期。"""
    if not jti:
        return
    import asyncio

    async with _lock():
        expire = time.time() + (ttl_s if ttl_s else 0)
        # 清掉已过期的旧条目
        now = time.time()
        expired = [k for k, v in _BLACKLIST.items() if v and v < now]
        for k in expired:
            _BLACKLIST.pop(k, None)
        _BLACKLIST[jti] = expire


async def is_revoked(jti: str) -> bool:
    """判断 jti 是否已被吊销。"""
    if not jti:
        return False
    async with _lock():
        expire = _BLACKLIST.get(jti)
        if expire is None:
            return False
        if expire and expire < time.time():
            _BLACKLIST.pop(jti, None)
            return False
        return True


async def revoke_all_for_subject(sub: str) -> None:
    """（占位）按主体吊销全部活跃令牌。OSS-lite 使用进程内黑名单，
    无法跨实例按 subject 检索，这里仅记录日志，生产应结合 Redis 实现。"""
    import structlog

    log = structlog.get_logger(__name__)
    log.warning("revoke_all_for_subject_not_supported_inproc", subject=sub)

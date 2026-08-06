"""认证路由：登录 / 刷新 / 登出 / 当前用户。

所有接口走 :mod:`omnistream.api.deps` 的统一鉴权与错误渲染。登录走邮箱 + 密码，
成功签发 access/refresh 令牌对；刷新用 refresh 令牌换发新对。
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from omnistream.api.deps import CurrentPrincipal, DbSession
from omnistream.config.settings import get_settings
from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import AuthError
from omnistream.schemas.auth import JWTClaims, LoginRequest, Permission, Principal, TokenPair
from omnistream.schemas.common import Ok
from omnistream.platform.auth import (
    authenticate_user,
    is_revoked,
    principal_from_claims,
    tokens_for_principal,
)
from omnistream.platform.auth.jwt import decode_token, jwt_config

router = APIRouter()
ROUTER_ORDER = 10
ROUTER_TAGS = ["auth"]


class RefreshRequest(BaseModel):
    refresh_token: str = ""


class LogoutRequest(BaseModel):
    refresh_token: str | None = None


@router.post("/login", response_model=Ok[TokenPair])
async def login(body: LoginRequest, session: DbSession, request: Request) -> Ok[TokenPair]:
    """邮箱 + 密码登录，返回令牌对。"""
    security = get_settings().security
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    _user, principal = await authenticate_user(
        session,
        email=body.email,
        password=body.password,
        tenant_slug=body.tenant_slug,
        security=security,
        client_ip=client_ip,
        user_agent=user_agent,
    )
    pair = await tokens_for_principal(principal, security=security)
    return Ok.of(pair)


@router.post("/refresh", response_model=Ok[TokenPair])
async def refresh(body: RefreshRequest, session: DbSession) -> Ok[TokenPair]:
    """用刷新令牌换发新的令牌对。"""
    security = get_settings().security
    config = jwt_config(security)
    raw = decode_token(body.refresh_token, config)
    if raw.get("typ") != "refresh":
        raise AuthError("不是刷新令牌", code=ErrorCode.TOKEN_EXPIRED)
    jti = str(raw.get("jti", ""))
    if await is_revoked(jti):
        raise AuthError("令牌已吊销", code=ErrorCode.TOKEN_EXPIRED)
    claims = JWTClaims(
        sub=str(raw.get("sub", "")),
        tid=str(raw.get("tid", "")),
        role=raw.get("role", "viewer"),
        scopes=list(raw.get("scopes", [])),
        jti=jti,
        sid=raw.get("sid"),
    )
    principal = principal_from_claims(claims)
    pair = await tokens_for_principal(principal, security=security)
    return Ok.of(pair)


@router.post("/logout", response_model=Ok[None])
async def logout(
    principal: CurrentPrincipal, body: LogoutRequest | None = None
) -> Ok[None]:
    """登出（无状态 JWT 无法真正失效，这里仅做约定入口；如需强失效请用黑名单）。"""
    return Ok.of(None)


@router.get("/me", response_model=Ok[dict])
async def me(principal: CurrentPrincipal) -> Ok[dict]:
    """返回当前调用主体信息。"""
    return Ok.of(principal.model_dump(mode="json"))

"""FastAPI 依赖注入集合。

这是 WS1 提供给所有上层 WS（会话 API、Provider 管理台、治理后台……）的**唯一**
依赖来源。任何 router 都不应该自己重新实现鉴权 / 取库 / 分页解析，否则各 WS 的语义会
迅速漂移。

设计约束：

* **零硬依赖**：``omnistream.db`` 与 ``omnistream.platform.auth`` 属于其他工作包，
  本模块一律 **延迟导入 + 优雅降级**，保证 WS1 单独 clone 下来也能 ``uvicorn`` 起来；
* **dev 友好**：``env=dev`` 且未带任何凭证时注入一个本地开发主体，
  免去「跑个 demo 还要先建租户」的摩擦；``env=prod`` 下永远强制鉴权；
* **错误统一**：所有失败都抛 :class:`~omnistream.core.errors.OmniError` 子类，
  由 :mod:`omnistream.api.errors` 统一渲染成 ``{code,message,detail,request_id}``。
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Depends, Header, Query, Request

from omnistream.config.settings import Settings, get_settings
from omnistream.core.constants import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT
from omnistream.core.context import get_request_id, set_tenant_id
from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import AuthError, OmniError, PermissionError_
from omnistream.core.logging import get_logger
from omnistream.schemas.auth import Permission, Principal, Role

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "CurrentPrincipal",
    "CurrentSettings",
    "CurrentTenantId",
    "DbSession",
    "Pagination",
    "PaginationParams",
    "get_db",
    "get_optional_principal",
    "get_pagination",
    "get_principal",
    "get_request_id_dep",
    "get_settings_dep",
    "get_tenant_id",
    "require_any_perm",
    "require_perm",
    "require_role",
]

log = get_logger(__name__)

DEV_TENANT_ID = "tnt_dev"
DEV_PRINCIPAL_ID = "usr_dev"
_BEARER_PREFIX = "bearer "

_ROLE_RANK: dict[Role, int] = {
    Role.VIEWER: 0,
    Role.SERVICE: 10,
    Role.DEVELOPER: 20,
    Role.ADMIN: 30,
    Role.OWNER: 40,
}


# --------------------------------------------------------------------------- 配置
def get_settings_dep() -> Settings:
    """注入全局配置（进程内 lru_cache，零开销）。"""
    return get_settings()


CurrentSettings = Annotated[Settings, Depends(get_settings_dep)]


def get_request_id_dep() -> str:
    """当前请求 ID。由 :class:`RequestIdMiddleware` 写入 contextvar。"""
    return get_request_id() or ""


# --------------------------------------------------------------------------- 数据库
async def get_db() -> AsyncIterator["AsyncSession"]:
    """注入一个请求级 ``AsyncSession``，请求结束自动提交 / 回滚 / 关闭。

    若持久化层（WS «Data & Platform»）尚未落地，则抛 ``UNAVAILABLE``，
    而不是在 import 阶段就把整个应用拖垮。
    """
    try:
        from omnistream.db.session import session_scope
    except Exception as exc:  # pragma: no cover - 取决于可选依赖是否安装
        raise OmniError(
            "persistence layer is not available",
            code=ErrorCode.UNAVAILABLE,
            detail={"hint": "install requirements.txt and run `make migrate`"},
            cause=exc,
        ) from exc

    async with session_scope() as session:
        yield session


DbSession = Annotated["AsyncSession", Depends(get_db)]


# --------------------------------------------------------------------------- 鉴权
def _dev_principal(tenant_id: str = DEV_TENANT_ID) -> Principal:
    """本地开发主体：OWNER + ADMIN_ALL，仅在 ``env=dev`` 未带凭证时使用。"""
    return Principal(
        id=DEV_PRINCIPAL_ID,
        tenant_id=tenant_id,
        kind="user",
        role=Role.OWNER,
        scopes=[Permission.ADMIN_ALL],
        email="dev@localhost",
    )


def _extract_credential(
    authorization: str | None,
    api_key_header: str | None,
) -> tuple[str, str] | None:
    """归一化凭证。返回 ``(kind, raw)``，``kind`` ∈ {``jwt``, ``apikey``}。"""
    if api_key_header:
        return ("apikey", api_key_header.strip())
    if not authorization:
        return None
    raw = authorization.strip()
    if raw.lower().startswith(_BEARER_PREFIX):
        token = raw[len(_BEARER_PREFIX):].strip()
        if not token:
            return None
        # API Key 与 JWT 共用 Authorization 头，用前缀区分：oms_live_xxx 是 API Key。
        if token.startswith("oms_"):
            return ("apikey", token)
        return ("jwt", token)
    if raw.startswith("oms_"):
        return ("apikey", raw)
    return None


async def _resolve_principal(
    kind: str,
    raw: str,
    settings: Settings,
) -> Principal | None:
    """把凭证换成 :class:`Principal`。鉴权实现缺席时返回 ``None``。"""
    try:
        if kind == "jwt":
            from omnistream.platform.auth.jwt import decode_access_token
            from omnistream.platform.auth.principal import principal_from_claims

            claims = decode_access_token(raw, settings.security)
            return principal_from_claims(claims)

        from omnistream.platform.auth.api_key import verify_api_key

        return await verify_api_key(raw, settings.security)
    except OmniError:
        raise
    except ImportError as exc:
        log.warning("auth_backend_missing", kind=kind, error=str(exc))
        return None
    except Exception as exc:  # 后端实现内部异常 → 统一 401，避免泄露细节
        raise AuthError(
            "credential verification failed",
            code=ErrorCode.UNAUTHENTICATED,
            detail={"kind": kind},
            cause=exc,
        ) from exc


async def get_optional_principal(
    request: Request,
    settings: CurrentSettings,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    x_api_key: Annotated[str | None, Header(alias="X-Api-Key")] = None,
) -> Principal | None:
    """尽力解析调用主体；无凭证时返回 ``None``（不抛异常）。

    公开路径（``/healthz``、``/docs`` 等）与匿名可访问的接口用这个依赖。
    """
    cached = getattr(request.state, "principal", None)
    if isinstance(cached, Principal):
        return cached

    credential = _extract_credential(authorization, x_api_key)
    if credential is None:
        if settings.env == "dev":
            principal = _dev_principal()
            request.state.principal = principal
            return principal
        return None

    principal = await _resolve_principal(credential[0], credential[1], settings)
    if principal is None and settings.env == "dev":
        principal = _dev_principal()
    if principal is not None:
        request.state.principal = principal
        set_tenant_id(principal.tenant_id)
    return principal


async def get_principal(
    principal: Annotated[Principal | None, Depends(get_optional_principal)],
) -> Principal:
    """强制鉴权版本。未认证直接 ``401 OMS-2000``。"""
    if principal is None:
        raise AuthError(
            "authentication required",
            code=ErrorCode.UNAUTHENTICATED,
            detail={"hint": "send `Authorization: Bearer <jwt>` or `X-Api-Key: oms_live_...`"},
        )
    return principal


CurrentPrincipal = Annotated[Principal, Depends(get_principal)]


async def get_tenant_id(principal: CurrentPrincipal) -> str:
    """当前租户 ID，并写入 contextvar 供日志/审计自动带上。"""
    set_tenant_id(principal.tenant_id)
    return principal.tenant_id


CurrentTenantId = Annotated[str, Depends(get_tenant_id)]


# --------------------------------------------------------------------------- 权限
def require_perm(*perms: Permission) -> Callable[..., Any]:
    """生成一个「必须同时具备全部 ``perms``」的依赖。

    ::

        @router.delete("/sessions/{sid}",
                       dependencies=[Depends(require_perm(Permission.SESSION_WRITE))])
        async def delete_session(...): ...
    """

    async def _dependency(principal: CurrentPrincipal) -> Principal:
        if not principal.has_all(*perms):
            raise PermissionError_(
                "insufficient permission",
                code=ErrorCode.FORBIDDEN,
                detail={
                    "required": [str(p) for p in perms],
                    "granted": sorted(str(p) for p in principal.permissions),
                },
            )
        return principal

    _dependency.__name__ = "require_perm_" + "_".join(p.name.lower() for p in perms)
    return _dependency


def require_any_perm(*perms: Permission) -> Callable[..., Any]:
    """生成一个「具备任意一个 ``perms`` 即可」的依赖。"""

    async def _dependency(principal: CurrentPrincipal) -> Principal:
        if not any(principal.has(p) for p in perms):
            raise PermissionError_(
                "insufficient permission",
                code=ErrorCode.FORBIDDEN,
                detail={"required_any": [str(p) for p in perms]},
            )
        return principal

    _dependency.__name__ = "require_any_perm_" + "_".join(p.name.lower() for p in perms)
    return _dependency


def require_role(minimum: Role) -> Callable[..., Any]:
    """生成一个「角色不低于 ``minimum``」的依赖（按 VIEWER<SERVICE<DEVELOPER<ADMIN<OWNER）。"""

    threshold = _ROLE_RANK.get(minimum, 0)

    async def _dependency(principal: CurrentPrincipal) -> Principal:
        if _ROLE_RANK.get(principal.role, 0) < threshold:
            raise PermissionError_(
                "role too low",
                code=ErrorCode.FORBIDDEN,
                detail={"required_role": str(minimum), "actual_role": str(principal.role)},
            )
        return principal

    _dependency.__name__ = f"require_role_{minimum.name.lower()}"
    return _dependency


# --------------------------------------------------------------------------- 分页
@dataclass(slots=True)
class PaginationParams:
    """归一化后的分页参数。``cursor`` 与 ``offset`` 二选一，游标优先。"""

    limit: int = DEFAULT_PAGE_LIMIT
    offset: int = 0
    cursor: str | None = None
    order: str = "desc"

    @property
    def use_cursor(self) -> bool:
        return bool(self.cursor)

    @property
    def descending(self) -> bool:
        return self.order == "desc"


async def get_pagination(
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT)] = DEFAULT_PAGE_LIMIT,
    offset: Annotated[int, Query(ge=0)] = 0,
    cursor: Annotated[str | None, Query(max_length=512)] = None,
    order: Annotated[str, Query(pattern="^(asc|desc)$")] = "desc",
) -> PaginationParams:
    """解析并校验列表接口的通用分页参数。"""
    return PaginationParams(limit=limit, offset=offset, cursor=cursor, order=order)


Pagination = Annotated[PaginationParams, Depends(get_pagination)]

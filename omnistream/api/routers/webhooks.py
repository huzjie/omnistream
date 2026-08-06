"""Webhook 路由：订阅端点管理 + 连通性探测。

订阅端点存于租户设置 ``tenants.settings['webhooks']['endpoints']``（见
:data:`omnistream.db.models.tenant.DEFAULT_TENANT_SETTINGS`）。写操作需要 ``tenant:write``。

端点结构：``{"id", "url", "secret", "events": [...], "active": true}``。
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from omnistream.api.deps import CurrentPrincipal, DbSession, require_perm
from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import NotFoundError
from omnistream.db.models.tenant import Tenant
from omnistream.platform.webhooks import PlatformEvent, default_dispatcher, make_event
from omnistream.schemas.auth import Permission
from omnistream.schemas.common import Ok

router = APIRouter()
ROUTER_ORDER = 90
ROUTER_TAGS = ["webhooks"]


class WebhookEndpointCreate(BaseModel):
    url: str
    secret: str = ""
    events: list[str] = []
    active: bool = True


class WebhookEndpointUpdate(BaseModel):
    url: str | None = None
    secret: str | None = None
    events: list[str] | None = None
    active: bool | None = None


def _mask_secret(secret: str) -> str:
    if not secret:
        return ""
    return secret[:4] + "*" * max(0, len(secret) - 8) + secret[-4:] if len(secret) > 8 else "*" * len(secret)


def _load_endpoints(tenant: Tenant) -> list[dict[str, Any]]:
    raw = tenant.setting_path("webhooks.endpoints")
    return list(raw) if isinstance(raw, list) else []


def _public(endpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": endpoint.get("id"),
        "url": endpoint.get("url"),
        "secret_masked": _mask_secret(str(endpoint.get("secret") or "")),
        "events": endpoint.get("events", []),
        "active": endpoint.get("active", True),
    }


@router.get("", response_model=Ok[list[dict]])
async def list_endpoints(
    principal: Annotated[Any, Depends(require_perm(Permission.TENANT_READ))],
    session: DbSession,
) -> Ok[list[dict]]:
    tenant = await session.get(Tenant, principal.tenant_id)
    if tenant is None:
        return Ok.of([])
    return Ok.of([_public(ep) for ep in _load_endpoints(tenant)])


@router.post("", response_model=Ok[dict])
async def create_endpoint(
    body: WebhookEndpointCreate,
    principal: Annotated[Any, Depends(require_perm(Permission.TENANT_WRITE))],
    session: DbSession,
) -> Ok[dict]:
    import uuid

    tenant = await session.get(Tenant, principal.tenant_id)
    if tenant is None:
        raise NotFoundError("租户不存在", code=ErrorCode.NOT_FOUND)
    endpoints = _load_endpoints(tenant)
    endpoint = {
        "id": f"wh_{uuid.uuid4().hex[:12]}",
        "url": body.url,
        "secret": body.secret,
        "events": list(body.events),
        "active": body.active,
    }
    endpoints.append(endpoint)
    tenant.set_setting_path("webhooks.endpoints", endpoints)
    await session.flush()
    return Ok.of(_public(endpoint))


@router.patch("/{endpoint_id}", response_model=Ok[dict])
async def update_endpoint(
    endpoint_id: str,
    body: WebhookEndpointUpdate,
    principal: Annotated[Any, Depends(require_perm(Permission.TENANT_WRITE))],
    session: DbSession,
) -> Ok[dict]:
    tenant = await session.get(Tenant, principal.tenant_id)
    if tenant is None:
        raise NotFoundError("租户不存在", code=ErrorCode.NOT_FOUND)
    endpoints = _load_endpoints(tenant)
    for ep in endpoints:
        if ep.get("id") == endpoint_id:
            if body.url is not None:
                ep["url"] = body.url
            if body.secret is not None:
                ep["secret"] = body.secret
            if body.events is not None:
                ep["events"] = list(body.events)
            if body.active is not None:
                ep["active"] = body.active
            tenant.set_setting_path("webhooks.endpoints", endpoints)
            await session.flush()
            return Ok.of(_public(ep))
    raise NotFoundError("订阅端点不存在", code=ErrorCode.NOT_FOUND)


@router.delete("/{endpoint_id}", response_model=Ok[None])
async def delete_endpoint(
    endpoint_id: str,
    principal: Annotated[Any, Depends(require_perm(Permission.TENANT_WRITE))],
    session: DbSession,
) -> Ok[None]:
    tenant = await session.get(Tenant, principal.tenant_id)
    if tenant is None:
        raise NotFoundError("租户不存在", code=ErrorCode.NOT_FOUND)
    endpoints = [ep for ep in _load_endpoints(tenant) if ep.get("id") != endpoint_id]
    tenant.set_setting_path("webhooks.endpoints", endpoints)
    await session.flush()
    return Ok.of(None)


@router.post("/ping", response_model=Ok[dict])
async def ping(
    principal: Annotated[Any, Depends(require_perm(Permission.TENANT_WRITE))],
    session: DbSession,
) -> Ok[dict]:
    """向当前租户所有订阅端点发送一次 ``webhook.ping`` 探测。"""
    event = make_event(PlatformEvent.WEBHOOK_PING, tenant_id=principal.tenant_id, payload={"ping": True})
    results = await default_dispatcher.dispatch(session, event)
    return Ok.of({"delivered": results})

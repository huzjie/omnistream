"""Webhook 事件订阅与投递。

特性：

* **HMAC 签名**：每个请求带 ``X-Omnistream-Signature: sha256=<hmac>``，订阅方可验真；
* **指数退避重试**：最多 ``max_attempts`` 次，间隔 ``base × 2^(n-1)``；
* **死信**：超过重试上限的事件进入进程内死信队列，可经 ``get_dead_letters`` 排查；
* **零硬依赖**：优先用 ``httpx``（若安装），否则退回标准库 ``urllib``（线程池执行）。

订阅来源：租户设置 ``tenants.settings['webhooks']['endpoints']``（见 :data:`DEFAULT_TENANT_SETTINGS`），
每个端点形如 ``{"url", "secret", "events": [...], "active": true}``。
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import ssl
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from omnistream.db.models.tenant import Tenant
from omnistream.platform.webhooks.events import WebhookEvent

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = ["WebhookDispatcher", "WebhookSubscription", "load_subscriptions"]

try:  # pragma: no cover - 取决于是否安装 httpx
    import httpx

    HTTPX_AVAILABLE = True
except Exception:  # noqa: BLE001
    httpx = None  # type: ignore[assignment]
    HTTPX_AVAILABLE = False


@dataclass(slots=True)
class WebhookSubscription:
    """一个 Webhook 订阅端点。"""

    id: str
    tenant_id: str
    url: str
    secret: str
    events: frozenset[str] = field(default_factory=frozenset)
    active: bool = True


async def load_subscriptions(
    session: "AsyncSession", tenant_id: str
) -> list[WebhookSubscription]:
    """从租户设置加载 Webhook 订阅端点。"""
    tenant = (
        await session.execute(select(Tenant).where(Tenant.id == tenant_id))
    ).scalar_one_or_none()
    subs: list[WebhookSubscription] = []
    if tenant is None:
        return subs
    endpoints = tenant.setting_path("webhooks.endpoints") or []
    if not isinstance(endpoints, list):
        return subs
    for index, ep in enumerate(endpoints):
        if not isinstance(ep, dict):
            continue
        url = ep.get("url")
        if not url:
            continue
        subs.append(
            WebhookSubscription(
                id=str(ep.get("id") or f"{tenant_id}-wh-{index}"),
                tenant_id=tenant_id,
                url=str(url),
                secret=str(ep.get("secret") or ""),
                events=frozenset(str(e) for e in (ep.get("events") or [])),
                active=bool(ep.get("active", True)),
            )
        )
    return subs


class WebhookDispatcher:
    """Webhook 投递器（单例使用）。"""

    def __init__(
        self,
        *,
        max_attempts: int = 5,
        base_backoff_s: float = 1.0,
        timeout_s: float = 10.0,
    ) -> None:
        self.max_attempts = max_attempts
        self.base_backoff_s = base_backoff_s
        self.timeout_s = timeout_s
        self._dead_letters: list[WebhookEvent] = []
        self._lock: Any = None

    def _sign(self, body: bytes, secret: str) -> str:
        mac = hmac.new(secret.encode("utf-8"), body, hashlib.sha256)
        return mac.hexdigest()

    async def dispatch(
        self, session: "AsyncSession", event: WebhookEvent
    ) -> dict[str, bool]:
        """投递事件到该租户所有匹配且启用的订阅方。

        返回 ``{subscription_id: 是否成功}``。
        """
        subs = await load_subscriptions(session, event.tenant_id)
        results: dict[str, bool] = {}
        for sub in subs:
            if not sub.active:
                continue
            if sub.events and event.type not in sub.events:
                continue
            ok = await self._deliver(sub, event)
            results[sub.id] = ok
            if not ok:
                self._dead_letters.append(event)
        return results

    async def _deliver(self, sub: WebhookSubscription, event: WebhookEvent) -> bool:
        body = json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        signature = self._sign(body, sub.secret)
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "omnistream-webhook/1.0",
            "X-Omnistream-Event": event.type,
            "X-Omnistream-Delivery": event.id,
            "X-Omnistream-Signature": f"sha256={signature}",
        }
        for attempt in range(1, self.max_attempts + 1):
            event.attempt = attempt
            try:
                await self._post(sub.url, body, headers, self.timeout_s)
                return True
            except Exception:  # noqa: BLE001 - 投递失败按退避重试
                if attempt < self.max_attempts:
                    await asyncio.sleep(self.base_backoff_s * (2 ** (attempt - 1)))
        return False

    async def _post(
        self, url: str, body: bytes, headers: dict[str, str], timeout_s: float
    ) -> None:
        if HTTPX_AVAILABLE and httpx is not None:
            async with httpx.AsyncClient(timeout=timeout_s, verify=ssl.create_default_context()) as client:
                resp = await client.post(url, content=body, headers=headers)
                resp.raise_for_status()
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._post_sync, url, body, headers, timeout_s)

    @staticmethod
    def _post_sync(url: str, body: bytes, headers: dict[str, str], timeout_s: float) -> None:
        import urllib.request

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 - 订阅方 URL 由租户配置
            if resp.status >= 400:
                raise RuntimeError(f"webhook 投递失败: HTTP {resp.status}")

    def get_dead_letters(self) -> list[WebhookEvent]:
        """返回本进程内累积的死信事件（超过重试上限）。"""
        return list(self._dead_letters)


#: 进程内默认投递器（多实例部署时各实例独立，生产建议用共享队列 / Redis）
default_dispatcher = WebhookDispatcher()

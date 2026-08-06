"""可订阅平台事件枚举与 payload 定义。

Webhook 投递的是这些事件的 JSON 快照。事件类型与治理 / 配额 / 计费动作对齐，
订阅方按 ``type`` 字段路由处理。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from omnistream.core.clock import utcnow
from omnistream.core.ids import new_event_id

__all__ = ["PlatformEvent", "WebhookEvent", "make_event"]


class PlatformEvent(StrEnum):
    """平台可投递给 Webhook 订阅方的事件类型。"""

    SESSION_CREATED = "session.created"
    SESSION_ENDED = "session.ended"
    RECORDING_READY = "recording.ready"
    TRANSCRIPT_DONE = "transcript.done"
    QUALITY_SCORED = "quality.scored"
    BILLING_PERIOD_CLOSED = "billing.period_closed"
    API_KEY_CREATED = "apikey.created"
    API_KEY_REVOKED = "apikey.revoked"
    TENANT_CREATED = "tenant.created"
    QUOTA_EXCEEDED = "quota.exceeded"
    WEBHOOK_PING = "webhook.ping"


@dataclass(slots=True)
class WebhookEvent:
    """一次要投递的事件。"""

    id: str = field(default_factory=new_event_id)
    type: str = ""
    tenant_id: str = ""
    created_at: datetime = field(default_factory=utcnow)
    payload: dict[str, Any] = field(default_factory=dict)
    attempt: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.isoformat(),
            "attempt": self.attempt,
            "payload": self.payload,
        }


def make_event(
    type: "PlatformEvent | str",
    *,
    tenant_id: str,
    payload: dict[str, Any] | None = None,
) -> WebhookEvent:
    """构造一个事件对象。"""
    return WebhookEvent(
        type=str(type),
        tenant_id=tenant_id,
        payload=dict(payload or {}),
    )

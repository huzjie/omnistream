"""审计与合规领域模型：操作审计事件（哈希链）与 PII 脱敏报告。"""
from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum
from typing import Any

import orjson
from pydantic import Field

from omnistream.core.clock import utcnow
from omnistream.schemas.base import OmniModel


class AuditAction(StrEnum):
    """可审计动作。新增动作必须在此登记，禁止使用裸字符串。"""

    TENANT_CREATE = "tenant.create"
    TENANT_UPDATE = "tenant.update"
    TENANT_SUSPEND = "tenant.suspend"

    USER_LOGIN = "user.login"
    USER_LOGIN_FAILED = "user.login_failed"
    USER_LOGOUT = "user.logout"
    USER_CREATE = "user.create"
    USER_UPDATE = "user.update"
    USER_DELETE = "user.delete"

    APIKEY_CREATE = "apikey.create"
    APIKEY_REVOKE = "apikey.revoke"

    PROVIDER_CONFIGURE = "provider.configure"
    PROVIDER_DISABLE = "provider.disable"

    SESSION_CREATE = "session.create"
    SESSION_END = "session.end"

    MEMORY_WRITE = "memory.write"
    MEMORY_DELETE = "memory.delete"
    MEMORY_EXPORT = "memory.export"

    RECORDING_ACCESS = "recording.access"
    RECORDING_DELETE = "recording.delete"
    RECORDING_EXPORT = "recording.export"

    QUOTA_UPDATE = "quota.update"
    SETTINGS_UPDATE = "settings.update"
    DATA_EXPORT = "data.export"
    DATA_PURGE = "data.purge"


class AuditEvent(OmniModel):
    """审计事件。``prev_hash`` + ``hash`` 构成防篡改哈希链。"""

    id: str = ""
    tenant_id: str = ""
    actor: str = Field(default="system", description="用户 ID / api_key_id / system")
    actor_kind: str = Field(default="system", description="user|apikey|system")
    action: AuditAction = AuditAction.SETTINGS_UPDATE
    resource: str = Field(default="", description="被操作资源标识，如 sess_01H...")
    resource_type: str = ""
    ip: str | None = None
    user_agent: str | None = None
    success: bool = True
    diff: dict[str, Any] = Field(default_factory=dict, description="变更前后差异（已脱敏）")
    created_at: datetime = Field(default_factory=utcnow)
    prev_hash: str = ""
    hash: str = ""

    def canonical_bytes(self) -> bytes:
        """参与哈希计算的规范化字节串（字段有序、排除 hash 自身）。"""
        payload = {
            "tenant_id": self.tenant_id,
            "actor": self.actor,
            "actor_kind": self.actor_kind,
            "action": str(self.action),
            "resource": self.resource,
            "resource_type": self.resource_type,
            "ip": self.ip or "",
            "success": self.success,
            "diff": self.diff,
            "created_at": self.created_at.isoformat(),
            "prev_hash": self.prev_hash,
        }
        return orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)

    def compute_hash(self) -> str:
        return hashlib.blake2b(self.canonical_bytes(), digest_size=16).hexdigest()

    def seal(self, prev_hash: str = "") -> "AuditEvent":
        """接入哈希链：设置 prev_hash 并计算自身 hash。"""
        self.prev_hash = prev_hash
        self.hash = self.compute_hash()
        return self

    def verify(self) -> bool:
        return bool(self.hash) and self.hash == self.compute_hash()


def verify_chain(events: list[AuditEvent]) -> tuple[bool, int]:
    """校验一段审计链。返回 ``(是否完好, 首个断裂位置)``，完好时位置为 -1。"""
    prev = ""
    for i, ev in enumerate(events):
        if ev.prev_hash != prev or not ev.verify():
            return False, i
        prev = ev.hash
    return True, -1


class PIIMatch(OmniModel):
    """一处 PII 命中。"""

    kind: str = Field(description="phone|email|id_card|bank_card|address|name|custom")
    start: int = 0
    end: int = 0
    original_hash: str = Field(default="", description="原文 blake2b 短哈希，便于去重统计")
    replacement: str = ""


class PIIRedactionReport(OmniModel):
    """一次脱敏处理的报告，写入审计日志。"""

    target: str = Field(default="", description="被处理对象，如 turn_01H... / recording_01H...")
    target_type: str = Field(default="transcript", description="transcript|memory|log|recording")
    matches: list[PIIMatch] = Field(default_factory=list)
    total_chars: int = 0
    redacted_chars: int = 0
    engine: str = "regex"
    elapsed_ms: float = 0.0
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def count(self) -> int:
        return len(self.matches)

    @property
    def by_kind(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for m in self.matches:
            out[m.kind] = out.get(m.kind, 0) + 1
        return out

    @property
    def redaction_ratio(self) -> float:
        return self.redacted_chars / self.total_chars if self.total_chars else 0.0

"""会话治理：合规审计日志。

记录会话生命周期关键事件（创建、鉴权、工具调用、PII 访问、终止），
满足企业合规审计需求。输出为结构化日志 + 可选文件。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("omnistream.audit")


class AuditLogger:
    """审计日志器。"""

    def __init__(self, enabled: bool = True, sink: str = "log") -> None:
        self.enabled = enabled
        self.sink = sink  # log | file | both
        self._file = None
        if sink in ("file", "both"):
            try:
                self._file = open("data/audit/audit.jsonl", "a", encoding="utf-8")
            except OSError:
                self._file = None

    def record(self, event: str, *, actor: str, tenant: str,
               session_id: str | None = None, **details: Any) -> None:
        if not self.enabled:
            return
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "actor": actor,
            "tenant": tenant,
            "session_id": session_id,
            **details,
        }
        line = json.dumps(entry, ensure_ascii=False, default=str)
        if self.sink in ("log", "both"):
            logger.info("AUDIT %s", line)
        if self._file is not None:
            self._file.write(line + "\n")
            self._file.flush()

    def session_created(self, actor: str, tenant: str,
                        session_id: str, provider: str) -> None:
        self.record("session.created", actor=actor, tenant=tenant,
                    session_id=session_id, provider=provider)

    def session_ended(self, actor: str, tenant: str,
                      session_id: str, reason: str) -> None:
        self.record("session.ended", actor=actor, tenant=tenant,
                    session_id=session_id, reason=reason)

    def tool_called(self, actor: str, tenant: str, session_id: str,
                    tool: str, args: dict[str, Any]) -> None:
        self.record("tool.called", actor=actor, tenant=tenant,
                    session_id=session_id, tool=tool, args=args)

    def pii_accessed(self, actor: str, tenant: str, session_id: str,
                     fields: list[str]) -> None:
        self.record("pii.accessed", actor=actor, tenant=tenant,
                    session_id=session_id, fields=fields)

    def close(self) -> None:
        if self._file:
            self._file.close()

"""会话录制管理。

录制会话的音视频 + 事件流，支持回放。文件按会话 ID 组织：
    data/recordings/<session_id>/audio.opus
    data/recordings/<session_id>/events.jsonl
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from omnistream.core.ids import new_recording_id

logger = logging.getLogger(__name__)


class RecordingManager:
    """录制管理器。"""

    def __init__(self, root: str = "data/recordings", enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        d = self.root / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def start_session(self, session_id: str, meta: dict[str, Any] | None = None) -> Path:
        d = self._session_dir(session_id)
        events = d / "events.jsonl"
        with events.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "session.start",
                                "session_id": session_id,
                                "meta": meta or {},
                                "ts": json_now()}, ensure_ascii=False) + "\n")
        return d

    def append_event(self, session_id: str, event: dict[str, Any]) -> Path | None:
        if not self.enabled:
            return None
        d = self._session_dir(session_id)
        events = d / "events.jsonl"
        with events.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        return events

    def append_audio(self, session_id: str, chunk: bytes,
                     encoding: str = "pcm16") -> None:
        if not self.enabled:
            return
        d = self._session_dir(session_id)
        (d / "audio.pcm").abspath  # noqa: B018 - 目录已存在
        with (d / f"audio.{encoding}").open("ab") as f:
            f.write(chunk)

    def list_sessions(self) -> list[str]:
        if not self.enabled or not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def events_of(self, session_id: str) -> list[dict[str, Any]]:
        events_file = self.root / session_id / "events.jsonl"
        if not events_file.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in events_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out


def json_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()

"""转写归档。

将实时转写（ASR 结果）落盘为结构化文档，支持按会话检索。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TranscriptArchiver:
    """转写归档器。"""

    def __init__(self, root: str = "data/transcripts") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, session_id: str, transcript: list[dict[str, Any]],
             meta: dict[str, Any] | None = None) -> Path:
        doc = {
            "session_id": session_id,
            "meta": meta or {},
            "entries": transcript,
        }
        path = self.root / f"{session_id}.json"
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load(self, session_id: str) -> dict[str, Any] | None:
        path = self.root / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("transcript %s 损坏", session_id)
            return None

    def list(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))

    def to_markdown(self, session_id: str) -> str:
        doc = self.load(session_id) or {}
        lines = [f"# 会话转写 {session_id}", ""]
        for e in doc.get("entries", []):
            role = e.get("role", "?")
            text = e.get("text", "")
            lines.append(f"**{role}**: {text}")
            lines.append("")
        return "\n".join(lines)

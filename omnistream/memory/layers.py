"""记忆分层与存储。

记忆分层（对齐 Meta 论文）：
- session          会话级：当前会话内的临时上下文，会话结束即弃
- private_state    私有状态：用户偏好、身份、私有事实（不进共享上下文）
- stable_knowledge 稳定知识：可长期复用的领域知识
- procedural       程序性记忆：怎么做事的流程/规则

存储层：内存 dict（进程内），支持可选的 SQLite 持久化。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


class MemoryLayerKind(StrEnum):
    SESSION = "session"
    PRIVATE_STATE = "private_state"
    STABLE_KNOWLEDGE = "stable_knowledge"
    PROCEDURAL = "procedural"


LAYER_PRIORITY: dict[MemoryLayerKind, int] = {
    MemoryLayerKind.SESSION: 0,
    MemoryLayerKind.PRIVATE_STATE: 1,
    MemoryLayerKind.STABLE_KNOWLEDGE: 2,
    MemoryLayerKind.PROCEDURAL: 3,
}


@dataclass(slots=True)
class MemoryItem:
    """一条记忆。"""
    id: str
    role: str            # user | assistant | system
    text: str
    layer: MemoryLayerKind = MemoryLayerKind.STABLE_KNOWLEDGE
    confidence: float = 0.9
    metadata: dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""
    created_ms: int = 0
    updated_ms: int = 0
    ttl_s: int | None = None   # None = 永不过期
    reinforced: int = 0

    @property
    def expired(self, now_ms: int = 0) -> bool:
        if self.ttl_s is None:
            return False
        from omnistream.core.clock import now_ms as _now

        return (now_ms or _now()) - self.created_ms > self.ttl_s * 1000


class MemoryStore:
    """内存 + 可选 SQLite 持久化。"""

    def __init__(self, sqlite_path: str | None = None) -> None:
        self._items: dict[str, MemoryItem] = {}
        self._sqlite_path = sqlite_path
        if sqlite_path:
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        try:
            import sqlite3

            conn = sqlite3.connect(self._sqlite_path)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS memory (id TEXT PRIMARY KEY, role TEXT,"
                " text TEXT, layer TEXT, confidence REAL, metadata TEXT,"
                " fingerprint TEXT, created_ms INTEGER, updated_ms INTEGER,"
                " ttl_s INTEGER, reinforced INTEGER)")
            conn.commit()
            conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("sqlite memory init failed, fallback memory-only: %s", exc)
            self._sqlite_path = None

    async def put(self, item: MemoryItem) -> None:
        old = self._items.get(item.id)
        if old:
            item.reinforced = old.reinforced
        self._items[item.id] = item
        if self._sqlite_path:
            await self._persist(item)

    async def _persist(self, item: MemoryItem) -> None:
        try:
            import json
            import sqlite3

            conn = sqlite3.connect(self._sqlite_path)
            conn.execute(
                "INSERT OR REPLACE INTO memory VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (item.id, item.role, item.text, item.layer.value, item.confidence,
                 json.dumps(item.metadata, ensure_ascii=False), item.fingerprint,
                 item.created_ms, item.updated_ms, item.ttl_s, item.reinforced))
            conn.commit()
            conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("memory persist failed: %s", exc)

    async def reinforce(self, item_id: str, delta: float = 0.02) -> None:
        item = self._items.get(item_id)
        if item:
            item.confidence = min(1.0, item.confidence + delta)
            item.reinforced += 1
            item.updated_ms = now_ms_import()
            if self._sqlite_path:
                await self._persist(item)

    async def forget_expired(self, now: int) -> int:
        expired = [i for i in self._items.values() if i.expired]
        for item in expired:
            self._items.pop(item.id, None)
        if expired and self._sqlite_path:
            await self._purge([i.id for i in expired])
        return len(expired)

    async def forget_low_confidence(self, min_confidence: float) -> int:
        stale = [i for i in self._items.values()
                 if i.confidence < min_confidence and i.reinforced == 0]
        for item in stale:
            self._items.pop(item.id, None)
        if stale and self._sqlite_path:
            await self._purge([i.id for i in stale])
        return len(stale)

    async def compact_duplicates(self) -> int:
        """压缩重复记忆（同 fingerprint 保留最新最高置信）。"""
        latest: dict[str, MemoryItem] = {}
        removed = 0
        for item in self._items.values():
            key = item.fingerprint or item.text.strip()
            old = latest.get(key)
            if old is None:
                latest[key] = item
            elif item.updated_ms >= old.updated_ms:
                latest[key] = item
                removed += 1
            else:
                removed += 1
        self._items = latest
        return removed

    async def _purge(self, ids: list[str]) -> None:
        try:
            import sqlite3

            conn = sqlite3.connect(self._sqlite_path)
            for iid in ids:
                conn.execute("DELETE FROM memory WHERE id=?", (iid,))
            conn.commit()
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    async def all(self, layer: MemoryLayerKind | str | None = None) -> list[MemoryItem]:
        if layer is None:
            return sorted(self._items.values(), key=lambda i: i.updated_ms)
        lk = MemoryLayerKind(layer)
        return [i for i in self._items.values() if i.layer == lk]

    async def clear(self) -> None:
        self._items.clear()
        if self._sqlite_path:
            try:
                import sqlite3

                conn = sqlite3.connect(self._sqlite_path)
                conn.execute("DELETE FROM memory")
                conn.commit()
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._items

    def __iter__(self) -> AsyncIterator[MemoryItem]:
        async def gen():
            for item in self._items.values():
                yield item

        return gen()


def now_ms_import() -> int:
    from omnistream.core.clock import now_ms

    return now_ms()

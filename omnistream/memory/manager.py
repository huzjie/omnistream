"""主动记忆治理层（Proactive Memory Governance）。

对标 Meta 2026-08 研究「独立记忆管理 Agent」：
- 记忆分层：session（会话）/ private_state（私有状态）/ stable_knowledge（稳定知识）/ procedural（程序性记忆）
- 选择性注入：三闸门（relevance / confidence / timing）只注入"一条定向提醒"，或保持沉默
- 置信度校准：避免对推测性推断过度自信
- 遗忘与压缩：阈值淘汰 + 定期压缩

管理逻辑与行动 Agent（对话编排器）解耦，通过注入接口协作。
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Any

from omnistream.core.clock import now_ms
from omnistream.core.ids import new_memory_id
from omnistream.memory.layers import (
    MemoryItem,
    MemoryLayerKind,
    MemoryStore,
)
from omnistream.memory.retrieval import VectorRetriever
from omnistream.memory.gates import InjectionDecision, InjectionGate

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MemoryStats:
    items: int = 0
    injected: int = 0
    silent: int = 0
    forgotten: int = 0
    compressed: int = 0
    last_decision: InjectionDecision | None = None


class MemoryManager:
    """主动记忆管理器。

    生命周期：::

        mm = MemoryManager(settings)
        await mm.ingest(role="user", text="我叫小明，喜欢编程", layer="stable_knowledge")
        decision = await mm.decide_injection(context="用户问起名字")
        if decision.inject:
            await provider.request_response(instructions_suffix=decision.reminder)
    """

    def __init__(self, settings: Any = None) -> None:
        self.store = MemoryStore()
        self.retriever = VectorRetriever(backend=getattr(settings, "vector_backend", "numpy"))
        self.gate = InjectionGate(
            relevance_threshold=getattr(settings, "relevance_threshold", 0.55),
            confidence_threshold=getattr(settings, "confidence_threshold", 0.60),
            max_injections_per_turn=getattr(settings, "max_injections_per_turn", 1),
            max_injections_per_session=getattr(settings, "max_injections_per_session", 6),
        )
        self.enabled: bool = getattr(settings, "enabled", True)
        self._injected_this_session = 0
        self._lock = asyncio.Lock()
        self.stats = MemoryStats()

    # ---------------- 写入 ----------------

    async def ingest(
        self,
        *,
        role: str,
        text: str,
        layer: MemoryLayerKind | str = "stable_knowledge",
        confidence: float = 0.9,
        metadata: dict[str, Any] | None = None,
        ttl_s: int | None = None,
    ) -> MemoryItem:
        """摄入一条记忆。自动去重（相同 fingerprint 合并，更新时间戳）。"""
        if not self.enabled:
            return MemoryItem(id=new_memory_id(), role=role, text=text, layer=layer)
        fp = self._fingerprint(text)
        item = MemoryItem(
            id=new_memory_id(),
            role=role,
            text=text,
            layer=MemoryLayerKind(layer),
            confidence=confidence,
            metadata=metadata or {},
            fingerprint=fp,
            created_ms=now_ms(),
            updated_ms=now_ms(),
            ttl_s=ttl_s,
        )
        async with self._lock:
            await self.store.put(item)
            self.retriever.add(item)
        logger.debug("memory ingest layer=%s id=%s", layer, item.id)
        return item

    async def ingest_batch(self, items: list[dict[str, Any]]) -> list[MemoryItem]:
        return [await self.ingest(**it) for it in items]

    # ---------------- 读取 ----------------

    async def search(self, query: str, k: int = 5,
                     layer: MemoryLayerKind | str | None = None) -> list[MemoryItem]:
        """向量检索，可选按层过滤。"""
        return await self.retriever.search(query, k=k, layer=layer)

    async def all(self, layer: MemoryLayerKind | str | None = None) -> list[MemoryItem]:
        return await self.store.all(layer=layer)

    async def clear(self) -> None:
        await self.store.clear()
        self.retriever.clear()
        self._injected_this_session = 0

    # ---------------- 注入决策（核心） ----------------

    async def decide_injection(self, context: str) -> InjectionDecision:
        """对给定上下文做注入决策：三闸门全过才注入，否则保持沉默。"""
        if not self.enabled:
            return InjectionDecision(inject=False, reason="memory_disabled")
        candidates = await self.retriever.search(context, k=8)
        decision = await self.gate.evaluate(
            context=context,
            candidates=candidates,
            injected_this_session=self._injected_this_session,
        )
        if decision.inject:
            self._injected_this_session += 1
            self.stats.injected += 1
            # 命中提升置信度（recently reinforced）
            for hit in decision.matched_items:
                await self.store.reinforce(hit.id, delta=0.02)
        else:
            self.stats.silent += 1
        self.stats.last_decision = decision
        return decision

    # ---------------- 遗忘与压缩 ----------------

    async def maintenance(self) -> dict[str, int]:
        """遗忘过期/低置信记忆，压缩冗余。定时任务调用。"""
        expired = await self.store.forget_expired(now_ms())
        low_conf = await self.store.forget_low_confidence(
            min_confidence=self.gate.confidence_threshold)
        compressed = await self.store.compact_duplicates()
        self.stats.forgotten += expired + low_conf
        self.stats.compressed += compressed
        return {"expired": expired, "low_confidence": low_conf,
                "compressed": compressed}

    def stats_report(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "items": len(self.store),
            "injected": self.stats.injected,
            "silent": self.stats.silent,
            "forgotten": self.stats.forgotten,
            "compressed": self.stats.compressed,
            "vector_backend": self.retriever.backend,
        }

    @staticmethod
    def _fingerprint(text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]

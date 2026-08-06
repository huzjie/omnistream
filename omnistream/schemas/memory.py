"""记忆领域模型：记忆条目、检索查询、命中结果、注入决策。"""
from __future__ import annotations

from datetime import datetime

from pydantic import Field

from omnistream.core.clock import utcnow
from omnistream.protocol.enums import MemoryLayer
from omnistream.schemas.base import OmniModel


class MemoryItem(OmniModel):
    id: str
    tenant_id: str
    layer: MemoryLayer
    content: str
    key: str | None = Field(default=None, description="private_state 层的键，同键写覆盖")
    subject: str = Field(default="user", description="记忆主体：user / agent / world")
    embedding: list[float] | None = Field(default=None, repr=False)
    confidence: float = 0.8
    importance: float = 0.5
    hit_count: int = 0
    last_hit_at: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = None
    source_session_id: str | None = None
    tags: list[str] = Field(default_factory=list)


class MemoryQuery(OmniModel):
    tenant_id: str
    text: str
    layers: list[MemoryLayer] = Field(default_factory=lambda: list(MemoryLayer))
    top_k: int = 8
    min_score: float = 0.0
    subject: str | None = None
    include_expired: bool = False


class MemoryHit(OmniModel):
    item: MemoryItem
    score: float
    vector_score: float = 0.0
    keyword_score: float = 0.0
    recency_boost: float = 0.0


class InjectionDecision(OmniModel):
    """记忆 Agent 的输出。should_inject=False 表示"保持沉默"（多数情况）。"""
    should_inject: bool
    item: MemoryItem | None = None
    rendered: str | None = Field(default=None, description="注入到系统提示尾部的那一句话")
    reason: str = Field(description="决策理由，写审计日志并在控制台展示")
    relevance: float = 0.0
    novelty: float = 0.0
    budget_left: int = 0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# 辅助 DTO（非契约）
# ---------------------------------------------------------------------------


class MemoryUpsert(OmniModel):
    """写入/更新一条记忆的请求体。"""

    layer: MemoryLayer = MemoryLayer.STABLE_KNOWLEDGE
    content: str
    key: str | None = None
    subject: str = "user"
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    ttl_s: int | None = Field(default=None, ge=1, description="留空表示永不过期")
    tags: list[str] = Field(default_factory=list)
    source_session_id: str | None = None


class MemoryBudget(OmniModel):
    """一次会话的注入预算，防止"话痨式"记忆污染对话。"""

    per_turn: int = 1
    per_session: int = 6
    used_this_turn: int = 0
    used_this_session: int = 0

    @property
    def turn_left(self) -> int:
        return max(0, self.per_turn - self.used_this_turn)

    @property
    def session_left(self) -> int:
        return max(0, self.per_session - self.used_this_session)

    @property
    def left(self) -> int:
        return min(self.turn_left, self.session_left)

    def consume(self, n: int = 1) -> None:
        self.used_this_turn += n
        self.used_this_session += n

    def reset_turn(self) -> None:
        self.used_this_turn = 0


class MemoryStats(OmniModel):
    """记忆子系统的运行统计。"""

    total_items: int = 0
    by_layer: dict[str, int] = Field(default_factory=dict)
    queries: int = 0
    hits: int = 0
    injections: int = 0
    silences: int = 0
    avg_query_ms: float = 0.0

    @property
    def injection_rate(self) -> float:
        total = self.injections + self.silences
        return self.injections / total if total else 0.0

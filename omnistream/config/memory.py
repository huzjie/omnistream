"""记忆子系统配置：向量后端、维度、注入预算、相关度阈值、遗忘窗口。"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from omnistream.core.constants import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_NOVELTY_THRESHOLD,
    DEFAULT_RELEVANCE_THRESHOLD,
    MAX_INJECTIONS_PER_SESSION,
    MAX_INJECTIONS_PER_TURN,
)


class MemorySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OMS_MEMORY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    enabled: bool = True

    # ------------------------------------------------------------ 存储
    store: str = Field(default="inmemory", description="inmemory|pgvector|qdrant|milvus")
    collection: str = "omnistream_memory"
    embedder: str = Field(default="hash", description="hash|bge|openai|volc")
    embedding_dim: int = Field(default=DEFAULT_EMBEDDING_DIM, ge=8)
    #: pgvector ivfflat 列表数，越大召回越准但建索引越慢
    ivfflat_lists: int = Field(default=100, ge=1)
    qdrant_url: str | None = None
    qdrant_api_key: str | None = Field(default=None, repr=False)

    # ------------------------------------------------------------ 检索
    top_k: int = Field(default=8, ge=1, le=100)
    #: 混合检索权重：向量 vs 关键词
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    #: 时间衰减半衰期（天），越小越偏好新记忆
    recency_half_life_days: float = Field(default=30.0, gt=0.0)
    recency_boost_max: float = Field(default=0.15, ge=0.0, le=1.0)
    query_timeout_ms: int = Field(default=120, ge=10,
                                  description="超时即放弃注入，绝不阻塞对话")

    # ------------------------------------------------------------ 注入决策
    relevance_threshold: float = Field(default=DEFAULT_RELEVANCE_THRESHOLD, ge=0.0, le=1.0)
    novelty_threshold: float = Field(default=DEFAULT_NOVELTY_THRESHOLD, ge=0.0, le=1.0)
    max_injections_per_turn: int = Field(default=MAX_INJECTIONS_PER_TURN, ge=0)
    max_injections_per_session: int = Field(default=MAX_INJECTIONS_PER_SESSION, ge=0)
    #: 注入语句最大长度，太长会打乱语音节奏
    max_injection_chars: int = Field(default=60, ge=10)
    #: 同一条记忆在同一会话内的最短重复注入间隔
    reinject_cooldown_ms: int = Field(default=120_000, ge=0)

    # ------------------------------------------------------------ 写入与遗忘
    auto_extract: bool = Field(default=True, description="轮次结束后自动抽取可记忆事实")
    extract_min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    session_layer_ttl_s: int = Field(default=86_400, ge=60)
    private_state_ttl_s: int = Field(default=0, ge=0, description="0 表示不过期")
    stable_knowledge_ttl_s: int = Field(default=0, ge=0)
    procedural_ttl_s: int = Field(default=0, ge=0)
    #: 遗忘扫描周期
    forget_scan_interval_s: int = Field(default=3600, ge=60)
    #: 单租户记忆条目上限，超出按 importance × recency 淘汰
    max_items_per_tenant: int = Field(default=100_000, ge=100)

    @property
    def normalized_weights(self) -> tuple[float, float]:
        """归一化后的 (向量权重, 关键词权重)。"""
        total = self.vector_weight + self.keyword_weight
        if total <= 0:
            return 1.0, 0.0
        return self.vector_weight / total, self.keyword_weight / total

    def ttl_for(self, layer: str) -> int:
        """按记忆层返回 TTL 秒数，0 表示不过期。"""
        mapping = {
            "session": self.session_layer_ttl_s,
            "private_state": self.private_state_ttl_s,
            "stable_knowledge": self.stable_knowledge_ttl_s,
            "procedural": self.procedural_ttl_s,
        }
        return mapping.get(layer, 0)

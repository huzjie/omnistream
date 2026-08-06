"""对话质检领域模型：评分维度、评分卡与评分结果。"""
from __future__ import annotations

from datetime import datetime

from pydantic import Field

from omnistream.core.clock import utcnow
from omnistream.schemas.base import OmniModel


class RubricDimension(OmniModel):
    """一个评分维度。"""

    key: str
    label: str = ""
    description: str = ""
    weight: float = Field(default=1.0, ge=0.0)
    max_score: float = Field(default=5.0, gt=0.0)


class QualityRubric(OmniModel):
    """评分卡。默认覆盖实时语音场景最关心的五个维度。"""

    name: str = "default"
    version: str = "1.0"
    dimensions: list[RubricDimension] = Field(default_factory=lambda: [
        RubricDimension(key="relevance", label="相关性",
                        description="回答是否切题、是否解决用户诉求", weight=1.5),
        RubricDimension(key="politeness", label="礼貌度",
                        description="语气是否得体、有无冒犯性表达", weight=1.0),
        RubricDimension(key="compliance", label="合规性",
                        description="是否触碰敏感话题、是否泄露隐私", weight=2.0),
        RubricDimension(key="interruption_grace", label="打断得体度",
                        description="被打断后是否干净停止、恢复是否自然", weight=1.2),
        RubricDimension(key="latency_feel", label="延迟体感",
                        description="停顿与首字延迟是否让人觉得卡顿", weight=1.0),
    ])

    @property
    def keys(self) -> list[str]:
        return [d.key for d in self.dimensions]

    @property
    def total_weight(self) -> float:
        return sum(d.weight for d in self.dimensions) or 1.0

    def dimension(self, key: str) -> RubricDimension | None:
        for d in self.dimensions:
            if d.key == key:
                return d
        return None

    def weighted_total(self, scores: dict[str, float]) -> float:
        """按权重归一化到 0~5 分。缺失维度按 0 计。"""
        acc = 0.0
        for d in self.dimensions:
            raw = float(scores.get(d.key, 0.0))
            acc += (raw / d.max_score) * d.weight
        return round(acc / self.total_weight * 5.0, 3)


class QualityScore(OmniModel):
    """一次质检评分结果，可针对整会话或单轮。"""

    id: str = ""
    tenant_id: str = ""
    session_id: str
    turn_id: str | None = None
    rubric: str = "default"
    rubric_version: str = "1.0"
    scores: dict[str, float] = Field(default_factory=dict)
    total: float = 0.0
    reviewer: str = Field(default="auto", description="auto|人工用户 ID")
    comment: str = ""
    flags: list[str] = Field(default_factory=list,
                             description="命中的风险标签，如 pii_leak / off_topic")
    created_at: datetime = Field(default_factory=utcnow)

    def recompute(self, rubric: QualityRubric) -> "QualityScore":
        self.total = rubric.weighted_total(self.scores)
        self.rubric = rubric.name
        self.rubric_version = rubric.version
        return self

    @property
    def passed(self) -> bool:
        """3.5 分以上且无风险标签视为通过。"""
        return self.total >= 3.5 and not self.flags


class QualityReport(OmniModel):
    """一批会话的质检汇总。"""

    tenant_id: str
    sample_size: int = 0
    avg_total: float = 0.0
    pass_rate: float = 0.0
    avg_by_dimension: dict[str, float] = Field(default_factory=dict)
    flag_counts: dict[str, int] = Field(default_factory=dict)
    worst_sessions: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utcnow)

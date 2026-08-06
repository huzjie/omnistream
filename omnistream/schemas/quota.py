"""配额与限流领域模型。"""
from __future__ import annotations

from datetime import datetime

from pydantic import Field

from omnistream.core.clock import utcnow
from omnistream.schemas.base import OmniModel


class RateLimitRule(OmniModel):
    """令牌桶限流规则。"""

    key: str = Field(description="限流维度，如 session.create / ws.connect / audio.append")
    capacity: int = Field(default=60, ge=1, description="桶容量（突发上限）")
    refill_per_second: float = Field(default=1.0, gt=0.0)
    scope: str = Field(default="tenant", description="tenant|apikey|ip|session")
    enabled: bool = True

    @property
    def per_minute(self) -> float:
        return self.refill_per_second * 60.0


class QuotaPolicy(OmniModel):
    """租户配额策略。周期内累计用量超过上限即拒绝新请求。"""

    period: str = Field(default="month", description="hour|day|month")
    max_sessions: int = Field(default=10_000, ge=0, description="0 表示不限制")
    max_audio_minutes: float = Field(default=10_000.0, ge=0.0)
    max_video_frames: int = Field(default=500_000, ge=0)
    max_input_tokens: int = Field(default=50_000_000, ge=0)
    max_output_tokens: int = Field(default=50_000_000, ge=0)
    max_tool_calls: int = Field(default=1_000_000, ge=0)
    max_concurrent_sessions: int = Field(default=20, ge=1)
    max_storage_gb: float = Field(default=100.0, ge=0.0)
    rate_limits: list[RateLimitRule] = Field(default_factory=lambda: [
        RateLimitRule(key="session.create", capacity=30, refill_per_second=0.5),
        RateLimitRule(key="ws.connect", capacity=60, refill_per_second=1.0),
        RateLimitRule(key="rest.default", capacity=300, refill_per_second=10.0),
    ])
    #: 超额行为：block 拒绝；degrade 降级到更便宜的 provider；allow 仅告警
    on_exceed: str = Field(default="block", description="block|degrade|allow")

    def rule_for(self, key: str) -> RateLimitRule | None:
        for rule in self.rate_limits:
            if rule.key == key and rule.enabled:
                return rule
        for rule in self.rate_limits:
            if rule.key == "rest.default" and rule.enabled:
                return rule
        return None


class QuotaUsage(OmniModel):
    """当前周期的累计用量。"""

    tenant_id: str
    period_start: datetime = Field(default_factory=utcnow)
    period_end: datetime | None = None
    sessions: int = 0
    audio_minutes: float = 0.0
    video_frames: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    storage_gb: float = 0.0
    concurrent_sessions: int = 0
    updated_at: datetime = Field(default_factory=utcnow)

    def ratio(self, policy: QuotaPolicy) -> dict[str, float]:
        """各维度使用率（0~1+），供控制台画进度条与阈值告警。"""
        def _r(used: float, cap: float) -> float:
            return used / cap if cap > 0 else 0.0

        return {
            "sessions": _r(self.sessions, policy.max_sessions),
            "audio_minutes": _r(self.audio_minutes, policy.max_audio_minutes),
            "video_frames": _r(self.video_frames, policy.max_video_frames),
            "input_tokens": _r(self.input_tokens, policy.max_input_tokens),
            "output_tokens": _r(self.output_tokens, policy.max_output_tokens),
            "tool_calls": _r(self.tool_calls, policy.max_tool_calls),
            "storage_gb": _r(self.storage_gb, policy.max_storage_gb),
        }

    def exceeded(self, policy: QuotaPolicy) -> list[str]:
        """返回已超额的维度名列表，空列表表示一切正常。"""
        return [name for name, r in self.ratio(policy).items() if r >= 1.0]


class QuotaCheck(OmniModel):
    """一次配额校验结果。"""

    allowed: bool = True
    reason: str = ""
    exceeded_dimensions: list[str] = Field(default_factory=list)
    retry_after_s: int = 0
    degrade_to: str | None = Field(default=None, description="on_exceed=degrade 时的目标 provider")

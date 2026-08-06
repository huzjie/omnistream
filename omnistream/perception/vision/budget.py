"""视觉预算控制器：给"边看边聊"装上刹车。

为什么必须有
------------
视频理解是整个平台**唯一会失控的成本项**。
音频是线性的（说 1 分钟就是 1 分钟），但视频不是：
一路 720p@30fps 的摄像头，如果不加约束地每帧都送 VLM，
一分钟就是 1800 次调用——几美元没了，而且 GPU 队列会直接堵死，
连带把语音链路的 CPU 也抢光。

所以视觉必须在四个维度上同时受限：

============  ====================================================
维度          约束
============  ====================================================
频率          每秒最多几次调用（令牌桶）
间隔          两次调用之间的最小时间（去抖）
总量          单会话累计调用次数 / token 数上限
时延          在途请求数（并发窗口），防止排队雪崩
============  ====================================================

任何一项越界，:meth:`VisionBudget.try_acquire` 都返回拒绝理由，
调用方**跳过这一帧**而不是排队——画面每秒都在变，排队等来的描述已经过时了。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Final

import structlog

from omnistream.perception.errors import VisionBudgetExceeded

log = structlog.get_logger(__name__)

__all__ = [
    "BudgetDecision",
    "BudgetSnapshot",
    "DEFAULT_POLICY",
    "VisionBudget",
    "VisionBudgetPolicy",
]

#: 拒绝原因常量
REASON_OK: Final[str] = ""
REASON_RATE: Final[str] = "rate_limited"
REASON_INTERVAL: Final[str] = "min_interval"
REASON_CALLS: Final[str] = "max_calls"
REASON_TOKENS: Final[str] = "max_tokens"
REASON_COST: Final[str] = "max_cost"
REASON_INFLIGHT: Final[str] = "max_inflight"
REASON_PAUSED: Final[str] = "paused"


@dataclass(slots=True)
class VisionBudgetPolicy:
    """预算策略。所有上限设为 0 表示该维度不限制。"""

    #: 每秒允许的调用次数（令牌桶速率）
    calls_per_second: float = 0.8
    #: 令牌桶容量，决定能容忍多大的突发
    burst: int = 3
    #: 两次调用的最小间隔（毫秒）
    min_interval_ms: int = 800
    #: 单会话累计调用上限
    max_calls: int = 400
    #: 单会话累计 token 上限（VLM 的输入+输出）
    max_tokens: int = 200_000
    #: 单会话累计成本上限（相对单位，1.0 约等于一次最贵的云端调用）
    max_cost: float = 0.0
    #: 同时在途的最大请求数
    max_inflight: int = 2
    #: 空闲（画面静止）时是否仍然允许周期性观察
    observe_while_idle: bool = True
    #: 空闲观察的最小间隔（毫秒），通常远大于活跃时
    idle_interval_ms: int = 8000

    def scaled(self, factor: float) -> "VisionBudgetPolicy":
        """按比例放宽/收紧速率类约束（用于按租户等级分档）。

        Args:
            factor: 缩放系数，>1 放宽，<1 收紧。

        Returns:
            新的策略对象，原对象不变。
        """
        f = max(0.01, float(factor))
        return VisionBudgetPolicy(
            calls_per_second=self.calls_per_second * f,
            burst=max(1, int(round(self.burst * f))),
            min_interval_ms=max(0, int(self.min_interval_ms / f)),
            max_calls=int(self.max_calls * f) if self.max_calls else 0,
            max_tokens=int(self.max_tokens * f) if self.max_tokens else 0,
            max_cost=self.max_cost * f if self.max_cost else 0.0,
            max_inflight=max(1, int(round(self.max_inflight * f))),
            observe_while_idle=self.observe_while_idle,
            idle_interval_ms=max(0, int(self.idle_interval_ms / f)),
        )


#: 默认策略：约每 1.2 秒一帧，单会话 400 次封顶
DEFAULT_POLICY: Final[VisionBudgetPolicy] = VisionBudgetPolicy()


@dataclass(slots=True)
class BudgetDecision:
    """一次预算判定的结果。"""

    allowed: bool
    reason: str = REASON_OK
    #: 建议多久之后重试（毫秒），0 表示无需等待或永远不必重试
    retry_after_ms: int = 0
    #: 剩余可用调用次数，-1 表示不限
    remaining_calls: int = -1

    def __bool__(self) -> bool:
        return self.allowed

    def as_dict(self) -> dict[str, Any]:
        """序列化。"""
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "retry_after_ms": self.retry_after_ms,
            "remaining_calls": self.remaining_calls,
        }


@dataclass(slots=True)
class BudgetSnapshot:
    """预算使用快照，用于指标上报与前端展示。"""

    calls: int = 0
    tokens: int = 0
    cost: float = 0.0
    rejected: int = 0
    inflight: int = 0
    tokens_available: float = 0.0
    rejections_by_reason: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """序列化。"""
        return {
            "calls": self.calls,
            "tokens": self.tokens,
            "cost": round(self.cost, 4),
            "rejected": self.rejected,
            "inflight": self.inflight,
            "tokens_available": round(self.tokens_available, 2),
            "rejections": dict(self.rejections_by_reason),
        }


class VisionBudget:
    """四维视觉预算控制器（令牌桶 + 去抖 + 总量 + 并发窗口）。

    Example::

        budget = VisionBudget(VisionBudgetPolicy(calls_per_second=1.0))
        decision = budget.try_acquire(idle=signal.is_idle)
        if not decision:
            stats.frames_dropped += 1
            return
        try:
            description = await engine.describe(request)
        finally:
            budget.release(tokens=used_tokens, cost=0.02)
    """

    def __init__(
        self,
        policy: VisionBudgetPolicy | None = None,
        *,
        session_id: str = "",
    ) -> None:
        """初始化。

        Args:
            policy: 预算策略，``None`` 用默认值。
            session_id: 会话 ID，仅用于日志。
        """
        self.policy = policy or DEFAULT_POLICY
        self.session_id = session_id
        self.paused = False

        self._tokens = float(self.policy.burst)
        self._last_refill = time.monotonic()
        self._last_acquire = 0.0
        self._inflight = 0

        self.calls = 0
        self.tokens_used = 0
        self.cost_used = 0.0
        self.rejected = 0
        self._rejections: dict[str, int] = {}

    # ------------------------------------------------------------------
    # 判定
    # ------------------------------------------------------------------

    @property
    def inflight(self) -> int:
        """当前在途请求数。"""
        return self._inflight

    @property
    def remaining_calls(self) -> int:
        """剩余调用次数，-1 表示不限。"""
        if self.policy.max_calls <= 0:
            return -1
        return max(0, self.policy.max_calls - self.calls)

    @property
    def exhausted(self) -> bool:
        """总量类预算是否已耗尽（不可恢复）。"""
        policy = self.policy
        if policy.max_calls and self.calls >= policy.max_calls:
            return True
        if policy.max_tokens and self.tokens_used >= policy.max_tokens:
            return True
        if policy.max_cost and self.cost_used >= policy.max_cost:
            return True
        return False

    def pause(self) -> None:
        """临时暂停视觉（例如用户关掉了摄像头）。"""
        self.paused = True

    def resume(self) -> None:
        """恢复视觉。"""
        self.paused = False

    def try_acquire(self, *, idle: bool = False, cost: float = 0.0) -> BudgetDecision:
        """尝试为一次视觉调用申请预算。

        Args:
            idle: 当前画面是否静止（静止时用更长的间隔）。
            cost: 本次调用的预估成本，用于提前判断成本上限。

        Returns:
            判定结果；``allowed=True`` 时已经扣减令牌并把在途数加一，
            调用方**必须**在结束后调用 :meth:`release`。
        """
        if self.paused:
            return self._reject(REASON_PAUSED, 0)

        policy = self.policy
        now = time.monotonic()

        if idle and not policy.observe_while_idle:
            return self._reject(REASON_INTERVAL, policy.idle_interval_ms)

        # 1) 总量约束（不可恢复，retry_after=0 表示别再试了）
        if policy.max_calls and self.calls >= policy.max_calls:
            return self._reject(REASON_CALLS, 0)
        if policy.max_tokens and self.tokens_used >= policy.max_tokens:
            return self._reject(REASON_TOKENS, 0)
        if policy.max_cost and (self.cost_used + cost) > policy.max_cost:
            return self._reject(REASON_COST, 0)

        # 2) 并发窗口
        if policy.max_inflight and self._inflight >= policy.max_inflight:
            return self._reject(REASON_INFLIGHT, 200)

        # 3) 最小间隔去抖
        interval_ms = policy.idle_interval_ms if idle else policy.min_interval_ms
        if interval_ms > 0 and self._last_acquire > 0.0:
            elapsed_ms = (now - self._last_acquire) * 1000.0
            if elapsed_ms < interval_ms:
                return self._reject(REASON_INTERVAL, int(interval_ms - elapsed_ms))

        # 4) 令牌桶
        self._refill(now)
        if policy.calls_per_second > 0.0 and self._tokens < 1.0:
            deficit = 1.0 - self._tokens
            wait_ms = int(deficit / policy.calls_per_second * 1000.0)
            return self._reject(REASON_RATE, max(1, wait_ms))

        if policy.calls_per_second > 0.0:
            self._tokens -= 1.0
        self._last_acquire = now
        self._inflight += 1
        self.calls += 1
        return BudgetDecision(allowed=True, remaining_calls=self.remaining_calls)

    def acquire_or_raise(self, *, idle: bool = False, cost: float = 0.0) -> BudgetDecision:
        """同 :meth:`try_acquire`，但被拒时抛异常。

        只在"必须成功"的路径上用（例如用户显式提问"你看到了什么"）。

        Raises:
            VisionBudgetExceeded: 预算不足。
        """
        decision = self.try_acquire(idle=idle, cost=cost)
        if not decision.allowed:
            raise VisionBudgetExceeded(
                f"视觉预算不足：{decision.reason}",
                used=self.calls,
                limit=self.policy.max_calls,
            )
        return decision

    def release(self, *, tokens: int = 0, cost: float = 0.0) -> None:
        """归还在途计数并累加实际消耗。

        Args:
            tokens: 本次实际消耗的 token 数。
            cost: 本次实际成本。
        """
        self._inflight = max(0, self._inflight - 1)
        self.tokens_used += max(0, int(tokens))
        self.cost_used += max(0.0, float(cost))

    def refund(self) -> None:
        """完全退款（请求根本没发出去时调用）。"""
        self._inflight = max(0, self._inflight - 1)
        self.calls = max(0, self.calls - 1)
        if self.policy.calls_per_second > 0.0:
            self._tokens = min(float(self.policy.burst), self._tokens + 1.0)

    def reset(self) -> None:
        """重置全部计数（新会话或新一轮配额周期）。"""
        self._tokens = float(self.policy.burst)
        self._last_refill = time.monotonic()
        self._last_acquire = 0.0
        self._inflight = 0
        self.calls = 0
        self.tokens_used = 0
        self.cost_used = 0.0
        self.rejected = 0
        self._rejections.clear()

    def snapshot(self) -> BudgetSnapshot:
        """当前使用快照。"""
        self._refill(time.monotonic())
        return BudgetSnapshot(
            calls=self.calls,
            tokens=self.tokens_used,
            cost=self.cost_used,
            rejected=self.rejected,
            inflight=self._inflight,
            tokens_available=self._tokens,
            rejections_by_reason=dict(self._rejections),
        )

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _refill(self, now: float) -> None:
        """按经过时间补充令牌。"""
        rate = self.policy.calls_per_second
        if rate <= 0.0:
            self._tokens = float(self.policy.burst)
            self._last_refill = now
            return
        elapsed = max(0.0, now - self._last_refill)
        if elapsed <= 0.0:
            return
        self._tokens = min(float(self.policy.burst), self._tokens + elapsed * rate)
        self._last_refill = now

    def _reject(self, reason: str, retry_after_ms: int) -> BudgetDecision:
        """记录一次拒绝。"""
        self.rejected += 1
        self._rejections[reason] = self._rejections.get(reason, 0) + 1
        return BudgetDecision(
            allowed=False,
            reason=reason,
            retry_after_ms=max(0, int(retry_after_ms)),
            remaining_calls=self.remaining_calls,
        )

    def __repr__(self) -> str:
        limit = self.policy.max_calls or "inf"
        return (f"<VisionBudget calls={self.calls}/{limit} "
                f"inflight={self._inflight} rejected={self.rejected}>")

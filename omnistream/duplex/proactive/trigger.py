"""主动发起：让助手在合适的时机**先开口**。

对标 SeedRealtime 的"主动交互"能力。三类触发源：

* **长静默** —— 用户超过 ``proactive_silence_ms`` 没说话，且当前处于 ``IDLE``；
* **视觉事件** —— 画面出现值得一提的变化（由 ``perception.vision.change_narrator`` 投递）；
* **记忆提醒** —— 记忆 Agent 认为有一条到点的、值得主动说出口的提醒。

**状态机约束（§5.5 冻结）**：只有 ``idle → speaking`` 一格标注了"主动发起"，
``listening → speaking`` 是非法迁移。因此本触发器默认只在 ``IDLE`` 态生效；
处于 ``LISTENING`` 时应先由静音超时把状态机带回 ``IDLE``，再考虑主动开口。

**不打扰原则**（防止变成话痨）：

1. 冷却窗口 ``cooldown_ms``；
2. 单会话上限 ``max_per_session``；
3. **退避**：连续 N 次主动开口都没得到用户回应时，静默阈值按
   ``1 + backoff_factor * N`` 放大，最多放大到 ``max_silence_ms``；
4. 用户一旦有任何动作（说话/打字/工具回填），立即清零退避并重置计时。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

import structlog

from omnistream.core.clock import now_ms, now_ns
from omnistream.protocol.enums import TurnState

log = structlog.get_logger(__name__)

#: 默认冷却窗口（毫秒）
DEFAULT_COOLDOWN_MS: Final[int] = 15_000

#: 默认单会话主动发起次数上限
DEFAULT_MAX_PER_SESSION: Final[int] = 3

#: 静默阈值退避系数
DEFAULT_BACKOFF_FACTOR: Final[float] = 0.75

#: 静默阈值放大上限（毫秒）
DEFAULT_MAX_SILENCE_MS: Final[int] = 90_000

#: 事件类触发源的存活时间（毫秒），过期即丢弃——过时的观察说出来很怪
DEFAULT_EVENT_TTL_MS: Final[int] = 12_000

#: 允许主动发起的状态（§5.5：只有 idle 行的 speaking 列为 ✓）
DEFAULT_ALLOWED_STATES: Final[tuple[TurnState, ...]] = (TurnState.IDLE,)


class ProactiveReason(StrEnum):
    """主动发起的原因。"""

    GREETING = "greeting"
    LONG_SILENCE = "long_silence"
    VISION_EVENT = "vision_event"
    MEMORY_REMINDER = "memory_reminder"


#: 各原因的优先级，数值大者先出口
PRIORITY: Final[dict[ProactiveReason, int]] = {
    ProactiveReason.GREETING: 4,
    ProactiveReason.MEMORY_REMINDER: 3,
    ProactiveReason.VISION_EVENT: 2,
    ProactiveReason.LONG_SILENCE: 1,
}

#: 静默触发的话术模板，按连续次数递进（越往后越克制）
_SILENCE_PROMPTS: Final[tuple[str, ...]] = (
    "用户已经安静了一会儿。用一句话自然地确认他是否还需要帮助，不要复述之前的内容。",
    "用户仍然没有回应。用不超过一句话温和地提示你随时在，然后保持安静。",
    "用户长时间没有回应。只说一句简短的收尾话，之后不要再主动开口。",
)


@dataclass(slots=True, frozen=True)
class ProactiveIntent:
    """一次主动发起意图。编排器据此调用 ``provider.request_response()``。"""

    reason: ProactiveReason
    #: 作为 ``instructions_suffix`` 追加给模型的一句指令
    prompt: str
    priority: int = 1
    #: 触发时刻（UTC 毫秒）
    created_at_ms: int = field(default_factory=now_ms)
    #: 触发源的附加信息，写审计日志与控制台
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "reason": str(self.reason),
            "prompt": self.prompt,
            "priority": self.priority,
            "created_at_ms": self.created_at_ms,
            "detail": dict(self.detail),
        }


@dataclass(slots=True)
class _PendingEvent:
    """排队中的事件型触发源。"""

    reason: ProactiveReason
    prompt: str
    priority: int
    enqueued_ns: int
    detail: dict[str, Any] = field(default_factory=dict)

    def expired(self, ttl_ms: int) -> bool:
        return (now_ns() - self.enqueued_ns) / 1e6 > ttl_ms


class ProactiveTrigger:
    """主动发起触发器。"""

    def __init__(
        self,
        *,
        session_id: str = "",
        enabled: bool = False,
        silence_ms: int = 8_000,
        cooldown_ms: int = DEFAULT_COOLDOWN_MS,
        max_per_session: int = DEFAULT_MAX_PER_SESSION,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        max_silence_ms: int = DEFAULT_MAX_SILENCE_MS,
        event_ttl_ms: int = DEFAULT_EVENT_TTL_MS,
        allowed_states: tuple[TurnState, ...] = DEFAULT_ALLOWED_STATES,
        greet_on_start: bool = False,
    ) -> None:
        self.session_id = session_id
        self.enabled = bool(enabled)
        self.silence_ms = int(silence_ms)
        self.cooldown_ms = int(cooldown_ms)
        self.max_per_session = int(max_per_session)
        self.backoff_factor = float(backoff_factor)
        self.max_silence_ms = int(max_silence_ms)
        self.event_ttl_ms = int(event_ttl_ms)
        self.allowed_states = tuple(allowed_states)

        self._last_activity_ns = now_ns()
        self._last_fire_ns = 0
        self._fired_total = 0
        self._unanswered = 0
        self._pending: list[_PendingEvent] = []
        self._exhausted_logged = False

        if greet_on_start and self.enabled:
            self.submit_greeting()

    # ------------------------------------------------------------------
    # 配置装载
    # ------------------------------------------------------------------

    @classmethod
    def from_session_config(cls, config: Any, *, session_id: str = "",
                            greet_on_start: bool = False) -> "ProactiveTrigger":
        """从 ``SessionConfig`` 构造（只读取 proactive_* 两个字段）。"""
        return cls(
            session_id=session_id,
            enabled=bool(getattr(config, "proactive_enabled", False)),
            silence_ms=int(getattr(config, "proactive_silence_ms", 8_000)),
            greet_on_start=greet_on_start,
        )

    # ------------------------------------------------------------------
    # 活动记录
    # ------------------------------------------------------------------

    def note_activity(self, kind: str = "user_speech") -> None:
        """用户有任何动作时调用：重置静默计时并清零退避。"""
        self._last_activity_ns = now_ns()
        if self._unanswered:
            log.debug("proactive.answered", session_id=self.session_id,
                      unanswered=self._unanswered, kind=kind)
        self._unanswered = 0

    def note_assistant_output(self) -> None:
        """助手（被动地）说完话时调用：静默计时从此刻重新开始。"""
        self._last_activity_ns = now_ns()

    @property
    def silence_elapsed_ms(self) -> float:
        """自上次活动以来的静默时长（毫秒）。"""
        return max(0.0, (now_ns() - self._last_activity_ns) / 1e6)

    @property
    def effective_silence_ms(self) -> int:
        """当前生效的静默阈值（含退避放大）。"""
        scaled = self.silence_ms * (1.0 + self.backoff_factor * self._unanswered)
        return int(min(float(self.max_silence_ms), scaled))

    @property
    def cooldown_remaining_ms(self) -> float:
        if self._last_fire_ns == 0:
            return 0.0
        elapsed = (now_ns() - self._last_fire_ns) / 1e6
        return max(0.0, self.cooldown_ms - elapsed)

    @property
    def budget_left(self) -> int:
        return max(0, self.max_per_session - self._fired_total)

    # ------------------------------------------------------------------
    # 事件投递
    # ------------------------------------------------------------------

    def submit_vision_event(self, description: str, *, score: float = 0.0,
                            pts_ms: int = 0) -> None:
        """视觉变化事件（由 change_narrator 投递）。"""
        if not description.strip():
            return
        prompt = (f"你刚刚注意到画面里的变化：{description.strip()}。"
                  f"如果这值得一提，用一句话自然地说出来；否则保持简短。")
        self._enqueue(_PendingEvent(
            reason=ProactiveReason.VISION_EVENT,
            prompt=prompt,
            priority=PRIORITY[ProactiveReason.VISION_EVENT],
            enqueued_ns=now_ns(),
            detail={"description": description, "score": round(float(score), 3),
                    "pts_ms": int(pts_ms)},
        ))

    def submit_memory_reminder(self, rendered: str, *, item_id: str = "",
                               importance: float = 0.5) -> None:
        """记忆提醒（由主动记忆 Agent 投递，已渲染成一句话）。"""
        if not rendered.strip():
            return
        prompt = f"你想起一件与用户相关的事：{rendered.strip()}。用一句话自然地提起它。"
        self._enqueue(_PendingEvent(
            reason=ProactiveReason.MEMORY_REMINDER,
            prompt=prompt,
            priority=PRIORITY[ProactiveReason.MEMORY_REMINDER],
            enqueued_ns=now_ns(),
            detail={"item_id": item_id, "importance": round(float(importance), 3)},
        ))

    def submit_greeting(self, text: str = "") -> None:
        """会话开始时的主动问候。"""
        prompt = text.strip() or "会话刚建立。用一句简短的话打招呼并说明你可以随时被打断。"
        self._enqueue(_PendingEvent(
            reason=ProactiveReason.GREETING,
            prompt=prompt,
            priority=PRIORITY[ProactiveReason.GREETING],
            enqueued_ns=now_ns(),
            detail={},
        ))

    def _enqueue(self, event: _PendingEvent) -> None:
        if not self.enabled:
            return
        # 同类事件只保留最新的一条，避免堆积后一次性说一堆
        self._pending = [e for e in self._pending if e.reason is not event.reason]
        self._pending.append(event)

    # ------------------------------------------------------------------
    # 轮询
    # ------------------------------------------------------------------

    def poll(self, state: TurnState) -> ProactiveIntent | None:
        """评估是否应当主动开口。

        Args:
            state: 当前轮次状态。

        Returns:
            可以主动开口时返回 :class:`ProactiveIntent`，否则 ``None``。
        """
        if not self.enabled:
            return None
        if state not in self.allowed_states:
            return None
        if self.budget_left <= 0:
            if not self._exhausted_logged:
                self._exhausted_logged = True
                log.info("proactive.budget_exhausted", session_id=self.session_id,
                         max_per_session=self.max_per_session)
            return None
        if self.cooldown_remaining_ms > 0.0:
            return None

        self._drop_expired()

        if self._pending:
            self._pending.sort(key=lambda e: (e.priority, e.enqueued_ns), reverse=True)
            event = self._pending.pop(0)
            return ProactiveIntent(
                reason=event.reason,
                prompt=event.prompt,
                priority=event.priority,
                detail=dict(event.detail),
            )

        if self.silence_elapsed_ms < self.effective_silence_ms:
            return None

        idx = min(self._unanswered, len(_SILENCE_PROMPTS) - 1)
        return ProactiveIntent(
            reason=ProactiveReason.LONG_SILENCE,
            prompt=_SILENCE_PROMPTS[idx],
            priority=PRIORITY[ProactiveReason.LONG_SILENCE],
            detail={
                "silence_ms": round(self.silence_elapsed_ms, 0),
                "threshold_ms": self.effective_silence_ms,
                "attempt": self._unanswered + 1,
            },
        )

    def next_deadline_ms(self) -> int:
        """距离下一次可能触发还有多久（毫秒），供 ``TimerSet`` 调度。"""
        if not self.enabled or self.budget_left <= 0:
            return -1
        if self._pending:
            return max(1, int(self.cooldown_remaining_ms))
        remain = self.effective_silence_ms - self.silence_elapsed_ms
        return max(1, int(max(remain, self.cooldown_remaining_ms)))

    # ------------------------------------------------------------------
    # 执行反馈
    # ------------------------------------------------------------------

    def mark_fired(self, intent: ProactiveIntent) -> None:
        """编排器确实发起了这次主动说话之后回调。"""
        self._last_fire_ns = now_ns()
        self._last_activity_ns = now_ns()
        self._fired_total += 1
        if intent.reason is ProactiveReason.LONG_SILENCE:
            self._unanswered += 1
        log.info("proactive.fired", session_id=self.session_id,
                 reason=str(intent.reason), fired_total=self._fired_total,
                 budget_left=self.budget_left, unanswered=self._unanswered)

    def _drop_expired(self) -> None:
        if not self._pending:
            return
        alive = [e for e in self._pending if not e.expired(self.event_ttl_ms)]
        dropped = len(self._pending) - len(alive)
        if dropped:
            log.debug("proactive.events_expired", session_id=self.session_id, dropped=dropped)
        self._pending = alive

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """会话复位。"""
        self._last_activity_ns = now_ns()
        self._last_fire_ns = 0
        self._fired_total = 0
        self._unanswered = 0
        self._pending.clear()
        self._exhausted_logged = False

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "fired_total": self._fired_total,
            "budget_left": self.budget_left,
            "unanswered": self._unanswered,
            "silence_elapsed_ms": round(self.silence_elapsed_ms, 0),
            "effective_silence_ms": self.effective_silence_ms,
            "cooldown_remaining_ms": round(self.cooldown_remaining_ms, 0),
            "pending": [str(e.reason) for e in self._pending],
        }

    def __repr__(self) -> str:
        return (f"<ProactiveTrigger enabled={self.enabled} fired={self._fired_total}"
                f"/{self.max_per_session} silence={self.effective_silence_ms}ms>")


__all__ = [
    "DEFAULT_ALLOWED_STATES",
    "DEFAULT_BACKOFF_FACTOR",
    "DEFAULT_COOLDOWN_MS",
    "DEFAULT_EVENT_TTL_MS",
    "DEFAULT_MAX_PER_SESSION",
    "DEFAULT_MAX_SILENCE_MS",
    "PRIORITY",
    "ProactiveIntent",
    "ProactiveReason",
    "ProactiveTrigger",
]

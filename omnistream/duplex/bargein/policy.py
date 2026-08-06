"""打断策略：决定"检测到插话之后到底怎么办"。

三种模式（``BargeInConfig.mode``）：

* ``immediate`` —— 立即止损。丢弃全部未播音频、取消 provider 生成，
  状态机 SPEAKING → INTERRUPTED → LISTENING。默认，也是延迟最优的模式。
* ``finish_sentence`` —— 让助手把当前这句话说完再停。适合朗读/播报类业务，
  避免半句被腰斩。会给一个不超过 ``max_grace_ms`` 的宽限期，
  宽限期内仍然会先取消 provider 的后续生成（不再产出新句子）。
* ``off`` —— 完全忽略打断（半双工体验），仅记录指标。

另有两道保护：

* **冷却窗口** ``cooldown_ms``：一次打断生效后的静默期，防止用户一句话触发多次截断。
* **起播保护** ``guard_ms``：助手刚出声的前若干毫秒内不接受打断，
  用于兜底浏览器 AEC 尚未收敛导致的自激回声。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import structlog

from omnistream.core.clock import now_ns
from omnistream.duplex.bargein.detector import BargeInSignal
from omnistream.protocol.enums import TurnState
from omnistream.schemas.session import BargeInConfig

log = structlog.get_logger(__name__)

#: finish_sentence 模式下最长等待的宽限时间。
DEFAULT_MAX_GRACE_MS: Final[int] = 600

#: 助手起播后的自激保护窗口。
DEFAULT_GUARD_MS: Final[int] = 80


class BargeInAction(StrEnum):
    """策略输出的动作。"""

    IGNORE = "ignore"
    TRUNCATE_NOW = "truncate_now"
    FINISH_SENTENCE = "finish_sentence"


@dataclass(slots=True, frozen=True)
class PolicyDecision:
    """一次打断决策。"""

    action: BargeInAction
    reason: str
    #: 截断播放前的等待时间（毫秒）。``TRUNCATE_NOW`` 恒为 0。
    grace_ms: int = 0
    #: 是否需要向 provider 发送 cancel_response
    cancel_provider: bool = True
    #: 是否需要清空播放队列
    truncate_playback: bool = True
    #: 是否需要驱动状态机迁移到 INTERRUPTED
    transition_state: bool = True
    #: 截断位置：``None`` 表示"从当前播放位置起全部丢弃"
    truncate_from_pts_ms: int | None = None

    @property
    def is_ignore(self) -> bool:
        return self.action is BargeInAction.IGNORE


#: 被忽略时复用的不可变决策实例工厂
def _ignore(reason: str) -> PolicyDecision:
    return PolicyDecision(
        action=BargeInAction.IGNORE,
        reason=reason,
        grace_ms=0,
        cancel_provider=False,
        truncate_playback=False,
        transition_state=False,
    )


class BargeInPolicy:
    """把 :class:`BargeInSignal` 翻译成可执行的 :class:`PolicyDecision`。"""

    def __init__(
        self,
        config: BargeInConfig | None = None,
        *,
        session_id: str = "",
        max_grace_ms: int = DEFAULT_MAX_GRACE_MS,
        guard_ms: int = DEFAULT_GUARD_MS,
    ) -> None:
        self.config = config or BargeInConfig()
        self.session_id = session_id
        self.max_grace_ms = int(max_grace_ms)
        self.guard_ms = int(guard_ms)

        self._last_fire_ns = 0
        self._fired_total = 0
        self._ignored_total = 0
        self._ignore_reasons: dict[str, int] = {}

    # ------------------------------------------------------------------
    # 冷却
    # ------------------------------------------------------------------

    @property
    def cooldown_remaining_ms(self) -> float:
        """冷却窗口剩余时间（毫秒），0 表示已可再次打断。"""
        if self._last_fire_ns == 0:
            return 0.0
        elapsed_ms = (now_ns() - self._last_fire_ns) / 1e6
        return max(0.0, float(self.config.cooldown_ms) - elapsed_ms)

    @property
    def in_cooldown(self) -> bool:
        return self.cooldown_remaining_ms > 0.0

    def mark_fired(self) -> None:
        """执行器成功止损后回调，开启冷却窗口。"""
        self._last_fire_ns = now_ns()
        self._fired_total += 1

    def reset(self) -> None:
        """会话/轮次复位，清空冷却。"""
        self._last_fire_ns = 0

    # ------------------------------------------------------------------
    # 决策
    # ------------------------------------------------------------------

    def decide(
        self,
        signal: BargeInSignal,
        *,
        state: TurnState,
        played_ms: int = 0,
        pending_ms: int = 0,
        sentence_end_pts_ms: int | None = None,
    ) -> PolicyDecision:
        """产出打断决策。

        Args:
            signal: 检测器确认的打断信号。
            state: 当前状态机状态。
            played_ms: 播放队列已播出的时长（毫秒）。
            pending_ms: 播放队列中尚未播出的时长（毫秒）。
            sentence_end_pts_ms: 当前句子在播放时间轴上的结束位置；
                ``finish_sentence`` 模式据此计算宽限期。

        Returns:
            :class:`PolicyDecision`
        """
        if not self.config.enabled or self.config.mode == "off":
            return self._record_ignore("disabled")

        if state not in (TurnState.SPEAKING, TurnState.THINKING):
            return self._record_ignore("state_not_interruptible")

        if self.in_cooldown:
            return self._record_ignore("cooldown")

        # 起播自激保护：只在真正出声之后才有意义。
        if state is TurnState.SPEAKING and 0 < played_ms < self.guard_ms:
            return self._record_ignore("startup_guard")

        # 队列已经空了，说明助手其实已经说完，没有必要走截断流程。
        if state is TurnState.SPEAKING and pending_ms <= 0 and played_ms > 0:
            return PolicyDecision(
                action=BargeInAction.TRUNCATE_NOW,
                reason="playback_drained",
                grace_ms=0,
                cancel_provider=True,
                truncate_playback=False,
                transition_state=True,
            )

        if self.config.mode == "finish_sentence" and state is TurnState.SPEAKING:
            grace = self._compute_grace_ms(played_ms=played_ms,
                                           pending_ms=pending_ms,
                                           sentence_end_pts_ms=sentence_end_pts_ms)
            if grace > 0:
                return PolicyDecision(
                    action=BargeInAction.FINISH_SENTENCE,
                    reason="finish_current_sentence",
                    grace_ms=grace,
                    cancel_provider=True,
                    truncate_playback=True,
                    transition_state=True,
                    truncate_from_pts_ms=sentence_end_pts_ms,
                )

        return PolicyDecision(
            action=BargeInAction.TRUNCATE_NOW,
            reason="immediate",
            grace_ms=0,
            cancel_provider=True,
            truncate_playback=True,
            transition_state=True,
        )

    def _compute_grace_ms(self, *, played_ms: int, pending_ms: int,
                          sentence_end_pts_ms: int | None) -> int:
        """计算 finish_sentence 的宽限时间，并夹在 [0, max_grace_ms]。"""
        if sentence_end_pts_ms is not None:
            remain = int(sentence_end_pts_ms) - int(played_ms)
        else:
            remain = int(pending_ms)
        if remain <= 0:
            return 0
        return int(min(self.max_grace_ms, remain))

    def _record_ignore(self, reason: str) -> PolicyDecision:
        self._ignored_total += 1
        self._ignore_reasons[reason] = self._ignore_reasons.get(reason, 0) + 1
        log.debug("barge_in.ignored", session_id=self.session_id, reason=reason)
        return _ignore(reason)

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, object]:
        """策略侧统计，供控制台调参。"""
        return {
            "mode": self.config.mode,
            "enabled": bool(self.config.enabled),
            "fired_total": self._fired_total,
            "ignored_total": self._ignored_total,
            "ignore_reasons": dict(self._ignore_reasons),
            "cooldown_remaining_ms": round(self.cooldown_remaining_ms, 1),
        }

    def __repr__(self) -> str:
        return f"<BargeInPolicy mode={self.config.mode} cooldown={self.config.cooldown_ms}ms>"


__all__ = [
    "DEFAULT_GUARD_MS",
    "DEFAULT_MAX_GRACE_MS",
    "BargeInAction",
    "BargeInPolicy",
    "PolicyDecision",
]

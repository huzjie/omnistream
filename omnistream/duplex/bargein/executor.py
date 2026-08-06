"""打断执行器：把决策落地成"三路并发止损"。

严格对齐 §4.1::

    BID->>SM: transition(INTERRUPTED)
    par 三路并发止损（<200ms）
        BID->>PQ: cancel_from(current_pts) 【丢弃未播帧】
    and
        BID->>PRV: cancel_response(INTERRUPTED)
    and
        BID-->>WS: barge_in.detected + playback.truncated{played_ms}
    end
    SM->>SM: INTERRUPTED → LISTENING

设计要点：

* **状态机迁移放在并发之前**，且是同步调用 —— 状态一旦翻转，
  上游音频路径立即知道"助手已经不在说话了"，避免竞态下继续入队。
* **播放队列截断是同步接口**（§5.4 冻结签名 ``cancel_from(pts_ms) -> int``），
  在事件循环里直接调用，不引入额外调度延迟，这是压住 200ms SLO 的关键。
* **provider 取消带硬超时**：契约要求 100ms 内停止 ``audio_delta``；
  这里给 ``provider_timeout_ms`` 的宽限，超时只记录不抛，绝不阻塞止损。
* **事件下发不阻塞止损**：与前两路并发执行。
"""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import structlog

from omnistream.core.clock import now_ns
from omnistream.duplex.bargein.detector import BargeInSignal
from omnistream.duplex.bargein.policy import BargeInAction, BargeInPolicy, PolicyDecision
from omnistream.duplex.turn.state_machine import TurnStateMachine
from omnistream.duplex.turn.transitions import PlaybackQueueProtocol
from omnistream.duplex.turn.turn_manager import TurnManager
from omnistream.protocol.enums import ServerEventType, StopReason, TurnState

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查期需要，运行期不依赖 WS5
    from omnistream.providers.base import BaseRealtimeProvider

log = structlog.get_logger(__name__)

#: provider.cancel_response 的等待上限（毫秒）。契约要求 100ms，这里留 4 倍冗余。
DEFAULT_PROVIDER_TIMEOUT_MS: Final[int] = 400

#: barge-in SLO（毫秒）。超过则打 WARNING 日志，指标侧也会统计越界率。
BARGE_IN_SLO_MS: Final[float] = 200.0

EventEmitter = Callable[[ServerEventType, dict[str, Any]], Awaitable[None] | None]


@dataclass(slots=True)
class BargeInOutcome:
    """一次打断执行的结果，写入轮次指标与审计日志。"""

    executed: bool
    action: BargeInAction
    reason: str
    #: 用户开口 → 播放停止（毫秒），即 SLO 指标 ``barge_in_ms``
    latency_ms: float = 0.0
    #: 播放队列在截断时刻已播出的时长
    played_ms: int = 0
    #: 被丢弃的音频时长
    truncated_ms: int = 0
    provider_cancelled: bool = False
    state_changed: bool = False
    within_slo: bool = True
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "executed": self.executed,
            "action": str(self.action),
            "reason": self.reason,
            "latency_ms": round(self.latency_ms, 2),
            "played_ms": self.played_ms,
            "truncated_ms": self.truncated_ms,
            "provider_cancelled": self.provider_cancelled,
            "state_changed": self.state_changed,
            "within_slo": self.within_slo,
            "errors": list(self.errors),
        }


class BargeInExecutor:
    """执行打断：撤销播放队列、通知 provider cancel、状态机迁移、记录延迟。"""

    def __init__(
        self,
        machine: TurnStateMachine,
        turns: TurnManager,
        *,
        playback: PlaybackQueueProtocol | None = None,
        provider: "BaseRealtimeProvider | None" = None,
        policy: BargeInPolicy | None = None,
        emit: EventEmitter | None = None,
        session_id: str = "",
        provider_timeout_ms: int = DEFAULT_PROVIDER_TIMEOUT_MS,
        auto_resume_listening: bool = True,
    ) -> None:
        self.machine = machine
        self.turns = turns
        self.playback = playback
        self.provider = provider
        self.policy = policy
        self.emit = emit
        self.session_id = session_id
        self.provider_timeout_ms = int(provider_timeout_ms)
        self.auto_resume_listening = bool(auto_resume_listening)

        self.total_executed = 0
        self.total_slo_violations = 0
        self.last_outcome: BargeInOutcome | None = None

    # ------------------------------------------------------------------
    # 运行期装配（WS3/WS5 就绪后换真身）
    # ------------------------------------------------------------------

    def bind_playback(self, playback: PlaybackQueueProtocol | None) -> None:
        """注入真实播放队列（WS3 的 ``media.audio.PlaybackQueue``）。"""
        self.playback = playback

    def bind_provider(self, provider: "BaseRealtimeProvider | None") -> None:
        """注入真实 provider 实例（WS5）。"""
        self.provider = provider

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    async def execute(self, signal: BargeInSignal, decision: PolicyDecision) -> BargeInOutcome:
        """执行一次打断。

        Args:
            signal: 检测器确认的信号（携带用户开口时刻）。
            decision: 策略决策。

        Returns:
            :class:`BargeInOutcome`，含 ``barge_in_ms`` 延迟。
        """
        if decision.is_ignore:
            return BargeInOutcome(executed=False, action=decision.action, reason=decision.reason)

        outcome = BargeInOutcome(executed=True, action=decision.action, reason=decision.reason)

        # ---- ① 状态机先翻转（同步、零延迟） ----
        if decision.transition_state:
            outcome.state_changed = self._enter_interrupted(signal, decision)

        # ---- ② 三路并发止损 ----
        results = await asyncio.gather(
            self._truncate_playback(signal, decision, outcome),
            self._cancel_provider(outcome),
            self._emit_events(signal, decision, outcome),
            return_exceptions=True,
        )
        for res in results:
            if isinstance(res, BaseException):
                outcome.errors.append(f"{type(res).__name__}: {res}")
                log.warning("barge_in.branch_failed", session_id=self.session_id,
                            error=str(res), error_type=type(res).__name__)

        # ---- ③ 记录延迟并回到 listening ----
        if outcome.latency_ms <= 0.0:
            outcome.latency_ms = max(0.0, (now_ns() - signal.onset_mono_ns) / 1e6)
        outcome.within_slo = outcome.latency_ms <= BARGE_IN_SLO_MS

        self.turns.mark_barge_in(outcome.latency_ms)
        if self.policy is not None:
            self.policy.mark_fired()

        self.total_executed += 1
        if not outcome.within_slo:
            self.total_slo_violations += 1

        if self.auto_resume_listening:
            self._resume_listening()

        self.last_outcome = outcome
        log_fn = log.info if outcome.within_slo else log.warning
        log_fn(
            "barge_in.executed",
            session_id=self.session_id,
            turn_id=self.turns.current_id,
            **outcome.as_dict(),
        )
        return outcome

    # ------------------------------------------------------------------
    # 三路分支
    # ------------------------------------------------------------------

    def _enter_interrupted(self, signal: BargeInSignal, decision: PolicyDecision) -> bool:
        """状态机迁移到 INTERRUPTED。非法迁移不抛异常，只记录。"""
        detail = {
            "onset_pts_ms": signal.onset_pts_ms,
            "energy_db": round(signal.energy_db, 1),
            "duration_ms": round(signal.duration_ms, 1),
            "action": str(decision.action),
        }
        return self.machine.try_transition(TurnState.INTERRUPTED, "barge_in", detail=detail)

    async def _truncate_playback(self, signal: BargeInSignal, decision: PolicyDecision,
                                 outcome: BargeInOutcome) -> None:
        """丢弃未播音频。这一路的完成时刻即 ``barge_in_ms`` 的终点。"""
        if decision.grace_ms > 0:
            # finish_sentence：让当前句播完，但宽限有上限。
            await asyncio.sleep(decision.grace_ms / 1000.0)

        if self.playback is None:
            outcome.latency_ms = max(0.0, (now_ns() - signal.onset_mono_ns) / 1e6)
            return

        try:
            outcome.played_ms = int(self.playback.played_ms)
        except Exception as exc:  # 播放队列实现异常不能中断止损
            outcome.errors.append(f"played_ms: {exc}")
            outcome.played_ms = 0

        if not decision.truncate_playback:
            outcome.latency_ms = max(0.0, (now_ns() - signal.onset_mono_ns) / 1e6)
            return

        try:
            dropped = self.playback.cancel_from(decision.truncate_from_pts_ms)
            outcome.truncated_ms = int(dropped)
        except Exception as exc:
            outcome.errors.append(f"cancel_from: {exc}")

        outcome.latency_ms = max(0.0, (now_ns() - signal.onset_mono_ns) / 1e6)

    async def _cancel_provider(self, outcome: BargeInOutcome) -> None:
        """通知上游停止生成。超时只记录，不影响本地止损。"""
        if self.provider is None:
            return
        try:
            await asyncio.wait_for(
                self.provider.cancel_response(StopReason.INTERRUPTED),
                timeout=self.provider_timeout_ms / 1000.0,
            )
            outcome.provider_cancelled = True
        except TimeoutError:
            outcome.errors.append("provider_cancel_timeout")
            log.warning("barge_in.provider_cancel_timeout", session_id=self.session_id,
                        timeout_ms=self.provider_timeout_ms)
        except Exception as exc:
            outcome.errors.append(f"provider_cancel: {exc}")
            log.warning("barge_in.provider_cancel_failed", session_id=self.session_id,
                        error=str(exc))

    async def _emit_events(self, signal: BargeInSignal, decision: PolicyDecision,
                           outcome: BargeInOutcome) -> None:
        """下发 barge_in.detected 与 playback.truncated。"""
        if self.emit is None:
            return
        await self._emit_one(ServerEventType.BARGE_IN, {
            "turn_id": self.turns.current_id,
            "detected_at_ms": signal.detected_at_ms,
            "onset_pts_ms": signal.onset_pts_ms,
            "duration_ms": round(signal.duration_ms, 1),
            "energy_db": round(signal.energy_db, 1),
            "prob": round(signal.prob, 3),
            "mode": str(decision.action),
            "reason": decision.reason,
        })
        if decision.truncate_playback:
            await self._emit_one(ServerEventType.PLAYBACK_TRUNCATED, {
                "turn_id": self.turns.current_id,
                "played_ms": outcome.played_ms,
                "truncated_from_pts_ms": decision.truncate_from_pts_ms,
            })

    async def _emit_one(self, kind: ServerEventType, payload: dict[str, Any]) -> None:
        if self.emit is None:
            return
        result = self.emit(kind, payload)
        if asyncio.iscoroutine(result):
            with contextlib.suppress(Exception):
                await result

    def _resume_listening(self) -> None:
        """INTERRUPTED → LISTENING（§4.1 收尾），用户的插话内容不丢。"""
        if self.machine.state is not TurnState.INTERRUPTED:
            return
        self.machine.try_transition(TurnState.LISTENING, "resume_after_barge_in")

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "total_executed": self.total_executed,
            "slo_violations": self.total_slo_violations,
            "last": self.last_outcome.as_dict() if self.last_outcome else None,
        }

    def __repr__(self) -> str:
        return (f"<BargeInExecutor executed={self.total_executed} "
                f"violations={self.total_slo_violations}>")


__all__ = [
    "BARGE_IN_SLO_MS",
    "DEFAULT_PROVIDER_TIMEOUT_MS",
    "BargeInExecutor",
    "BargeInOutcome",
    "EventEmitter",
]

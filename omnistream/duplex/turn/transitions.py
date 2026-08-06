"""迁移守卫与副作用绑定。

把"状态机"与"外设"解耦：状态机只管合法性，本模块负责在正确的时机
启动播放、清空队列、重排定时器、下发 ``turn.state`` 事件。

绑定关系（对齐 §4.1）::

    enter LISTENING   → 重排 silence 定时器；清 thinking 定时器；打点 speech_start
    exit  LISTENING   → 清 silence 定时器
    enter THINKING    → 排 thinking 超时；打点 speech_end
    exit  THINKING    → 清 thinking 定时器
    enter SPEAKING    → 允许播放队列出声（resume）；排 max_turn 定时器
    exit  SPEAKING    → 清空播放队列（未播帧全部作废）
    enter INTERRUPTED → 清空播放队列（幂等）；停一切定时器
    enter IDLE        → 全部定时器清零；播放队列清空

所有回调都是**同步**的（必要的异步动作用 ``schedule_async`` 派发），
保证 barge-in 路径上 ``transition()`` 不会挂起。
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

import structlog

from omnistream.duplex.turn.state_machine import StateChange, TurnStateMachine
from omnistream.duplex.turn.timers import TimerName, TimerSet
from omnistream.protocol.enums import TurnState

log = structlog.get_logger(__name__)


@runtime_checkable
class PlaybackQueueProtocol(Protocol):
    """WS3 ``media.audio.playback_queue.PlaybackQueue`` 的最小契约（§5.4 冻结）。"""

    async def enqueue(self, chunk: Any) -> None:
        ...

    def cancel_from(self, pts_ms: int | None = None) -> int:
        ...

    @property
    def played_ms(self) -> int:
        ...


class FakePlaybackQueue:
    """零依赖播放队列，WS3 未就绪或纯文本模态时使用。

    行为与真实 ``PlaybackQueue`` 一致：入队计时长、``cancel_from`` 同步丢弃、
    ``played_ms`` 单调递增。**不做真实播放**，只维护时间轴。
    """

    def __init__(self) -> None:
        self._pending: list[Any] = []
        self._played_ms = 0
        self._dropped_ms = 0
        self._paused = True

    async def enqueue(self, chunk: Any) -> None:
        self._pending.append(chunk)
        duration = float(getattr(chunk, "duration_ms", 20.0) or 20.0)
        if not self._paused:
            self._played_ms += int(duration)
            self._pending.pop(0)

    def cancel_from(self, pts_ms: int | None = None) -> int:
        dropped = 0.0
        keep: list[Any] = []
        for chunk in self._pending:
            chunk_pts = int(getattr(chunk, "pts_ms", 0) or 0)
            if pts_ms is None or chunk_pts >= pts_ms:
                dropped += float(getattr(chunk, "duration_ms", 20.0) or 20.0)
            else:
                keep.append(chunk)
        self._pending = keep
        self._dropped_ms += int(dropped)
        return int(dropped)

    def resume(self) -> None:
        self._paused = False

    def pause(self) -> None:
        self._paused = True

    @property
    def played_ms(self) -> int:
        return self._played_ms

    @property
    def dropped_ms(self) -> int:
        return self._dropped_ms

    @property
    def pending_ms(self) -> int:
        return int(sum(float(getattr(c, "duration_ms", 20.0) or 20.0) for c in self._pending))


class TransitionEffects:
    """把状态机迁移翻译成对播放队列/定时器/事件下发的副作用。"""

    def __init__(
        self,
        machine: TurnStateMachine,
        timers: TimerSet,
        playback: PlaybackQueueProtocol | FakePlaybackQueue,
        *,
        silence_timeout_ms: int = 8_000,
        thinking_timeout_ms: int = 15_000,
        max_turn_ms: int = 120_000,
        idle_timeout_ms: int = 300_000,
        on_state_event: Callable[[StateChange], Awaitable[None] | None] | None = None,
    ) -> None:
        self.machine = machine
        self.timers = timers
        self.playback = playback
        self.silence_timeout_ms = int(silence_timeout_ms)
        self.thinking_timeout_ms = int(thinking_timeout_ms)
        self.max_turn_ms = int(max_turn_ms)
        self.idle_timeout_ms = int(idle_timeout_ms)
        self.on_state_event = on_state_event
        self._bound = False

    # ------------------------------------------------------------------

    def bind(self) -> None:
        """把回调注册到状态机（幂等）。"""
        if self._bound:
            return
        self._bound = True
        self.machine.on_enter(TurnState.IDLE, self._enter_idle)
        self.machine.on_enter(TurnState.LISTENING, self._enter_listening)
        self.machine.on_exit(TurnState.LISTENING, self._exit_listening)
        self.machine.on_enter(TurnState.THINKING, self._enter_thinking)
        self.machine.on_exit(TurnState.THINKING, self._exit_thinking)
        self.machine.on_enter(TurnState.SPEAKING, self._enter_speaking)
        self.machine.on_exit(TurnState.SPEAKING, self._exit_speaking)
        self.machine.on_enter(TurnState.INTERRUPTED, self._enter_interrupted)
        self.machine.on_any(self._emit_state)
        log.debug("transition_effects_bound", session_id=self.machine.session_id)

    # ------------------------------------------------------------------
    # 各状态副作用
    # ------------------------------------------------------------------

    def _enter_idle(self, change: StateChange) -> None:
        self.timers.cancel(TimerName.SILENCE)
        self.timers.cancel(TimerName.THINKING)
        self.timers.cancel(TimerName.MAX_TURN)
        dropped = self._safe_cancel_playback(None)
        self.timers.schedule(TimerName.IDLE, self.idle_timeout_ms)
        log.debug("effect_enter_idle", reason=change.reason, dropped_ms=dropped)

    def _enter_listening(self, change: StateChange) -> None:
        self.timers.cancel(TimerName.IDLE)
        self.timers.cancel(TimerName.THINKING)
        self.timers.schedule(TimerName.SILENCE, self.silence_timeout_ms)
        self.timers.schedule(TimerName.MAX_TURN, self.max_turn_ms)
        self._pause_playback()

    def _exit_listening(self, change: StateChange) -> None:
        self.timers.cancel(TimerName.SILENCE)

    def _enter_thinking(self, change: StateChange) -> None:
        self.timers.cancel(TimerName.SILENCE)
        self.timers.schedule(TimerName.THINKING, self.thinking_timeout_ms)

    def _exit_thinking(self, change: StateChange) -> None:
        self.timers.cancel(TimerName.THINKING)

    def _enter_speaking(self, change: StateChange) -> None:
        """进入 speaking：放行播放队列。"""
        self.timers.cancel(TimerName.THINKING)
        self.timers.schedule(TimerName.MAX_TURN, self.max_turn_ms)
        self._resume_playback()

    def _exit_speaking(self, change: StateChange) -> None:
        """离开 speaking：清队列。被打断时 executor 已先清过一次，这里幂等兜底。"""
        dropped = self._safe_cancel_playback(None)
        self._pause_playback()
        if dropped:
            log.debug("effect_exit_speaking_dropped", dropped_ms=dropped, reason=change.reason)

    def _enter_interrupted(self, change: StateChange) -> None:
        self.timers.cancel(TimerName.THINKING)
        self.timers.cancel(TimerName.MAX_TURN)
        self._safe_cancel_playback(None)

    # ------------------------------------------------------------------

    def _emit_state(self, change: StateChange) -> None:
        if self.on_state_event is None:
            return
        try:
            result = self.on_state_event(change)
        except Exception as exc:  # noqa: BLE001
            log.error("state_event_emit_failed", error=str(exc))
            return
        if asyncio.iscoroutine(result):
            self._schedule_async(result, f"emit_state:{change.to_state}")

    def _schedule_async(self, coro: Awaitable[Any], name: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            coro.close()  # type: ignore[union-attr]
            return
        task = loop.create_task(coro, name=name)  # type: ignore[arg-type]
        task.add_done_callback(self._log_task_error)

    @staticmethod
    def _log_task_error(task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.error("transition_async_effect_failed", task=task.get_name(), error=str(exc))

    def _safe_cancel_playback(self, pts_ms: int | None) -> int:
        try:
            return int(self.playback.cancel_from(pts_ms))
        except Exception as exc:  # noqa: BLE001 - 播放队列异常不得阻断状态机
            log.warning("playback_cancel_failed", error=str(exc))
            return 0

    def _resume_playback(self) -> None:
        resume = getattr(self.playback, "resume", None)
        if callable(resume):
            try:
                resume()
            except Exception as exc:  # noqa: BLE001
                log.warning("playback_resume_failed", error=str(exc))

    def _pause_playback(self) -> None:
        pause = getattr(self.playback, "pause", None)
        if callable(pause):
            try:
                pause()
            except Exception as exc:  # noqa: BLE001
                log.warning("playback_pause_failed", error=str(exc))

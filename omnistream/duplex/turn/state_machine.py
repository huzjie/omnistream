"""TurnStateMachine —— 轮次状态机（迁移表已冻结，WS6 控制台按此渲染）。

迁移表（设计文档 §5.5，一字不改）::

    from \\ to     idle   listening   thinking   speaking   interrupted
    idle           —      VAD触发     文本输入   主动发起   ✗
    listening      静音超时  —        端点确认   ✗          ✗
    thinking       取消    打断        —         首音到达   打断
    speaking       播完    播完转听    工具回填   —          barge-in
    interrupted    ✓      立即转听     ✗         ✗          —

设计要点
--------
* ``transition()`` **必须同步、必须快**：barge-in 路径上不允许 await。
  异步副作用由回调返回协程，机器内部用 ``asyncio.create_task`` 调度（带 name）。
* 非法迁移抛 :class:`~omnistream.core.errors.DuplexError`（``OMS-5001``），
  调用方若允许失败请用 :meth:`try_transition`。
* 全量迁移历史保留在 :attr:`history`（有界 200 条），供延迟瀑布图与审计使用。
"""
from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

from omnistream.core.clock import now_ms, now_ns
from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import DuplexError
from omnistream.protocol.enums import TurnState

log = structlog.get_logger(__name__)

S = TurnState

#: 合法迁移表：``ALLOWED[from]`` = {to: 规范原因标签}
ALLOWED: dict[TurnState, dict[TurnState, str]] = {
    S.IDLE: {
        S.LISTENING: "vad_triggered",
        S.THINKING: "text_input",
        S.SPEAKING: "proactive",
    },
    S.LISTENING: {
        S.IDLE: "silence_timeout",
        S.THINKING: "endpoint_confirmed",
    },
    S.THINKING: {
        S.IDLE: "cancelled",
        S.LISTENING: "barge_in",
        S.SPEAKING: "first_audio",
        S.INTERRUPTED: "barge_in",
    },
    S.SPEAKING: {
        S.IDLE: "playback_finished",
        S.LISTENING: "playback_finished_to_listen",
        S.THINKING: "tool_result_continue",
        S.INTERRUPTED: "barge_in",
    },
    S.INTERRUPTED: {
        S.IDLE: "reset",
        S.LISTENING: "resume_listening",
    },
}

#: 终态集合（不再自动流转，需要外部驱动）
TERMINAL_STATES: frozenset[TurnState] = frozenset()

Callback = Callable[["StateChange"], Any]


@dataclass(slots=True, frozen=True)
class StateChange:
    """一次状态迁移的完整描述。"""

    from_state: TurnState
    to_state: TurnState
    reason: str
    ts_ms: int
    mono_ns: int
    dwell_ms: float
    turn_id: str | None = None
    session_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


class TurnStateMachine:
    """轮次状态机。一个会话一个实例。"""

    def __init__(
        self,
        *,
        session_id: str | None = None,
        initial: TurnState = TurnState.IDLE,
        history_size: int = 200,
    ) -> None:
        self.session_id = session_id
        self._state = initial
        self._entered_ns = now_ns()
        self._enter_cbs: dict[TurnState, list[Callback]] = {s: [] for s in TurnState}
        self._exit_cbs: dict[TurnState, list[Callback]] = {s: [] for s in TurnState}
        self._any_cbs: list[Callback] = []
        self._history: deque[StateChange] = deque(maxlen=max(1, int(history_size)))
        self._tasks: set[asyncio.Task[Any]] = set()
        self.turn_id: str | None = None

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    @property
    def state(self) -> TurnState:
        return self._state

    @property
    def history(self) -> list[StateChange]:
        return list(self._history)

    @property
    def dwell_ms(self) -> float:
        """在当前状态已停留的毫秒数（单调时钟）。"""
        return (now_ns() - self._entered_ns) / 1e6

    def can(self, to: TurnState) -> bool:
        """是否允许迁移到目标状态。"""
        return to in ALLOWED.get(self._state, {})

    def allowed_targets(self) -> list[TurnState]:
        return list(ALLOWED.get(self._state, {}).keys())

    def is_any(self, *states: TurnState) -> bool:
        return self._state in states

    # ------------------------------------------------------------------
    # 回调注册
    # ------------------------------------------------------------------

    def on_enter(self, state: TurnState, cb: Callback) -> None:
        """注册进入某状态的回调。回调可返回协程，将被自动调度。"""
        self._enter_cbs[state].append(cb)

    def on_exit(self, state: TurnState, cb: Callback) -> None:
        """注册离开某状态的回调。"""
        self._exit_cbs[state].append(cb)

    def on_any(self, cb: Callback) -> None:
        """注册任意迁移回调（用于下发 ``turn.state`` 事件与埋点）。"""
        self._any_cbs.append(cb)

    # ------------------------------------------------------------------
    # 迁移
    # ------------------------------------------------------------------

    def transition(self, to: TurnState, reason: str = "", *, force: bool = False,
                   detail: dict[str, Any] | None = None) -> TurnState:
        """执行迁移。

        Args:
            to: 目标状态。
            reason: 迁移原因；留空时使用迁移表中的规范标签。
            force: 绕过合法性检查（仅用于错误恢复/会话强制终止）。
            detail: 附加信息，写入 :class:`StateChange`。

        Returns:
            迁移后的状态。

        Raises:
            DuplexError: 非法迁移且未 force。
        """
        if to == self._state:
            return self._state

        table = ALLOWED.get(self._state, {})
        if to not in table and not force:
            raise DuplexError(
                f"非法轮次状态迁移: {self._state} -> {to}",
                code=ErrorCode.ILLEGAL_TURN_TRANSITION,
                detail={"from": str(self._state), "to": str(to),
                        "allowed": [str(s) for s in table], "reason": reason},
            )

        from_state = self._state
        change = StateChange(
            from_state=from_state,
            to_state=to,
            reason=reason or table.get(to, "forced" if force else "unspecified"),
            ts_ms=now_ms(),
            mono_ns=now_ns(),
            dwell_ms=self.dwell_ms,
            turn_id=self.turn_id,
            session_id=self.session_id,
            detail=dict(detail or {}),
        )

        self._state = to
        self._entered_ns = change.mono_ns
        self._history.append(change)

        log.info("turn_state_changed", session_id=self.session_id, turn_id=self.turn_id,
                 **{"from": str(from_state)}, to=str(to), reason=change.reason,
                 dwell_ms=round(change.dwell_ms, 2))

        self._fire(self._exit_cbs.get(from_state, ()), change)
        self._fire(self._enter_cbs.get(to, ()), change)
        self._fire(self._any_cbs, change)
        return self._state

    def try_transition(self, to: TurnState, reason: str = "",
                       detail: dict[str, Any] | None = None) -> bool:
        """尝试迁移；非法时返回 False 而不抛异常。"""
        if not self.can(to):
            log.debug("turn_transition_skipped", **{"from": str(self._state)},
                      to=str(to), reason=reason)
            return False
        self.transition(to, reason, detail=detail)
        return True

    def force_idle(self, reason: str = "session_reset") -> TurnState:
        """强制回到 idle（会话结束/致命错误时使用）。"""
        return self.transition(TurnState.IDLE, reason, force=True)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """等待所有被调度的异步回调结束（会话关闭时调用）。"""
        pending = [t for t in self._tasks if not t.done()]
        for task in pending:
            task.cancel()
        for task in pending:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _fire(self, callbacks: object, change: StateChange) -> None:
        for cb in tuple(callbacks or ()):  # type: ignore[arg-type]
            try:
                result = cb(change)
            except Exception as exc:  # noqa: BLE001 - 回调异常绝不能打断状态机
                log.error("turn_state_callback_failed", error=str(exc),
                          to=str(change.to_state), exc_info=True)
                continue
            if asyncio.iscoroutine(result):
                self._schedule(result, change)

    def _schedule(self, coro: Awaitable[Any], change: StateChange) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 无事件循环（同步测试/CLI），直接关闭协程避免 "never awaited" 警告
            coro.close()  # type: ignore[union-attr]
            return
        task = loop.create_task(coro, name=f"turn_cb:{change.to_state}:{self.session_id}")  # type: ignore[arg-type]
        self._tasks.add(task)
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.error("turn_state_async_callback_failed", error=str(exc), task=task.get_name())

    def __repr__(self) -> str:
        return f"<TurnStateMachine session={self.session_id} state={self._state}>"

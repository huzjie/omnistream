"""轮次与会话的各类超时定时器。

四类定时器（名称即 ``TimerName``）::

    silence   —— listening 态下长时间无语音 → 回落 idle
    thinking  —— thinking 态下 provider 迟迟不出首音 → 报超时并回落
    max_turn  —— 单轮最长时长（防跑飞）
    idle      —— 会话空闲超时 → 关闭会话
    proactive —— 主动发起的静默计时（可选）

实现要点：每个定时器一个 asyncio.Task（带 name，便于 ``oms doctor`` 排查），
``schedule`` 会先取消同名任务再重排，天然幂等。
"""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

import structlog

from omnistream.core.clock import now_ns

log = structlog.get_logger(__name__)


class TimerName(StrEnum):
    """定时器名称枚举。"""

    SILENCE = "silence"
    THINKING = "thinking"
    MAX_TURN = "max_turn"
    IDLE = "idle"
    PROACTIVE = "proactive"


FireCallback = Callable[[TimerName], Awaitable[None] | None]


class TimerSet:
    """一组命名定时器。"""

    def __init__(self, on_fire: FireCallback, *, session_id: str = "") -> None:
        self._on_fire = on_fire
        self.session_id = session_id
        self._tasks: dict[TimerName, asyncio.Task[None]] = {}
        self._deadlines: dict[TimerName, int] = {}

    # ------------------------------------------------------------------

    def schedule(self, name: TimerName, delay_ms: int) -> None:
        """（重新）排定一个定时器。``delay_ms <= 0`` 等价于取消。"""
        self.cancel(name)
        if delay_ms <= 0:
            return
        self._deadlines[name] = now_ns() + int(delay_ms) * 1_000_000
        task = asyncio.create_task(
            self._run(name, delay_ms / 1000.0),
            name=f"timer:{name}:{self.session_id}",
        )
        self._tasks[name] = task

    def cancel(self, name: TimerName) -> None:
        """取消一个定时器（幂等）。"""
        task = self._tasks.pop(name, None)
        self._deadlines.pop(name, None)
        if task is not None and not task.done():
            task.cancel()

    def cancel_all(self) -> None:
        """取消全部定时器。"""
        for name in list(self._tasks.keys()):
            self.cancel(name)

    def is_active(self, name: TimerName) -> bool:
        task = self._tasks.get(name)
        return task is not None and not task.done()

    def remaining_ms(self, name: TimerName) -> float:
        """剩余毫秒；未排定返回 0。"""
        deadline = self._deadlines.get(name)
        if deadline is None:
            return 0.0
        return max(0.0, (deadline - now_ns()) / 1e6)

    def active_timers(self) -> dict[str, float]:
        """当前活跃定时器与剩余时间，供 ``/healthz`` 与调试面板展示。"""
        return {str(name): round(self.remaining_ms(name), 1)
                for name in self._tasks
                if self.is_active(name)}

    async def aclose(self) -> None:
        """取消并等待所有定时器任务退出。"""
        tasks = list(self._tasks.values())
        self.cancel_all()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    # ------------------------------------------------------------------

    async def _run(self, name: TimerName, delay_s: float) -> None:
        try:
            await asyncio.sleep(delay_s)
        except asyncio.CancelledError:
            raise
        self._tasks.pop(name, None)
        self._deadlines.pop(name, None)
        log.debug("timer_fired", session_id=self.session_id, timer=str(name),
                  delay_ms=int(delay_s * 1000))
        try:
            result: Any = self._on_fire(name)
            if asyncio.iscoroutine(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 定时器回调异常不得杀死会话
            log.error("timer_callback_failed", session_id=self.session_id,
                      timer=str(name), error=str(exc), exc_info=True)

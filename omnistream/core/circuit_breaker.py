"""三态熔断器：closed -> open -> half_open -> closed。

用于包裹 provider 调用：某家上游连续失败时快速失败（抛 :class:`CircuitOpen`），
避免把用户请求全部堆在一个已经挂掉的上游上，同时按 ``recovery_timeout_s``
周期性放行少量探测请求以自动恢复。
"""
from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar

from omnistream.core.clock import monotonic_s, now_ms
from omnistream.core.errors import CircuitOpen
from omnistream.core.logging import get_logger

__all__ = ["CircuitBreaker", "CircuitState", "CircuitStats"]

log = get_logger(__name__)

R = TypeVar("R")


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(slots=True)
class CircuitStats:
    """熔断器可观测快照。"""

    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    successes: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    rejected: int = 0
    opened_count: int = 0
    last_error: str | None = None
    last_state_change_ms: int = field(default_factory=now_ms)

    @property
    def error_rate(self) -> float:
        total = self.failures + self.successes
        return (self.failures / total) if total else 0.0


class CircuitBreaker:
    """异步熔断器。

    参数语义：

    * ``failure_threshold``      —— closed 态连续失败多少次后打开；
    * ``recovery_timeout_s``     —— open 态维持多久后转 half_open 放行探测；
    * ``half_open_max_calls``    —— half_open 态最多并发放行几个探测请求；
    * ``success_threshold``      —— half_open 态连续成功多少次后关闭回 closed。
    """

    def __init__(
        self,
        name: str = "default",
        *,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 30.0,
        half_open_max_calls: int = 1,
        success_threshold: int = 2,
        window_size: int = 50,
    ) -> None:
        self.name = name
        self._failure_threshold = max(1, failure_threshold)
        self._recovery_timeout_s = max(0.1, recovery_timeout_s)
        self._half_open_max_calls = max(1, half_open_max_calls)
        self._success_threshold = max(1, success_threshold)
        self._state = CircuitState.CLOSED
        self._opened_at = 0.0
        self._half_open_inflight = 0
        self._lock = asyncio.Lock()
        self._recent: deque[bool] = deque(maxlen=max(5, window_size))
        self.stats = CircuitStats()

    # ------------------------------------------------------------- 状态
    @property
    def state(self) -> CircuitState:
        """当前状态。读取时会惰性完成 open -> half_open 的时间驱动迁移。"""
        if self._state is CircuitState.OPEN and self._recovery_elapsed():
            return CircuitState.HALF_OPEN
        return self._state

    def _recovery_elapsed(self) -> bool:
        return (monotonic_s() - self._opened_at) >= self._recovery_timeout_s

    def _set_state(self, new_state: CircuitState) -> None:
        if new_state is self._state:
            return
        previous, self._state = self._state, new_state
        self.stats.state = new_state
        self.stats.last_state_change_ms = now_ms()
        if new_state is CircuitState.OPEN:
            self._opened_at = monotonic_s()
            self.stats.opened_count += 1
        if new_state is CircuitState.CLOSED:
            self.stats.consecutive_failures = 0
        if new_state is CircuitState.HALF_OPEN:
            self._half_open_inflight = 0
            self.stats.consecutive_successes = 0
        log.warning("circuit_state_changed", circuit=self.name, from_state=str(previous), to_state=str(new_state))

    # ------------------------------------------------------------- 判定
    async def allow(self) -> bool:
        """是否放行一次调用。half_open 态会占用一个探测名额。"""
        async with self._lock:
            current = self.state
            if current is CircuitState.CLOSED:
                return True
            if current is CircuitState.OPEN:
                self.stats.rejected += 1
                return False
            if self._state is CircuitState.OPEN:
                self._set_state(CircuitState.HALF_OPEN)
            if self._half_open_inflight >= self._half_open_max_calls:
                self.stats.rejected += 1
                return False
            self._half_open_inflight += 1
            return True

    async def record_success(self) -> None:
        """登记一次成功调用。"""
        async with self._lock:
            self.stats.successes += 1
            self.stats.consecutive_failures = 0
            self.stats.consecutive_successes += 1
            self._recent.append(True)
            if self._state is CircuitState.HALF_OPEN:
                self._half_open_inflight = max(0, self._half_open_inflight - 1)
                if self.stats.consecutive_successes >= self._success_threshold:
                    self._set_state(CircuitState.CLOSED)

    async def record_failure(self, error: BaseException | None = None) -> None:
        """登记一次失败调用。"""
        async with self._lock:
            self.stats.failures += 1
            self.stats.consecutive_successes = 0
            self.stats.consecutive_failures += 1
            self.stats.last_error = str(error) if error is not None else None
            self._recent.append(False)
            if self._state is CircuitState.HALF_OPEN:
                self._half_open_inflight = max(0, self._half_open_inflight - 1)
                self._set_state(CircuitState.OPEN)
            elif self.stats.consecutive_failures >= self._failure_threshold:
                self._set_state(CircuitState.OPEN)

    async def reset(self) -> None:
        """强制复位为 closed（运维手动恢复用）。"""
        async with self._lock:
            self._half_open_inflight = 0
            self.stats.consecutive_failures = 0
            self.stats.consecutive_successes = 0
            self._set_state(CircuitState.CLOSED)

    # ------------------------------------------------------------- 调用
    async def call(self, fn: Callable[..., Awaitable[R]], *args: Any, **kwargs: Any) -> R:
        """在熔断保护下执行一个协程函数。

        熔断打开时抛 :class:`CircuitOpen`（HTTP 503 / ``OMS-6006``）。
        """
        if not await self.allow():
            raise CircuitOpen(
                f"熔断器 '{self.name}' 处于打开状态，暂不放行请求",
                provider=self.name,
                detail={
                    "circuit": self.name,
                    "state": str(self.state),
                    "consecutive_failures": self.stats.consecutive_failures,
                    "retry_after_s": round(
                        max(0.0, self._recovery_timeout_s - (monotonic_s() - self._opened_at)), 2
                    ),
                },
            )
        try:
            result = await fn(*args, **kwargs)
        except asyncio.CancelledError:
            await self.record_success()  # 主动取消不算上游故障
            raise
        except BaseException as exc:  # noqa: BLE001 - 需要登记后再抛
            await self.record_failure(exc)
            raise
        await self.record_success()
        return result

    def snapshot(self) -> dict[str, Any]:
        """导出可直接进 Prometheus / 健康检查响应的快照。"""
        return {
            "name": self.name,
            "state": str(self.state),
            "failures": self.stats.failures,
            "successes": self.stats.successes,
            "consecutive_failures": self.stats.consecutive_failures,
            "rejected": self.stats.rejected,
            "opened_count": self.stats.opened_count,
            "error_rate": round(self.stats.error_rate, 4),
            "window_error_rate": round(
                (self._recent.count(False) / len(self._recent)) if self._recent else 0.0, 4
            ),
            "last_error": self.stats.last_error,
            "last_state_change_ms": self.stats.last_state_change_ms,
        }

    def __repr__(self) -> str:
        return f"<CircuitBreaker name={self.name} state={self.state}>"

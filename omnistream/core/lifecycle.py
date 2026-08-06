"""生命周期管理：有序 startup / shutdown 钩子，带超时强杀。

FastAPI 的 lifespan、Celery worker 的 bootstrap、CLI 长驻命令共用这一套。
关闭顺序 = 启动顺序的逆序（后启动的先关），与 LIFO 资源释放语义一致。
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from omnistream.core.clock import monotonic_s
from omnistream.core.constants import DEFAULT_SHUTDOWN_TIMEOUT_S
from omnistream.core.logging import get_logger

__all__ = ["Hook", "LifecycleManager"]

log = get_logger(__name__)

HookFn = Callable[[], Any] | Callable[[], Awaitable[Any]]


@dataclass(slots=True)
class Hook:
    """一个生命周期钩子。``order`` 小者先启动、后关闭。"""

    name: str
    fn: HookFn
    order: int = 100
    timeout_s: float = 10.0
    critical: bool = True
    tags: tuple[str, ...] = field(default_factory=tuple)


class LifecycleManager:
    """有序执行 startup / shutdown 钩子。

    ::

        lifecycle = LifecycleManager()
        lifecycle.on_startup("db", connect_db, order=10)
        lifecycle.on_shutdown("db", close_db, order=10)

        await lifecycle.startup()
        ...
        await lifecycle.shutdown()

    * ``critical=True`` 的启动钩子失败会中止启动并回滚已完成的部分；
    * ``critical=False`` 的钩子失败只记 warning（如可观测性上报器）；
    * 关闭钩子的异常一律不向外抛，保证进程能干净退出。
    """

    def __init__(self, *, shutdown_timeout_s: float = DEFAULT_SHUTDOWN_TIMEOUT_S) -> None:
        self._startup: list[Hook] = []
        self._shutdown: list[Hook] = []
        self._started: list[Hook] = []
        self._shutdown_timeout_s = shutdown_timeout_s
        self._state: str = "idle"

    # ------------------------------------------------------------- 注册
    def on_startup(self, name: str, fn: HookFn, *, order: int = 100,
                   timeout_s: float = 10.0, critical: bool = True,
                   tags: tuple[str, ...] = ()) -> None:
        self._startup.append(Hook(name=name, fn=fn, order=order, timeout_s=timeout_s,
                                  critical=critical, tags=tags))

    def on_shutdown(self, name: str, fn: HookFn, *, order: int = 100,
                    timeout_s: float = 10.0, tags: tuple[str, ...] = ()) -> None:
        self._shutdown.append(Hook(name=name, fn=fn, order=order, timeout_s=timeout_s,
                                   critical=False, tags=tags))

    @property
    def state(self) -> str:
        """``idle`` | ``starting`` | ``running`` | ``stopping`` | ``stopped`` | ``failed``。"""
        return self._state

    @property
    def running(self) -> bool:
        return self._state == "running"

    # ------------------------------------------------------------- 执行
    async def startup(self) -> None:
        """按 order 升序执行全部启动钩子。任一 critical 钩子失败则回滚并抛出。"""
        self._state = "starting"
        self._started.clear()
        for hook in sorted(self._startup, key=lambda h: (h.order, h.name)):
            began = monotonic_s()
            try:
                await self._invoke(hook)
            except Exception as exc:
                elapsed_ms = round((monotonic_s() - began) * 1000, 2)
                if hook.critical:
                    log.error("lifecycle_startup_failed", hook=hook.name,
                              elapsed_ms=elapsed_ms, error=str(exc))
                    self._state = "failed"
                    await self._rollback()
                    raise
                log.warning("lifecycle_startup_degraded", hook=hook.name,
                            elapsed_ms=elapsed_ms, error=str(exc))
                continue
            self._started.append(hook)
            log.info("lifecycle_startup_ok", hook=hook.name,
                     elapsed_ms=round((monotonic_s() - began) * 1000, 2))
        self._state = "running"

    async def shutdown(self) -> None:
        """按 order 降序执行全部关闭钩子。异常只记录，不外抛。"""
        if self._state in ("stopped", "idle"):
            return
        self._state = "stopping"
        deadline = monotonic_s() + self._shutdown_timeout_s
        for hook in sorted(self._shutdown, key=lambda h: (h.order, h.name), reverse=True):
            remaining = deadline - monotonic_s()
            if remaining <= 0:
                log.warning("lifecycle_shutdown_budget_exhausted", hook=hook.name)
                break
            began = monotonic_s()
            try:
                await self._invoke(hook, timeout_s=min(hook.timeout_s, remaining))
            except Exception as exc:
                log.warning("lifecycle_shutdown_failed", hook=hook.name, error=str(exc))
            else:
                log.info("lifecycle_shutdown_ok", hook=hook.name,
                         elapsed_ms=round((monotonic_s() - began) * 1000, 2))
        self._started.clear()
        self._state = "stopped"

    async def _rollback(self) -> None:
        """启动失败时，逆序关闭已经成功启动的部分。"""
        started_names = {hook.name for hook in self._started}
        for hook in sorted(self._shutdown, key=lambda h: (h.order, h.name), reverse=True):
            if hook.name not in started_names:
                continue
            with contextlib.suppress(Exception):
                await self._invoke(hook, timeout_s=hook.timeout_s)
                log.info("lifecycle_rollback_ok", hook=hook.name)
        self._started.clear()

    @staticmethod
    async def _invoke(hook: Hook, *, timeout_s: float | None = None) -> None:
        budget = hook.timeout_s if timeout_s is None else timeout_s
        async with asyncio.timeout(budget):
            result = hook.fn()
            if inspect.isawaitable(result):
                await result

    async def __aenter__(self) -> LifecycleManager:
        await self.startup()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.shutdown()

"""asyncio 工具箱：实时管线里反复用到的并发原语。

全双工场景对并发的要求和普通 Web 服务不同：

* 任何一路协程崩溃都**不能**静默丢失（否则会话会「假死」）→ :func:`fire_and_forget`
  强制持有引用并记录异常；
* barge-in 要求 100ms 内真正停下 → :func:`cancel_and_wait` 保证等到任务真的结束；
* 上行音频不能因为下游慢而无限堆积 → :class:`Debouncer` / :func:`gather_limited`
  提供背压与并发上限。
"""
from __future__ import annotations

import asyncio
import contextlib
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterable
from typing import Any, TypeVar

from omnistream.core.clock import monotonic_s
from omnistream.core.logging import get_logger

__all__ = [
    "Debouncer",
    "PeriodicTask",
    "cancel_and_wait",
    "fire_and_forget",
    "first_completed",
    "gather_limited",
    "iter_with_timeout",
    "run_in_thread",
    "sleep_until",
    "with_timeout",
]

T = TypeVar("T")

log = get_logger(__name__)

#: 强引用池，防止 fire-and-forget 任务被 GC 提前回收（CPython 已知陷阱）。
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


def fire_and_forget(coro: Coroutine[Any, Any, Any], *, name: str = "background") -> asyncio.Task[Any]:
    """启动一个后台任务，保持强引用并在异常时记录日志。

    Args:
        coro: 待运行的协程对象。
        name: 任务名，会出现在日志与 ``asyncio`` 调试输出里。

    Returns:
        已调度的 ``asyncio.Task``。调用方通常不需要 await 它。
    """
    task = asyncio.create_task(coro, name=name)
    _BACKGROUND_TASKS.add(task)

    def _done(finished: asyncio.Task[Any]) -> None:
        _BACKGROUND_TASKS.discard(finished)
        if finished.cancelled():
            return
        error = finished.exception()
        if error is not None:
            log.error("background_task_failed", task=name, error=str(error),
                      error_type=type(error).__name__)

    task.add_done_callback(_done)
    return task


async def cancel_and_wait(task: asyncio.Task[Any] | None, *, timeout_s: float = 1.0) -> bool:
    """取消任务并**确认**它真的结束了。

    barge-in 路径必须用这个而不是裸 ``task.cancel()``——后者只是投递取消请求，
    协程可能还在吐音频帧。

    Returns:
        ``True`` 表示已在预算内结束，``False`` 表示超时（调用方应告警）。
    """
    if task is None or task.done():
        return True
    task.cancel()
    try:
        async with asyncio.timeout(timeout_s):
            with contextlib.suppress(asyncio.CancelledError):
                await task
    except TimeoutError:
        log.warning("cancel_timeout", task=task.get_name(), timeout_s=timeout_s)
        return False
    return True


async def with_timeout(awaitable: Awaitable[T], timeout_s: float, *,
                       default: T | None = None) -> T | None:
    """带超时地 await，超时返回 ``default`` 而不是抛异常。"""
    try:
        async with asyncio.timeout(timeout_s):
            return await awaitable
    except TimeoutError:
        return default


async def gather_limited(
    coros: Iterable[Coroutine[Any, Any, T]],
    *,
    limit: int = 8,
    return_exceptions: bool = False,
) -> list[T]:
    """带并发上限的 ``gather``，结果顺序与输入一致。

    Args:
        coros: 协程序列。
        limit: 最大并发数，必须 ≥ 1。
        return_exceptions: 为 True 时异常作为结果返回而不是向外抛。
    """
    if limit < 1:
        raise ValueError("limit must be >= 1")
    semaphore = asyncio.Semaphore(limit)

    async def _guarded(coro: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(_guarded(c) for c in coros),
                                return_exceptions=return_exceptions)  # type: ignore[return-value]


async def first_completed(*awaitables: Awaitable[T],
                          timeout_s: float | None = None) -> T | None:
    """返回最先完成者的结果，其余全部取消（并等待其真正结束）。"""
    tasks = [asyncio.ensure_future(a) for a in awaitables]
    if not tasks:
        return None
    try:
        done, pending = await asyncio.wait(
            tasks, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        pass
    for task in tasks:
        if not task.done():
            task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    if not done:
        return None
    return next(iter(done)).result()


async def sleep_until(deadline_monotonic_s: float) -> None:
    """睡到指定的单调时钟时刻（已过期立即返回）。"""
    remaining = deadline_monotonic_s - monotonic_s()
    if remaining > 0:
        await asyncio.sleep(remaining)


async def run_in_thread(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """在默认线程池里跑阻塞函数（numpy 重计算、本地模型推理、磁盘 IO）。"""
    if kwargs:
        def _call() -> T:
            return fn(*args, **kwargs)

        return await asyncio.to_thread(_call)
    return await asyncio.to_thread(fn, *args)


async def iter_with_timeout(
    source: AsyncIterator[T],
    idle_timeout_s: float,
) -> AsyncIterator[T]:
    """给异步迭代器加「相邻两个元素之间」的空闲超时。

    Provider 的事件流卡死时，靠它把会话从 ``responding`` 拽回 ``idle``，
    而不是让用户对着沉默的麦克风干等。超时抛 ``TimeoutError``。
    """
    while True:
        try:
            async with asyncio.timeout(idle_timeout_s):
                item = await anext(source)
        except StopAsyncIteration:
            return
        yield item


class Debouncer:
    """防抖器：在 ``wait_s`` 内的重复触发只执行最后一次。

    典型用途：视频帧「场景变化」检测抖动、前端配置面板的实时保存。

    ::

        debouncer = Debouncer(0.3)
        debouncer.trigger(lambda: save(config))   # 300ms 内多次调用只保存一次
    """

    def __init__(self, wait_s: float, *, name: str = "debounce") -> None:
        if wait_s < 0:
            raise ValueError("wait_s must be >= 0")
        self._wait_s = wait_s
        self._name = name
        self._task: asyncio.Task[Any] | None = None

    def trigger(self, fn: Callable[[], Any]) -> None:
        """调度一次执行，取消尚未触发的上一次。"""
        if self._task is not None and not self._task.done():
            self._task.cancel()

        async def _runner() -> None:
            await asyncio.sleep(self._wait_s)
            result = fn()
            if asyncio.iscoroutine(result):
                await result

        self._task = fire_and_forget(_runner(), name=self._name)

    async def flush(self) -> None:
        """立刻等待挂起的那次执行完成（若有）。"""
        if self._task is not None and not self._task.done():
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def cancel(self) -> None:
        """丢弃挂起的执行。"""
        await cancel_and_wait(self._task)
        self._task = None


class PeriodicTask:
    """周期任务：固定间隔执行，支持抖动、异常自愈与优雅停止。

    健康探测、指标上报、会话空闲清理都用它，避免每处再写一遍 ``while True``。
    """

    def __init__(
        self,
        interval_s: float,
        fn: Callable[[], Any],
        *,
        name: str = "periodic",
        jitter: float = 0.1,
        run_immediately: bool = False,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self._interval_s = interval_s
        self._fn = fn
        self._name = name
        self._jitter = max(0.0, min(jitter, 1.0))
        self._run_immediately = run_immediately
        self._task: asyncio.Task[Any] | None = None
        self._stopping = asyncio.Event()
        self.runs: int = 0
        self.failures: int = 0

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """启动循环（幂等）。"""
        if self.running:
            return
        self._stopping.clear()
        self._task = fire_and_forget(self._loop(), name=self._name)

    async def stop(self, *, timeout_s: float = 2.0) -> None:
        """请求停止并等待当前这轮跑完。"""
        self._stopping.set()
        await cancel_and_wait(self._task, timeout_s=timeout_s)
        self._task = None

    def _next_delay(self) -> float:
        if self._jitter <= 0:
            return self._interval_s
        spread = self._interval_s * self._jitter
        return max(0.0, self._interval_s + random.uniform(-spread, spread))

    async def _loop(self) -> None:
        if not self._run_immediately:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(self._next_delay()):
                    await self._stopping.wait()
        while not self._stopping.is_set():
            try:
                result = self._fn()
                if asyncio.iscoroutine(result):
                    await result
                self.runs += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.failures += 1
                log.warning("periodic_task_failed", task=self._name, error=str(exc),
                            failures=self.failures)
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(self._next_delay()):
                    await self._stopping.wait()

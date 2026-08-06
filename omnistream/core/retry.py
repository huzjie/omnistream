"""异步重试：指数退避 + 抖动 + 可重试异常白名单。

设计取舍：实时链路上的重试必须"快失败"，因此默认 ``max_attempts=3`` 且
``max_delay_s=2.0``；离线任务可以自行放宽。``OmniError.retryable``（由错误码表
驱动）是默认的判定依据，避免每个调用点各写一套白名单。
"""
from __future__ import annotations

import asyncio
import functools
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, ParamSpec, TypeVar

from omnistream.core.errors import OmniError
from omnistream.core.logging import get_logger

__all__ = ["RetryPolicy", "retry_async", "retry_call"]

log = get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(slots=True)
class RetryPolicy:
    """重试策略。"""

    max_attempts: int = 3
    base_delay_s: float = 0.2
    max_delay_s: float = 2.0
    multiplier: float = 2.0
    #: 抖动比例，0.2 表示在计算出的延迟上叠加 ±20% 随机量，避免惊群。
    jitter: float = 0.2
    #: 额外的可重试异常类型；``OmniError`` 走 ``retryable`` 属性判定。
    retry_on: tuple[type[BaseException], ...] = (asyncio.TimeoutError, ConnectionError, OSError)
    #: 明确不重试的异常类型，优先级高于 ``retry_on``。
    give_up_on: tuple[type[BaseException], ...] = ()

    def delay_for(self, attempt: int) -> float:
        """第 ``attempt`` 次（从 1 开始）失败后应等待的秒数。"""
        raw = self.base_delay_s * (self.multiplier ** max(0, attempt - 1))
        capped = min(raw, self.max_delay_s)
        if self.jitter <= 0:
            return capped
        span = capped * self.jitter
        return max(0.0, capped + random.uniform(-span, span))

    def should_retry(self, exc: BaseException) -> bool:
        """判断该异常是否值得重试。"""
        if isinstance(exc, asyncio.CancelledError):
            return False
        if self.give_up_on and isinstance(exc, self.give_up_on):
            return False
        if isinstance(exc, OmniError):
            return exc.retryable
        return isinstance(exc, self.retry_on)


DEFAULT_POLICY = RetryPolicy()


async def retry_call(
    fn: Callable[..., Awaitable[R]],
    *args: Any,
    policy: RetryPolicy | None = None,
    op: str = "",
    **kwargs: Any,
) -> R:
    """按策略重试执行一个协程函数，返回其结果。

    最后一次尝试仍失败时原样抛出最后一个异常（保留 traceback 与 ``code``）。
    """
    active = policy or DEFAULT_POLICY
    label = op or getattr(fn, "__qualname__", "anonymous")
    last: BaseException | None = None
    for attempt in range(1, max(1, active.max_attempts) + 1):
        try:
            return await fn(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - 由 policy 决定去留
            last = exc
            if attempt >= active.max_attempts or not active.should_retry(exc):
                raise
            delay = active.delay_for(attempt)
            log.warning(
                "retry_scheduled",
                op=label,
                attempt=attempt,
                max_attempts=active.max_attempts,
                delay_ms=round(delay * 1000, 1),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            await asyncio.sleep(delay)
    assert last is not None  # noqa: S101 - 逻辑上不可达，仅安抚类型检查
    raise last


def retry_async(
    max_attempts: int = 3,
    *,
    base_delay_s: float = 0.2,
    max_delay_s: float = 2.0,
    multiplier: float = 2.0,
    jitter: float = 0.2,
    retry_on: tuple[type[BaseException], ...] = (asyncio.TimeoutError, ConnectionError, OSError),
    give_up_on: tuple[type[BaseException], ...] = (),
    op: str = "",
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """协程函数重试装饰器。

    ::

        @retry_async(max_attempts=4, base_delay_s=0.3)
        async def fetch_token() -> str:
            ...
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        base_delay_s=base_delay_s,
        max_delay_s=max_delay_s,
        multiplier=multiplier,
        jitter=jitter,
        retry_on=retry_on,
        give_up_on=give_up_on,
    )

    def decorator(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        label = op or getattr(fn, "__qualname__", "anonymous")

        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return await retry_call(fn, *args, policy=policy, op=label, **kwargs)

        return wrapper

    return decorator

"""会话工厂与作用域管理。

* ``session_scope()``  —— 供后台任务 / worker 使用的 async 上下文管理器（自动 commit/rollback）
* ``get_db()``         —— FastAPI 依赖项，WS1 的 ``api/deps.py`` 直接转发到这里
* ``run_sync_session`` —— Celery 同步任务里安全跑一段 async DB 逻辑
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from omnistream.db.engine import get_engine

log = structlog.get_logger(__name__)

T = TypeVar("T")

_session_factory: async_sessionmaker[AsyncSession] | None = None


def async_session_factory() -> async_sessionmaker[AsyncSession]:
    """全局会话工厂（懒构建，绑定全局 engine）。"""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
    return _session_factory


def reset_session_factory() -> None:
    """重置工厂（切换 engine 后调用，例如测试或 CLI 指定了别的库）。"""
    global _session_factory
    _session_factory = None


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """事务作用域：正常退出 commit，异常 rollback，始终 close。"""
    factory = async_session_factory()
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


@asynccontextmanager
async def readonly_scope() -> AsyncIterator[AsyncSession]:
    """只读作用域：结束时统一 rollback，避免误写。"""
    factory = async_session_factory()
    session = factory()
    try:
        yield session
    finally:
        await session.rollback()
        await session.close()


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI 依赖项：每请求一个会话，请求结束自动收尾。

    用法::

        from fastapi import Depends
        from omnistream.db.session import get_db

        @router.get("/things")
        async def list_things(db: AsyncSession = Depends(get_db)) -> dict:
            ...
    """
    factory = async_session_factory()
    session = factory()
    try:
        yield session
        if session.in_transaction():
            await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


def run_sync(coro_factory: Callable[[], Awaitable[T]]) -> T:
    """在同步上下文（Celery worker）里跑一段协程。

    已有运行中的事件循环时使用独立线程 + 新循环，避免 ``asyncio.run`` 抛错。
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    import concurrent.futures

    def _runner() -> T:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro_factory())
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_runner).result()


def run_with_session(func: Callable[[AsyncSession], Awaitable[T]]) -> T:
    """同步入口：开一个 session 作用域执行 ``func``。"""

    async def _inner() -> T:
        async with session_scope() as session:
            return await func(session)

    return run_sync(_inner)


async def healthcheck() -> dict[str, Any]:
    """给 /readyz 用的数据库健康信息。"""
    from omnistream.db.engine import current_engine_url, ping

    ok = await ping()
    url = current_engine_url()
    # 不回显凭据
    safe = url
    if "@" in safe:
        head, _, tail = safe.partition("://")
        _, _, hostpart = tail.partition("@")
        safe = f"{head}://***@{hostpart}"
    return {"ok": ok, "url": safe}


__all__ = [
    "async_session_factory",
    "get_db",
    "healthcheck",
    "readonly_scope",
    "reset_session_factory",
    "run_sync",
    "run_with_session",
    "session_scope",
]

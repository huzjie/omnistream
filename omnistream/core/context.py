"""请求 / 会话级上下文（contextvars）。

这些变量由中间件与会话编排层写入，日志系统（``core/logging.py``）自动读取并
注入到每条结构化日志，业务代码**不需要**手工传递。
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

__all__ = [
    "bind_context",
    "clear_context",
    "current_request_id",
    "current_session_id",
    "current_span_id",
    "current_tenant_id",
    "current_trace_id",
    "current_user_id",
    "get_context",
    "get_request_id",
    "get_session_id",
    "get_tenant_id",
    "set_request_id",
    "set_session_id",
    "set_tenant_id",
]

current_request_id: ContextVar[str | None] = ContextVar("current_request_id", default=None)
current_trace_id: ContextVar[str | None] = ContextVar("current_trace_id", default=None)
current_span_id: ContextVar[str | None] = ContextVar("current_span_id", default=None)
current_tenant_id: ContextVar[str | None] = ContextVar("current_tenant_id", default=None)
current_user_id: ContextVar[str | None] = ContextVar("current_user_id", default=None)
current_session_id: ContextVar[str | None] = ContextVar("current_session_id", default=None)

_ALL_VARS: dict[str, ContextVar[str | None]] = {
    "request_id": current_request_id,
    "trace_id": current_trace_id,
    "span_id": current_span_id,
    "tenant_id": current_tenant_id,
    "user_id": current_user_id,
    "session_id": current_session_id,
}


def get_context() -> dict[str, str]:
    """当前全部已设置的上下文字段（值为 None 的键会被剔除）。"""
    return {key: var.get() or "" for key, var in _ALL_VARS.items() if var.get()}


def set_request_id(value: str | None) -> Token[str | None]:
    return current_request_id.set(value)


def set_tenant_id(value: str | None) -> Token[str | None]:
    return current_tenant_id.set(value)


def set_session_id(value: str | None) -> Token[str | None]:
    return current_session_id.set(value)


def get_request_id() -> str | None:
    return current_request_id.get()


def get_tenant_id() -> str | None:
    return current_tenant_id.get()


def get_session_id() -> str | None:
    return current_session_id.get()


@contextmanager
def bind_context(**values: Any) -> Iterator[None]:
    """临时绑定若干上下文字段，退出时精确还原。

    ::

        with bind_context(session_id=sid, tenant_id=tid):
            log.info("turn_started")
    """
    tokens: list[tuple[ContextVar[str | None], Token[str | None]]] = []
    try:
        for key, raw in values.items():
            var = _ALL_VARS.get(key)
            if var is None:
                continue
            tokens.append((var, var.set(None if raw is None else str(raw))))
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def clear_context() -> None:
    """清空全部上下文字段（长驻 worker 每个任务开头调用）。"""
    for var in _ALL_VARS.values():
        var.set(None)

"""structlog 初始化。

两种渲染模式：

* ``json``    —— 生产环境，一行一个 JSON 对象，直接被 Loki / ES 摄取；
* ``console`` —— 开发环境，彩色可读。

所有日志自动携带 ``core/context.py`` 里的 contextvars（request_id / trace_id /
span_id / tenant_id / session_id），业务代码无需手写。
"""
from __future__ import annotations

import logging
import logging.config
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor, WrappedLogger

from omnistream.__version__ import __version__
from omnistream.core.context import get_context

__all__ = ["bind_logger", "configure_logging", "get_logger", "is_configured"]

_CONFIGURED = False

#: 这些第三方 logger 太吵，统一压到 WARNING。
_NOISY_LOGGERS: tuple[str, ...] = (
    "aioice",
    "aiortc",
    "aiosqlite",
    "asyncio",
    "botocore",
    "boto3",
    "httpcore",
    "httpx",
    "multipart",
    "urllib3",
    "watchfiles",
)


def _inject_context(_logger: WrappedLogger, _name: str, event_dict: EventDict) -> EventDict:
    """把 contextvars 合并进日志事件（已存在的显式 kwargs 优先）。"""
    for key, value in get_context().items():
        event_dict.setdefault(key, value)
    return event_dict


def _inject_service(_logger: WrappedLogger, _name: str, event_dict: EventDict) -> EventDict:
    event_dict.setdefault("service", "omnistream")
    event_dict.setdefault("version", __version__)
    return event_dict


def _rename_event_key(_logger: WrappedLogger, _name: str, event_dict: EventDict) -> EventDict:
    """把 structlog 的 ``event`` 键改名为 ``msg``，避免与"业务事件"语义混淆。"""
    if "event" in event_dict:
        event_dict["msg"] = event_dict.pop("event")
    return event_dict


def _drop_color_message(_logger: WrappedLogger, _name: str, event_dict: EventDict) -> EventDict:
    """uvicorn 会塞一个重复的 ``color_message``，JSON 模式下丢掉。"""
    event_dict.pop("color_message", None)
    return event_dict


def configure_logging(
    level: str = "INFO",
    *,
    fmt: str = "json",
    utc: bool = True,
    force: bool = False,
) -> None:
    """初始化 structlog 与标准库 logging。进程内只需调用一次。

    Args:
        level: 根日志级别，``DEBUG`` / ``INFO`` / ``WARNING`` / ``ERROR``。
        fmt: ``json`` 或 ``console``。
        utc: 时间戳是否用 UTC（生产恒为 True）。
        force: 为 True 时允许重复初始化（测试或 CLI 二次配置用）。
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    use_json = fmt.lower() == "json"

    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        _inject_context,
        _inject_service,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=utc),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _drop_color_message,
    ]

    if use_json:
        renderer: Processor = structlog.processors.JSONRenderer(ensure_ascii=False)
        final: list[Processor] = [
            structlog.processors.format_exc_info,
            _rename_event_key,
            renderer,
        ]
    else:
        final = [
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty(), exception_formatter=structlog.dev.plain_traceback)
        ]

    structlog.configure(
        processors=[*shared, *final],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # 标准库 logging 走同一个渲染管线，保证 uvicorn / sqlalchemy 日志格式一致。
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, *final],
    )
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(numeric_level)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(numeric_level, logging.WARNING))
    # uvicorn 自带 handler 会造成重复输出，交给 root 统一处理。
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True

    _CONFIGURED = True


def is_configured() -> bool:
    """日志系统是否已初始化。"""
    return _CONFIGURED


def get_logger(name: str | None = None) -> Any:
    """获取一个 structlog logger。未初始化时按默认参数自动初始化。"""
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name) if name else structlog.get_logger()


def bind_logger(**values: Any) -> Any:
    """返回一个预绑定若干字段的 logger。"""
    return get_logger().bind(**values)

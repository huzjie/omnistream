"""HTTP 中间件集合与统一装配入口。

装配顺序很重要。Starlette 的 ``add_middleware`` 是 **后加先执行**（洋葱最外层是最后
添加的那一个），因此本模块按「希望的执行顺序」倒序添加：

1. ``RequestIdMiddleware``   —— 最外层，最先拿到请求、最后返回响应，保证任何异常
   （包括后面中间件抛出的）都带上 ``X-Request-Id``；
2. ``CORSMiddleware``        —— 次外层，保证 4xx/5xx 响应同样带 CORS 头，
   否则浏览器只会看到不透明的网络错误；
3. ``AccessLogMiddleware``   —— 最内层，记录真实的业务处理耗时。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from omnistream.api.middleware.cors import install_cors, resolve_origins
from omnistream.api.middleware.logging import AccessLogMiddleware
from omnistream.api.middleware.request_id import RequestIdMiddleware

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

    from omnistream.config.settings import Settings

__all__ = [
    "AccessLogMiddleware",
    "RequestIdMiddleware",
    "install_cors",
    "install_middlewares",
    "resolve_origins",
]


def install_middlewares(app: "FastAPI", settings: "Settings") -> list[str]:
    """把全部标准中间件装到 ``app`` 上，返回实际执行顺序的名字列表。

    Args:
        app: 目标 FastAPI 应用。
        settings: 已解析的全局配置。

    Returns:
        按实际执行顺序（从外到内）排列的中间件名称，便于启动日志打印与自检。
    """
    order: list[str] = []

    # --- 倒序添加：最内层先加 ---
    if settings.observability.access_log:
        app.add_middleware(
            AccessLogMiddleware,
            slow_request_ms=settings.observability.slow_request_ms,
        )

    install_cors(app, settings)

    app.add_middleware(RequestIdMiddleware)

    # --- 正序汇报 ---
    order.append("RequestIdMiddleware")
    if resolve_origins(settings):
        order.append("CORSMiddleware")
    if settings.observability.access_log:
        order.append("AccessLogMiddleware")
    return order

"""访问日志中间件：结构化输出方法、路径、状态码、耗时与租户。"""
from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

log = structlog.get_logger("omnistream.access")

#: 这些路径不打访问日志，避免探针刷屏
_QUIET_PATHS = frozenset({"/healthz", "/readyz", "/metrics", "/favicon.ico"})


class AccessLogMiddleware(BaseHTTPMiddleware):
    """记录每个 HTTP 请求的结构化访问日志。"""

    def __init__(self, app: ASGIApp, *, slow_ms: int = 1000,
                 quiet_paths: frozenset[str] = _QUIET_PATHS) -> None:
        super().__init__(app)
        self.slow_ms = slow_ms
        self.quiet_paths = quiet_paths

    async def dispatch(self, request: Request,
                       call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        path = request.url.path
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if path not in self.quiet_paths:
                client = request.client.host if request.client else None
                fields = {
                    "method": request.method,
                    "path": path,
                    "status": status,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "client_ip": client,
                    "tenant_id": getattr(request.state, "tenant_id", None),
                }
                if status >= 500:
                    log.error("http_request", **fields)
                elif status >= 400 or elapsed_ms >= self.slow_ms:
                    log.warning("http_request", **fields)
                else:
                    log.info("http_request", **fields)

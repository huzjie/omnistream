"""请求 ID 中间件：生成/透传 ``X-Request-Id`` 并注入 contextvar。

所有结构化日志会自动带上该 ID（见 ``core.logging._inject_context``），
错误响应体也会回填，便于用户报障时一键定位。
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from omnistream.core.constants import REQUEST_ID_HEADER
from omnistream.core.context import set_request_id, set_tenant_id
from omnistream.core.ids import new_id

_TENANT_HEADER = "X-Tenant-Id"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """为每个请求分配唯一 ID 并写入响应头。"""

    def __init__(self, app: ASGIApp, *, header: str = REQUEST_ID_HEADER,
                 tenant_header: str = _TENANT_HEADER) -> None:
        super().__init__(app)
        self.header = header
        self.tenant_header = tenant_header

    async def dispatch(self, request: Request,
                       call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        incoming = request.headers.get(self.header, "").strip()
        request_id = incoming if 8 <= len(incoming) <= 128 else new_id("req")

        tenant_id = request.headers.get(self.tenant_header, "").strip() or None

        rid_token = set_request_id(request_id)
        tid_token = set_tenant_id(tenant_id)
        request.state.request_id = request_id
        request.state.tenant_id = tenant_id
        try:
            response = await call_next(request)
        finally:
            # contextvar Token 必须在同一上下文中归还，避免污染后续请求
            try:
                rid_token.var.reset(rid_token)
                tid_token.var.reset(tid_token)
            except (ValueError, LookupError):
                pass
        response.headers[self.header] = request_id
        return response

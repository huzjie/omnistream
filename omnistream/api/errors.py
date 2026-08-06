"""全局异常处理器：把任何异常统一成 ``{code, message, detail, request_id}``。

对外契约（HTTP 与 WS ``error`` 事件同构）::

    {
      "code": "OMS-4001",
      "message": "会话不存在",
      "detail": {"session_id": "sess_..."},
      "request_id": "req_...",
      "retryable": false
    }
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from omnistream.core.context import get_context
from omnistream.core.error_codes import ErrorCode, http_status_for, is_retryable
from omnistream.core.errors import OmniError

if TYPE_CHECKING:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

log = structlog.get_logger(__name__)


def error_body(code: ErrorCode, message: str, *,
               detail: dict[str, Any] | None = None,
               request_id: str | None = None) -> dict[str, Any]:
    """构造统一错误响应体。"""
    ctx = get_context()
    return {
        "code": str(code),
        "message": message,
        "detail": detail or {},
        "request_id": request_id or ctx.get("request_id"),
        "session_id": ctx.get("session_id"),
        "retryable": is_retryable(code),
    }


def _json_response(status: int, payload: dict[str, Any]) -> "JSONResponse":
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status, content=payload)


async def omni_error_handler(request: "Request", exc: Exception) -> "JSONResponse":
    """``OmniError`` → 结构化响应，HTTP 状态由错误码映射表决定。"""
    assert isinstance(exc, OmniError)
    status = http_status_for(exc.code)
    payload = error_body(exc.code, exc.message, detail=exc.detail)
    log_fn = log.warning if status < 500 else log.error
    log_fn("request_failed", code=str(exc.code), status=status,
           path=request.url.path, message=exc.message, detail=exc.detail)
    return _json_response(status, payload)


async def validation_error_handler(request: "Request", exc: Exception) -> "JSONResponse":
    """FastAPI 请求校验失败 → ``OMS-1002``。"""
    errors: list[dict[str, Any]] = []
    raw = getattr(exc, "errors", None)
    if callable(raw):
        for item in raw():
            errors.append({
                "loc": [str(p) for p in item.get("loc", ())],
                "msg": str(item.get("msg", "")),
                "type": str(item.get("type", "")),
            })
    payload = error_body(ErrorCode.INVALID_REQUEST, "请求参数校验失败",
                         detail={"errors": errors[:20]})
    log.warning("request_validation_failed", path=request.url.path, errors=errors[:5])
    return _json_response(422, payload)


async def http_exception_handler(request: "Request", exc: Exception) -> "JSONResponse":
    """Starlette ``HTTPException`` → 统一错误体，保留原状态码。"""
    status = int(getattr(exc, "status_code", 500))
    detail = getattr(exc, "detail", "")
    code = {
        400: ErrorCode.INVALID_REQUEST,
        401: ErrorCode.UNAUTHENTICATED,
        403: ErrorCode.FORBIDDEN,
        404: ErrorCode.NOT_FOUND,
        409: ErrorCode.CONFLICT,
        413: ErrorCode.INVALID_REQUEST,
        422: ErrorCode.INVALID_REQUEST,
        429: ErrorCode.RATE_LIMITED,
        503: ErrorCode.UNAVAILABLE,
    }.get(status, ErrorCode.INTERNAL)
    payload = error_body(code, str(detail) or "请求失败")
    return _json_response(status, payload)


async def unhandled_error_handler(request: "Request", exc: Exception) -> "JSONResponse":
    """兜底：任何未捕获异常 → 500 ``OMS-1001``，不泄露堆栈给客户端。"""
    log.exception("unhandled_exception", path=request.url.path,
                  error_type=type(exc).__name__, error=str(exc))
    payload = error_body(ErrorCode.INTERNAL, "服务内部错误",
                         detail={"type": type(exc).__name__})
    return _json_response(500, payload)


def install_error_handlers(app: "FastAPI") -> None:
    """把全部异常处理器注册到 FastAPI 应用。"""
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException

    app.add_exception_handler(OmniError, omni_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)

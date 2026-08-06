"""协议级错误 -> ``error`` 事件的转换。

WS 层不返回 HTTP 状态码，所有异常都要变成一条下行 ``error`` 事件：

    {"v":"oms.v1","type":"error","payload":{"code":"OMS-6003","message":"...",
     "detail":{},"recoverable":true}}

约定（见架构文档 §7.3）：后端**不做 i18n**，message 是英文/技术描述，
前端按 ``code`` 映射本地化文案。
"""
from __future__ import annotations

from typing import Any

from omnistream.core.error_codes import ErrorCode, is_retryable
from omnistream.core.errors import OmniError
from omnistream.protocol.enums import ServerEventType
from omnistream.protocol.envelope import Envelope
from omnistream.protocol.server_events import ErrorOut

__all__ = [
    "FATAL_CODES",
    "error_envelope",
    "error_payload",
    "is_fatal",
    "to_error_out",
]

#: 这些错误发生后连接无法继续，服务端应在下发 error 事件后主动关闭 WS。
FATAL_CODES: frozenset[ErrorCode] = frozenset({
    ErrorCode.PROTOCOL_VERSION_MISMATCH,
    ErrorCode.UNAUTHENTICATED,
    ErrorCode.INVALID_API_KEY,
    ErrorCode.TOKEN_EXPIRED,
    ErrorCode.FORBIDDEN,
    ErrorCode.TENANT_SUSPENDED,
    ErrorCode.QUOTA_EXCEEDED,
    ErrorCode.CONCURRENCY_LIMIT,
    ErrorCode.TRANSPORT_CLOSED,
    ErrorCode.SESSION_NOT_ACTIVE,
})


def is_fatal(exc: BaseException) -> bool:
    """该异常是否应导致连接关闭。"""
    if isinstance(exc, OmniError):
        return exc.code in FATAL_CODES
    return True  # 未知异常一律视为致命，避免连接停留在不确定状态


def to_error_out(exc: BaseException, *, request_id: str | None = None) -> ErrorOut:
    """把任意异常转成下行 ``error`` 负载模型。"""
    if isinstance(exc, OmniError):
        return ErrorOut(
            code=str(exc.code),
            message=exc.message,
            detail=dict(exc.detail),
            recoverable=is_retryable(exc.code) and exc.code not in FATAL_CODES,
            request_id=request_id,
        )
    return ErrorOut(
        code=str(ErrorCode.INTERNAL),
        message=f"{type(exc).__name__}: {exc}",
        detail={},
        recoverable=False,
        request_id=request_id,
    )


def error_payload(exc: BaseException, *, request_id: str | None = None) -> dict[str, Any]:
    """把异常转成可直接 JSON 序列化的 dict。"""
    return to_error_out(exc, request_id=request_id).model_dump(mode="json", exclude_none=True)


def error_envelope(
    exc: BaseException,
    *,
    session_id: str | None = None,
    seq: int | None = None,
    request_id: str | None = None,
) -> Envelope:
    """把异常包装成可直接下发的 ``error`` :class:`Envelope`。"""
    return Envelope(
        type=str(ServerEventType.ERROR),
        payload=error_payload(exc, request_id=request_id),
        session_id=session_id,
        seq=seq,
    )

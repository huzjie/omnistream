"""统一异常树。业务代码只抛这些异常，API 层统一转 HTTP，WS 层统一转 error 事件。"""
from __future__ import annotations

from typing import Any

from omnistream.core.error_codes import ErrorCode, http_status_for, is_retryable


class OmniError(Exception):
    """所有自定义异常的根。"""

    default_code: ErrorCode = ErrorCode.INTERNAL

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        detail: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.detail = detail or {}
        self.__cause__ = cause

    @property
    def http_status(self) -> int:
        return http_status_for(self.code)

    @property
    def retryable(self) -> bool:
        return is_retryable(self.code)

    def to_dict(self, request_id: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "code": str(self.code),
            "message": self.message,
            "detail": self.detail,
        }
        if request_id:
            body["request_id"] = request_id
        return body

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code}, message={self.message!r})"


class ConfigError(OmniError):
    default_code = ErrorCode.CONFIG_INVALID


class ValidationError(OmniError):
    default_code = ErrorCode.INVALID_REQUEST


class NotFoundError(OmniError):
    default_code = ErrorCode.NOT_FOUND


class ConflictError(OmniError):
    default_code = ErrorCode.CONFLICT


class AuthError(OmniError):
    default_code = ErrorCode.UNAUTHENTICATED


class PermissionError_(OmniError):  # 避免遮蔽内建 PermissionError
    default_code = ErrorCode.FORBIDDEN


class QuotaError(OmniError):
    default_code = ErrorCode.QUOTA_EXCEEDED


class ProtocolError(OmniError):
    default_code = ErrorCode.PROTOCOL_MALFORMED


class MediaError(OmniError):
    default_code = ErrorCode.CODEC_ERROR


class TransportClosed(MediaError):
    default_code = ErrorCode.TRANSPORT_CLOSED


class DuplexError(OmniError):
    default_code = ErrorCode.SESSION_NOT_ACTIVE


class PerceptionError(OmniError):
    default_code = ErrorCode.ASR_FAILED


class ProviderError(OmniError):
    default_code = ErrorCode.PROVIDER_BAD_RESPONSE

    def __init__(self, message: str, *, provider: str = "", **kw: Any) -> None:
        super().__init__(message, **kw)
        self.provider = provider
        self.detail.setdefault("provider", provider)


class ProviderTimeout(ProviderError):
    default_code = ErrorCode.PROVIDER_TIMEOUT


class ProviderUnauthorized(ProviderError):
    default_code = ErrorCode.PROVIDER_UNAUTHORIZED


class CapabilityUnsupported(ProviderError):
    default_code = ErrorCode.PROVIDER_CAPABILITY_UNSUPPORTED


class CircuitOpen(ProviderError):
    default_code = ErrorCode.PROVIDER_CIRCUIT_OPEN


class ToolError(OmniError):
    default_code = ErrorCode.TOOL_EXECUTION_FAILED


class ToolTimeout(ToolError):
    default_code = ErrorCode.TOOL_TIMEOUT


class MemoryError_(OmniError):
    default_code = ErrorCode.MEMORY_STORE_ERROR


class GovernanceError(OmniError):
    default_code = ErrorCode.CONTENT_BLOCKED

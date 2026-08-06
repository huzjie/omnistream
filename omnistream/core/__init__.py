"""omnistream 内核原语。

本包**不依赖**项目内任何其他包（见架构文档 §7.1 的分层依赖方向），
因此可以被 protocol / schemas / config / providers 等任意层安全导入。
"""

from __future__ import annotations

from omnistream.core.circuit_breaker import CircuitBreaker, CircuitState
from omnistream.core.clock import MediaClock, monotonic_s, now_ms, now_ns, utcnow, utcnow_iso
from omnistream.core.constants import (
    API_PREFIX,
    BARGE_IN_BUDGET_MS,
    CANCEL_BUDGET_MS,
    DEFAULT_FRAME_MS,
    DEFAULT_INPUT_SAMPLE_RATE,
    DEFAULT_OUTPUT_SAMPLE_RATE,
    PROTOCOL_VERSION,
    REQUEST_ID_HEADER,
)
from omnistream.core.context import bind_context, get_context
from omnistream.core.error_codes import ErrorCode, http_status_for, is_retryable
from omnistream.core.errors import (
    AuthError,
    CapabilityUnsupported,
    CircuitOpen,
    ConfigError,
    ConflictError,
    DuplexError,
    GovernanceError,
    MediaError,
    MemoryError_,
    NotFoundError,
    OmniError,
    PerceptionError,
    PermissionError_,
    ProtocolError,
    ProviderError,
    ProviderTimeout,
    ProviderUnauthorized,
    QuotaError,
    ToolError,
    ToolTimeout,
    TransportClosed,
    ValidationError,
)
from omnistream.core.eventbus import BusEvent, EventBus, InProcEventBus, RedisStreamEventBus, create_event_bus
from omnistream.core.ids import (
    new_event_id,
    new_id,
    new_memory_id,
    new_session_id,
    new_tenant_id,
    new_tool_call_id,
    new_turn_id,
)
from omnistream.core.lifecycle import LifecycleManager
from omnistream.core.logging import configure_logging, get_logger
from omnistream.core.registry import Registry
from omnistream.core.retry import RetryPolicy, retry_async, retry_call
from omnistream.core.ring_buffer import AudioRingBuffer
from omnistream.core.serialization import dumps, loads, to_jsonable
from omnistream.core.streams import BoundedStream, OverflowPolicy, StreamClosed

__all__ = [
    "API_PREFIX",
    "BARGE_IN_BUDGET_MS",
    "CANCEL_BUDGET_MS",
    "DEFAULT_FRAME_MS",
    "DEFAULT_INPUT_SAMPLE_RATE",
    "DEFAULT_OUTPUT_SAMPLE_RATE",
    "PROTOCOL_VERSION",
    "REQUEST_ID_HEADER",
    "AudioRingBuffer",
    "AuthError",
    "BoundedStream",
    "BusEvent",
    "CapabilityUnsupported",
    "CircuitBreaker",
    "CircuitOpen",
    "CircuitState",
    "ConfigError",
    "ConflictError",
    "DuplexError",
    "ErrorCode",
    "EventBus",
    "GovernanceError",
    "InProcEventBus",
    "LifecycleManager",
    "MediaClock",
    "MediaError",
    "MemoryError_",
    "NotFoundError",
    "OmniError",
    "OverflowPolicy",
    "PerceptionError",
    "PermissionError_",
    "ProtocolError",
    "ProviderError",
    "ProviderTimeout",
    "ProviderUnauthorized",
    "QuotaError",
    "RedisStreamEventBus",
    "Registry",
    "RetryPolicy",
    "StreamClosed",
    "ToolError",
    "ToolTimeout",
    "TransportClosed",
    "ValidationError",
    "bind_context",
    "configure_logging",
    "create_event_bus",
    "dumps",
    "get_context",
    "get_logger",
    "http_status_for",
    "is_retryable",
    "loads",
    "monotonic_s",
    "new_event_id",
    "new_id",
    "new_memory_id",
    "new_session_id",
    "new_tenant_id",
    "new_tool_call_id",
    "new_turn_id",
    "now_ms",
    "now_ns",
    "retry_async",
    "retry_call",
    "to_jsonable",
    "utcnow",
    "utcnow_iso",
]

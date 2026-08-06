"""领域模型统一导出。

约定：所有模型继承 :class:`omnistream.schemas.base.OmniModel`，
只依赖 ``omnistream.core`` 与 ``omnistream.protocol``，不得反向依赖上层。
"""
from __future__ import annotations

from omnistream.schemas.audio import AudioChunk, AudioFormat, AudioStats
from omnistream.schemas.audit import (
    AuditAction,
    AuditEvent,
    PIIMatch,
    PIIRedactionReport,
    verify_chain,
)
from omnistream.schemas.auth import (
    ROLE_PERMISSIONS,
    ApiKey,
    ApiKeyCreate,
    ApiKeyIssued,
    JWTClaims,
    LoginRequest,
    Permission,
    Principal,
    Role,
    TokenPair,
    role_permissions,
)
from omnistream.schemas.base import OmniModel
from omnistream.schemas.billing import (
    BillingMetric,
    BillingSummary,
    Invoice,
    InvoiceLine,
    PriceBook,
    PriceEntry,
    UsageRecord,
)
from omnistream.schemas.common import (
    Cursor,
    ErrorBody,
    HealthStatus,
    Ok,
    Page,
    PageRequest,
    TimeRange,
)
from omnistream.schemas.memory import (
    InjectionDecision,
    MemoryBudget,
    MemoryHit,
    MemoryItem,
    MemoryQuery,
    MemoryStats,
    MemoryUpsert,
)
from omnistream.schemas.provider import (
    GenerationParams,
    ProviderCapabilities,
    ProviderCredential,
    ProviderHealth,
    ProviderInfo,
    ProviderRoute,
)
from omnistream.schemas.quality import (
    QualityReport,
    QualityRubric,
    QualityScore,
    RubricDimension,
)
from omnistream.schemas.quota import QuotaCheck, QuotaPolicy, QuotaUsage, RateLimitRule
from omnistream.schemas.recording import (
    Recording,
    RecordingSegment,
    ReplayEvent,
    ReplayManifest,
)
from omnistream.schemas.session import (
    BargeInConfig,
    MemoryConfig,
    Session,
    SessionConfig,
    SessionCreate,
    SessionStats,
    SessionToken,
    SessionUpdate,
    VADConfig,
)
from omnistream.schemas.tenant import Tenant, TenantCreate, TenantSettings, TenantUpdate
from omnistream.schemas.tool import ToolCall, ToolPolicy, ToolResult, ToolSpec
from omnistream.schemas.transcript import (
    Transcript,
    TranscriptDelta,
    TranscriptSegment,
    WordTiming,
)
from omnistream.schemas.turn import Turn, TurnMetrics, TurnSummary
from omnistream.schemas.video import (
    VideoFrame,
    VideoSamplingPolicy,
    VideoStats,
)
from omnistream.schemas.vision import (
    BoundingBox,
    SceneDescription,
    VideoMemoryNode,
    VisionObservation,
    VisualEntity,
)

__all__ = [
    # base
    "OmniModel",
    # common
    "Cursor", "ErrorBody", "HealthStatus", "Ok", "Page", "PageRequest", "TimeRange",
    # audio
    "AudioChunk", "AudioFormat", "AudioStats",
    # video / vision
    "VideoFrame", "VideoSamplingPolicy", "VideoStats",
    "BoundingBox", "SceneDescription", "VideoMemoryNode", "VisionObservation", "VisualEntity",
    # transcript
    "Transcript", "TranscriptDelta", "TranscriptSegment", "WordTiming",
    # session / turn
    "BargeInConfig", "MemoryConfig", "Session", "SessionConfig", "SessionCreate",
    "SessionStats", "SessionToken", "SessionUpdate", "VADConfig",
    "Turn", "TurnMetrics", "TurnSummary",
    # memory
    "InjectionDecision", "MemoryBudget", "MemoryHit", "MemoryItem", "MemoryQuery",
    "MemoryStats", "MemoryUpsert",
    # tool
    "ToolCall", "ToolPolicy", "ToolResult", "ToolSpec",
    # provider
    "GenerationParams", "ProviderCapabilities", "ProviderCredential", "ProviderHealth",
    "ProviderInfo", "ProviderRoute",
    # tenant / auth
    "Tenant", "TenantCreate", "TenantSettings", "TenantUpdate",
    "ROLE_PERMISSIONS", "ApiKey", "ApiKeyCreate", "ApiKeyIssued", "JWTClaims",
    "LoginRequest", "Permission", "Principal", "Role", "TokenPair", "role_permissions",
    # quota / billing
    "QuotaCheck", "QuotaPolicy", "QuotaUsage", "RateLimitRule",
    "BillingMetric", "BillingSummary", "Invoice", "InvoiceLine", "PriceBook",
    "PriceEntry", "UsageRecord",
    # recording / quality / audit
    "Recording", "RecordingSegment", "ReplayEvent", "ReplayManifest",
    "QualityReport", "QualityRubric", "QualityScore", "RubricDimension",
    "AuditAction", "AuditEvent", "PIIMatch", "PIIRedactionReport", "verify_chain",
]

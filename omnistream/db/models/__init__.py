"""汇总导入所有 ORM 模型 —— alembic autogenerate 依赖本模块发现全部表。

新增模型后**必须**在这里导入，否则迁移会漏表。
"""
from __future__ import annotations

from omnistream.db.base import Base, metadata_obj
from omnistream.db.models.api_key import ApiKey
from omnistream.db.models.audit_log import SYSTEM_TENANT, AuditLog
from omnistream.db.models.memory_item import MemoryItem
from omnistream.db.models.provider_config import (
    ProviderConfig,
    decrypt_credentials,
    encrypt_credentials,
    mask_credentials,
)
from omnistream.db.models.quality_score import QualityScore
from omnistream.db.models.quota import DEFAULT_QUOTA_POLICY, Quota
from omnistream.db.models.recording import Recording, RecordingStatus
from omnistream.db.models.session import LIVE_STATES, Session
from omnistream.db.models.tenant import DEFAULT_TENANT_SETTINGS, Tenant, TenantStatus
from omnistream.db.models.tool_call import ToolCall
from omnistream.db.models.transcript import Transcript
from omnistream.db.models.turn import Turn
from omnistream.db.models.usage_record import UsageMetric, UsageRecord
from omnistream.db.models.user import User, UserStatus
from omnistream.db.models.vision_event import VisionEvent

#: 供治理/清理任务遍历的全部业务表（含租户列）
TENANT_SCOPED_MODELS = (
    User,
    ApiKey,
    Session,
    Turn,
    Transcript,
    VisionEvent,
    MemoryItem,
    ToolCall,
    Recording,
    UsageRecord,
    Quota,
    AuditLog,
    QualityScore,
    ProviderConfig,
)

ALL_MODELS = (Tenant,) + TENANT_SCOPED_MODELS

__all__ = [
    "ALL_MODELS",
    "ApiKey",
    "AuditLog",
    "Base",
    "DEFAULT_QUOTA_POLICY",
    "DEFAULT_TENANT_SETTINGS",
    "LIVE_STATES",
    "MemoryItem",
    "ProviderConfig",
    "QualityScore",
    "Quota",
    "Recording",
    "RecordingStatus",
    "SYSTEM_TENANT",
    "Session",
    "TENANT_SCOPED_MODELS",
    "Tenant",
    "TenantStatus",
    "ToolCall",
    "Transcript",
    "Turn",
    "UsageMetric",
    "UsageRecord",
    "User",
    "UserStatus",
    "VisionEvent",
    "decrypt_credentials",
    "encrypt_credentials",
    "mask_credentials",
    "metadata_obj",
]

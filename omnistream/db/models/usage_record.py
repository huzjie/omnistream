"""``usage_records`` 表：计量明细（秒级打点 → 小时桶聚合）。"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import Boolean, Float, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from omnistream.db.base import Base
from omnistream.db.mixins import TenantMixin, TimestampMixin, ULIDPkMixin
from omnistream.db.types import JSONB, ULIDType, UTCDateTime


class UsageMetric(StrEnum):
    """计量项。单位见 ``omnistream.platform.billing.pricing.METRIC_UNITS``。"""

    AUDIO_IN_MINUTES = "audio_in_minutes"
    AUDIO_OUT_MINUTES = "audio_out_minutes"
    ASR_SECONDS = "asr_seconds"
    TTS_CHARACTERS = "tts_characters"
    LLM_INPUT_TOKENS = "llm_input_tokens"
    LLM_OUTPUT_TOKENS = "llm_output_tokens"
    VIDEO_FRAMES = "video_frames"
    VISION_CALLS = "vision_calls"
    TOOL_CALLS = "tool_calls"
    STORAGE_GB_HOURS = "storage_gb_hours"
    SESSION_MINUTES = "session_minutes"
    EMBEDDING_TOKENS = "embedding_tokens"


class UsageRecord(ULIDPkMixin, TenantMixin, TimestampMixin, Base):
    __tablename__ = "usage_records"
    __id_prefix__ = "use_"

    session_id: Mapped[str | None] = mapped_column(ULIDType, nullable=True, index=True)
    turn_id: Mapped[str | None] = mapped_column(ULIDType, nullable=True)
    metric: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    quantity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    unit: Mapped[str] = mapped_column(String(32), nullable=False, default="unit")
    unit_price: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    currency: Mapped[str] = mapped_column(String(8), nullable=False, default="USD")
    #: 归属小时桶（UTC，分钟秒归零），聚合与账单以此为准
    bucket_hour: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, index=True)
    provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    #: True 表示这是聚合任务生成的汇总行，避免重复计费
    is_rollup: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    #: 幂等键：同键只写一次
    idempotency_key: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    extra: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    __table_args__ = (
        Index("ix_usage_records_tenant_bucket_metric", "tenant_id", "bucket_hour", "metric"),
        Index("ix_usage_records_tenant_rollup", "tenant_id", "is_rollup"),
    )


__all__ = ["UsageMetric", "UsageRecord"]

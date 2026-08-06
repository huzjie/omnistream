"""用量计量打点。

在会话 / 转录 / TTS / LLM 等热点路径上调用，把一次资源消耗写入 ``usage_records`` 表
（按小时桶聚合）。金额默认按单价 0 计算（「仅计量不计费」，见 ADR-3763），价格在出账时
由 :mod:`omnistream.platform.billing.pricing` 再核算。

设计要点：

* 计量与计费解耦 —— 计量只记量，单价由价目表决定，方便事后调价不影响历史数据；
* ``idempotency_key`` 保证同一事件（如一次 turn 的 LLM 调用）不会因为重试重复计费；
* 所有写入走 ``tenant_id`` 隔离，配合 ``TenantScopedRepository`` 的多租户约束。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterable

from omnistream.core.clock import utcnow
from omnistream.core.ids import new_id
from omnistream.db.models.usage_record import UsageMetric, UsageRecord

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = ["METRIC_UNITS", "UsageMeter", "UsagePoint", "record_usage"]

#: 各计量指标的物理单位（写入 usage_records.unit 列）
METRIC_UNITS: dict[UsageMetric, str] = {
    UsageMetric.AUDIO_IN_MINUTES: "minute",
    UsageMetric.AUDIO_OUT_MINUTES: "minute",
    UsageMetric.ASR_SECONDS: "second",
    UsageMetric.TTS_CHARACTERS: "character",
    UsageMetric.LLM_INPUT_TOKENS: "token",
    UsageMetric.LLM_OUTPUT_TOKENS: "token",
    UsageMetric.VIDEO_FRAMES: "frame",
    UsageMetric.VISION_CALLS: "call",
    UsageMetric.TOOL_CALLS: "call",
    UsageMetric.STORAGE_GB_HOURS: "gb_hour",
    UsageMetric.SESSION_MINUTES: "minute",
    UsageMetric.EMBEDDING_TOKENS: "token",
}


def _floor_hour(dt: datetime) -> datetime:
    """把时间归到所在小时的 0 分 0 秒，作为聚合桶边界。"""
    return dt.replace(minute=0, second=0, microsecond=0)


@dataclass(slots=True)
class UsagePoint:
    """一次计量打点。"""

    metric: UsageMetric
    quantity: float
    session_id: str | None = None
    turn_id: str | None = None
    provider: str | None = None
    model: str | None = None
    unit_price: float = 0.0
    currency: str = "USD"
    idempotency_key: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class UsageMeter:
    """用量计量器，绑定一个 SQLAlchemy session。"""

    def __init__(self, session: "AsyncSession") -> None:
        self.session = session

    async def record(
        self, tenant_id: str, point: UsagePoint, *, bucket_hour: datetime | None = None
    ) -> UsageRecord:
        """写入一条用量记录，返回 ORM 对象。"""
        bucket = bucket_hour or _floor_hour(utcnow())
        amount = round(float(point.quantity) * float(point.unit_price or 0.0), 6)
        record = UsageRecord(
            id=new_id("usage"),
            tenant_id=tenant_id,
            session_id=point.session_id,
            turn_id=point.turn_id,
            metric=str(point.metric),
            quantity=float(point.quantity),
            unit=str(METRIC_UNITS.get(point.metric, "unit")),
            unit_price=float(point.unit_price or 0.0),
            amount=amount,
            currency=point.currency or "USD",
            bucket_hour=bucket,
            provider=point.provider,
            model=point.model,
            idempotency_key=point.idempotency_key,
            extra=dict(point.extra or {}),
        )
        self.session.add(record)
        await self.session.flush()
        return record

    async def record_many(
        self, tenant_id: str, points: Iterable[UsagePoint]
    ) -> list[UsageRecord]:
        """批量写入多条用量记录。"""
        out: list[UsageRecord] = []
        for point in points:
            out.append(await self.record(tenant_id, point))
        return out

    # ---- 便捷封装：常见指标 ----

    async def audio_minutes(
        self, tenant_id: str, minutes: float, *, direction: str = "in", **kw: Any
    ) -> UsageRecord:
        metric = (
            UsageMetric.AUDIO_IN_MINUTES if direction == "in" else UsageMetric.AUDIO_OUT_MINUTES
        )
        return await self.record(tenant_id, UsagePoint(metric=metric, quantity=minutes, **kw))

    async def asr_seconds(self, tenant_id: str, seconds: float, **kw: Any) -> UsageRecord:
        return await self.record(tenant_id, UsagePoint(metric=UsageMetric.ASR_SECONDS, quantity=seconds, **kw))

    async def tts_characters(self, tenant_id: str, characters: float, **kw: Any) -> UsageRecord:
        return await self.record(
            tenant_id, UsagePoint(metric=UsageMetric.TTS_CHARACTERS, quantity=characters, **kw)
        )

    async def llm_tokens(
        self, tenant_id: str, input_tokens: float, output_tokens: float, **kw: Any
    ) -> list[UsageRecord]:
        recs: list[UsageRecord] = []
        recs.append(
            await self.record(
                tenant_id, UsagePoint(metric=UsageMetric.LLM_INPUT_TOKENS, quantity=input_tokens, **kw)
            )
        )
        recs.append(
            await self.record(
                tenant_id, UsagePoint(metric=UsageMetric.LLM_OUTPUT_TOKENS, quantity=output_tokens, **kw)
            )
        )
        return recs

    async def storage_gb_hours(
        self, tenant_id: str, gb_hours: float, **kw: Any
    ) -> UsageRecord:
        return await self.record(
            tenant_id, UsagePoint(metric=UsageMetric.STORAGE_GB_HOURS, quantity=gb_hours, **kw)
        )


async def record_usage(
    session: "AsyncSession",
    tenant_id: str,
    point: UsagePoint,
    *,
    bucket_hour: datetime | None = None,
) -> UsageRecord:
    """独立函数版（不持有 meter 实例时的便捷入口）。"""
    return await UsageMeter(session).record(tenant_id, point, bucket_hour=bucket_hour)

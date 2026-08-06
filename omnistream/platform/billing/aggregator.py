"""用量聚合与账单摘要。

把 ``usage_records`` 按 (指标, provider) 汇总成 :class:`~omnistream.schemas.billing.BillingSummary`，
供控制台账单页与出账任务消费。

* :func:`aggregate` —— 对一段时间内的**原始**记录（排除 rollup 行）做实时聚合；
* :func:`aggregate_rollup` —— 把某一小时桶的原始记录压缩成 ``is_rollup=True`` 的汇总行，
  避免后续聚合重复计数（出账任务周期性调用）。
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import func, select

from omnistream.db.models.usage_record import UsageRecord
from omnistream.schemas.billing import BillingSummary
from omnistream.schemas.common import TimeRange

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = ["aggregate", "aggregate_rollup"]

_DB_METRIC_TO_BILLING: dict[str, str] = {
    "audio_in_minutes": "audio_in_minutes",
    "audio_out_minutes": "audio_out_minutes",
    "asr_seconds": "asr_minutes",
    "tts_characters": "tts_characters",
    "llm_input_tokens": "input_tokens",
    "llm_output_tokens": "output_tokens",
    "video_frames": "video_frames",
    "vision_calls": "vlm_images",
    "tool_calls": "tool_calls",
    "storage_gb_hours": "storage_gb_hours",
    "session_minutes": "sessions",
    "embedding_tokens": "input_tokens",
}


async def aggregate(
    session: "AsyncSession",
    tenant_id: str,
    start: datetime,
    end: datetime,
    *,
    currency: str = "USD",
) -> BillingSummary:
    """汇总 ``[start, end)`` 内的原始用量记录，生成账单摘要。"""
    rows = await session.execute(
        select(
            UsageRecord.metric,
            UsageRecord.provider,
            func.coalesce(func.sum(UsageRecord.quantity), 0.0),
            func.coalesce(func.sum(UsageRecord.amount), 0.0),
        )
        .where(
            UsageRecord.tenant_id == tenant_id,
            UsageRecord.is_rollup.is_(False),
            UsageRecord.bucket_hour >= start,
            UsageRecord.bucket_hour < end,
        )
        .group_by(UsageRecord.metric, UsageRecord.provider)
    )

    summary = BillingSummary(
        tenant_id=tenant_id,
        period=TimeRange(start=start, end=end),
        currency=currency,
    )
    for metric, provider, quantity, amount in rows:
        metric_str = str(metric)
        billing_metric = _DB_METRIC_TO_BILLING.get(metric_str, metric_str)
        qty = float(quantity)
        amt = Decimal(str(amount))
        summary.by_metric[billing_metric] = summary.by_metric.get(billing_metric, 0.0) + qty
        summary.amount_by_metric[billing_metric] = (
            summary.amount_by_metric.get(billing_metric, Decimal("0")) + amt
        )
        summary.amount_by_provider[provider or "mock"] = (
            summary.amount_by_provider.get(provider or "mock", Decimal("0")) + amt
        )
        summary.total_amount += amt
    return summary


async def aggregate_rollup(
    session: "AsyncSession",
    tenant_id: str,
    bucket_hour: datetime,
    *,
    end_exclusive: datetime | None = None,
) -> int:
    """把 ``bucket_hour`` 这一小时桶的原始记录压缩成 rollup 行（按指标分组）。

    返回写出的 rollup 行数。原始记录保留（便于审计），但后续聚合会排除 ``is_rollup`` 行。
    """
    from omnistream.core.ids import new_id

    boundary = end_exclusive or (
        bucket_hour.replace(minute=0, second=0, microsecond=0) + __import__("datetime").timedelta(hours=1)
    )
    rows = await session.execute(
        select(
            UsageRecord.metric,
            UsageRecord.provider,
            func.coalesce(func.sum(UsageRecord.quantity), 0.0),
            func.coalesce(func.sum(UsageRecord.amount), 0.0),
            func.coalesce(func.sum(UsageRecord.unit_price), 0.0),
        )
        .where(
            UsageRecord.tenant_id == tenant_id,
            UsageRecord.is_rollup.is_(False),
            UsageRecord.bucket_hour >= bucket_hour,
            UsageRecord.bucket_hour < boundary,
        )
        .group_by(UsageRecord.metric, UsageRecord.provider)
    )

    count = 0
    for metric, provider, quantity, amount, unit_price in rows:
        rollup = UsageRecord(
            id=new_id("usage"),
            tenant_id=tenant_id,
            session_id=None,
            turn_id=None,
            metric=str(metric),
            quantity=float(quantity),
            unit="rollup",
            unit_price=float(unit_price),
            amount=round(float(amount), 6),
            currency="USD",
            bucket_hour=bucket_hour,
            provider=provider,
            model=None,
            is_rollup=True,
            extra={"rolled_up": True},
        )
        session.add(rollup)
        count += 1
    if count:
        await session.flush()
    return count

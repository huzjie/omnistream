"""计费路由：用量明细、账单摘要、价目表。

读操作需要 ``billing:read``。用量数据来自 ``usage_records`` 表，按小时桶聚合；
金额按当前租户价目表核算（默认全 0，仅计量不计费）。
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select

from omnistream.api.deps import CurrentPrincipal, DbSession, Pagination, require_perm
from omnistream.db.models.usage_record import UsageRecord
from omnistream.platform.billing import aggregate, load_price_book
from omnistream.schemas.auth import Permission
from omnistream.schemas.common import Ok, Page, TimeRange

router = APIRouter()
ROUTER_ORDER = 60
ROUTER_TAGS = ["billing"]


def _record_to_dict(rec: UsageRecord) -> dict[str, Any]:
    return {
        "id": rec.id,
        "session_id": rec.session_id,
        "metric": rec.metric,
        "quantity": rec.quantity,
        "unit": rec.unit,
        "unit_price": rec.unit_price,
        "amount": rec.amount,
        "currency": rec.currency,
        "bucket_hour": rec.bucket_hour.isoformat() if rec.bucket_hour else None,
        "provider": rec.provider,
        "model": rec.model,
        "is_rollup": rec.is_rollup,
    }


@router.get("/usage", response_model=Ok[Page[dict]])
async def usage(
    principal: Annotated[Any, Depends(require_perm(Permission.BILLING_READ))],
    session: DbSession,
    page: Pagination,
    metric: str | None = Query(default=None),
) -> Ok[Page[dict]]:
    """分页查询用量明细。"""
    from sqlalchemy import func

    stmt = select(UsageRecord).where(UsageRecord.tenant_id == principal.tenant_id)
    if metric:
        stmt = stmt.where(UsageRecord.metric == metric)
    total = await session.execute(
        func.count().select_from(UsageRecord).where(UsageRecord.tenant_id == principal.tenant_id)
    )
    rows = await session.execute(
        stmt.order_by(UsageRecord.bucket_hour.desc()).limit(page.limit).offset(page.offset)
    )
    items = [_record_to_dict(r) for r in rows.scalars().all()]
    return Ok.of(Page.of(items, total=int(total.scalar_one() or 0)))


@router.get("/summary", response_model=Ok[dict])
async def summary(
    principal: Annotated[Any, Depends(require_perm(Permission.BILLING_READ))],
    session: DbSession,
    days: int = Query(default=30, ge=1, le=365),
) -> Ok[dict]:
    """返回最近 N 天的账单摘要。"""
    end = datetime.now(UTC)
    start = end - timedelta(days=days)
    result = await aggregate(session, principal.tenant_id, start, end)
    return Ok.of(result.model_dump(mode="json"))


@router.get("/price-book", response_model=Ok[dict])
async def price_book(
    principal: Annotated[Any, Depends(require_perm(Permission.BILLING_READ))],
    session: DbSession,
) -> Ok[dict]:
    """返回当前租户价目表。"""
    view = await load_price_book(session, principal.tenant_id)
    return Ok.of(view.book.model_dump(mode="json"))

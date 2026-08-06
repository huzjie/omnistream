"""价目表加载与金额计算（支持租户级折扣）。

默认内置一份「参考价目表」，所有单价 = 0（仅计量不计费，见 ADR-3763），避免开源项目
内置商业定价引发误解。租户可通过 ``tenants.settings['billing']['price_book']`` 覆盖。

金额公式：``amount = quantity / per × unit_price × discount``，结果保留 6 位小数。
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from omnistream.db.models.tenant import Tenant
from omnistream.schemas.billing import BillingMetric, PriceBook, PriceEntry

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "DEFAULT_CURRENCY",
    "DEFAULT_PRICE_BOOK",
    "PriceBookView",
    "compute_amount",
    "load_price_book",
    "price_for",
]

DEFAULT_CURRENCY = "USD"

#: 全 0 单价参考价目表（开源默认：仅计量不计费）
DEFAULT_PRICE_BOOK = PriceBook(
    name="oss-default",
    currency=DEFAULT_CURRENCY,
    entries=[PriceEntry(metric=m, unit_price=Decimal("0"), per=1.0) for m in BillingMetric],
)


@dataclass(slots=True)
class PriceBookView:
    """一次价目表查询的结果视图。"""

    tenant_id: str
    currency: str
    book: PriceBook


async def load_price_book(session: "AsyncSession", tenant_id: str) -> PriceBookView:
    """加载租户价目表；无覆盖时回落到全 0 默认表。"""
    tenant = (
        await session.execute(select(Tenant).where(Tenant.id == tenant_id))
    ).scalar_one_or_none()
    book = DEFAULT_PRICE_BOOK
    if tenant is not None:
        raw = tenant.setting_path("billing.price_book")
        if isinstance(raw, dict) and raw:
            try:
                book = PriceBook.model_validate(raw)
            except Exception:
                book = DEFAULT_PRICE_BOOK
    currency = book.currency or DEFAULT_CURRENCY
    return PriceBookView(tenant_id=tenant_id, currency=currency, book=book)


def price_for(
    book: PriceBook,
    metric: "BillingMetric | str",
    *,
    provider: str = "*",
    model: str = "*",
    quantity: float = 1.0,
) -> Decimal:
    """查价目表得到某指标在给定 provider/model 下的金额。"""
    try:
        return book.price(BillingMetric(str(metric)), quantity, provider=provider, model=model)
    except Exception:
        return Decimal("0")


def compute_amount(
    metric: "BillingMetric | str",
    quantity: float,
    unit_price: float,
    *,
    per: float = 1.0,
    discount: "Decimal | float" = Decimal("1"),
) -> Decimal:
    """按 ``quantity / per × unit_price × discount`` 计算金额。"""
    base = Decimal(str(quantity)) / Decimal(str(per)) * Decimal(str(unit_price))
    return (base * Decimal(str(discount))).quantize(Decimal("0.000001"))

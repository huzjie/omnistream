"""计量与计费领域模型。"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import Field

from omnistream.core.clock import utcnow
from omnistream.schemas.base import OmniModel
from omnistream.schemas.common import TimeRange


class BillingMetric(StrEnum):
    """可计费指标。与 ``usage_records.metric`` 列取值一致。"""

    AUDIO_IN_MINUTES = "audio_in_minutes"
    AUDIO_OUT_MINUTES = "audio_out_minutes"
    VIDEO_FRAMES = "video_frames"
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    TOOL_CALLS = "tool_calls"
    ASR_MINUTES = "asr_minutes"
    TTS_CHARACTERS = "tts_characters"
    VLM_IMAGES = "vlm_images"
    STORAGE_GB_HOURS = "storage_gb_hours"
    SESSIONS = "sessions"


class UsageRecord(OmniModel):
    """一条计量明细。按小时桶聚合写入。"""

    id: str = ""
    tenant_id: str
    session_id: str | None = None
    provider: str = "mock"
    model: str | None = None
    metric: BillingMetric = BillingMetric.SESSIONS
    quantity: float = 0.0
    unit_price: Decimal = Decimal("0")
    amount: Decimal = Decimal("0")
    currency: str = "CNY"
    bucket_hour: datetime = Field(default_factory=utcnow)
    created_at: datetime = Field(default_factory=utcnow)

    def recompute(self) -> "UsageRecord":
        """按 quantity × unit_price 重算金额。"""
        self.amount = (Decimal(str(self.quantity)) * self.unit_price).quantize(Decimal("0.000001"))
        return self


class PriceEntry(OmniModel):
    """单条定价。``per`` 表示单价对应的数量单位。"""

    metric: BillingMetric
    provider: str = "*"
    model: str = "*"
    unit_price: Decimal = Decimal("0")
    per: float = Field(default=1.0, gt=0.0, description="例如 tokens 按 1000 计价则填 1000")
    currency: str = "CNY"

    def cost(self, quantity: float) -> Decimal:
        return (Decimal(str(quantity)) / Decimal(str(self.per)) * self.unit_price).quantize(
            Decimal("0.000001"))


class PriceBook(OmniModel):
    """价目表。按 (metric, provider, model) 精确优先、通配兜底匹配。"""

    name: str = "default"
    currency: str = "CNY"
    entries: list[PriceEntry] = Field(default_factory=list)
    effective_from: datetime = Field(default_factory=utcnow)

    def lookup(self, metric: BillingMetric, provider: str = "*",
               model: str = "*") -> PriceEntry | None:
        candidates = [e for e in self.entries if e.metric == metric]
        if not candidates:
            return None

        def _score(e: PriceEntry) -> int:
            return (2 if e.provider == provider else 0) + (1 if e.model == model else 0)

        exact = [e for e in candidates
                 if e.provider in (provider, "*") and e.model in (model, "*")]
        if not exact:
            return None
        return max(exact, key=_score)

    def price(self, metric: BillingMetric, quantity: float, *, provider: str = "*",
              model: str = "*") -> Decimal:
        entry = self.lookup(metric, provider, model)
        return entry.cost(quantity) if entry else Decimal("0")


class BillingSummary(OmniModel):
    """一个时间范围内的账单聚合视图，控制台首页直接消费。"""

    tenant_id: str
    period: TimeRange
    currency: str = "CNY"
    total_amount: Decimal = Decimal("0")
    by_metric: dict[str, float] = Field(default_factory=dict)
    amount_by_metric: dict[str, Decimal] = Field(default_factory=dict)
    amount_by_provider: dict[str, Decimal] = Field(default_factory=dict)
    sessions: int = 0
    generated_at: datetime = Field(default_factory=utcnow)

    def add(self, record: UsageRecord) -> "BillingSummary":
        metric = str(record.metric)
        self.by_metric[metric] = self.by_metric.get(metric, 0.0) + record.quantity
        self.amount_by_metric[metric] = self.amount_by_metric.get(metric, Decimal("0")) + record.amount
        self.amount_by_provider[record.provider] = (
            self.amount_by_provider.get(record.provider, Decimal("0")) + record.amount)
        self.total_amount += record.amount
        return self


class InvoiceLine(OmniModel):
    description: str
    metric: BillingMetric
    quantity: float = 0.0
    unit_price: Decimal = Decimal("0")
    amount: Decimal = Decimal("0")


class Invoice(OmniModel):
    """账单。状态机：draft → issued → paid / void。"""

    id: str = ""
    tenant_id: str
    number: str = ""
    period: TimeRange
    currency: str = "CNY"
    lines: list[InvoiceLine] = Field(default_factory=list)
    subtotal: Decimal = Decimal("0")
    tax_rate: Decimal = Decimal("0")
    tax: Decimal = Decimal("0")
    total: Decimal = Decimal("0")
    status: str = Field(default="draft", description="draft|issued|paid|void")
    issued_at: datetime | None = None
    due_at: datetime | None = None
    paid_at: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)

    def recompute(self) -> "Invoice":
        self.subtotal = sum((line.amount for line in self.lines), Decimal("0"))
        self.tax = (self.subtotal * self.tax_rate).quantize(Decimal("0.01"))
        self.total = (self.subtotal + self.tax).quantize(Decimal("0.01"))
        return self

"""配额策略解析。

策略来源三层叠加（后者覆盖前者）：

1. :data:`PLAN_POLICIES` —— 套餐内置基线（free / starter / pro / enterprise）
2. ``tenants.settings["quota"]`` —— 租户级微调（运营后台改）
3. ``quotas.policy`` —— 该租户的显式配额行（销售/客服单独授权）

任何一项取值 ``<= 0`` 表示**不限制**。
"""
from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from omnistream.db.models.quota import DEFAULT_QUOTA_POLICY, Quota
from omnistream.db.models.tenant import Tenant
from omnistream.db.types import utc_now

__all__ = [
    "PLAN_POLICIES",
    "POLICY_KEYS",
    "QuotaPolicy",
    "current_period",
    "ensure_quota_row",
    "merge_policy",
    "plan_policy",
    "resolve_policy",
]

#: 全部合法策略键（未知键会被丢弃，避免脏数据污染）
POLICY_KEYS: tuple[str, ...] = tuple(DEFAULT_QUOTA_POLICY.keys())

#: 各套餐的基线配额
PLAN_POLICIES: dict[str, dict[str, Any]] = {
    "free": {
        "max_concurrent_sessions": 3,
        "max_sessions_per_day": 200,
        "max_session_minutes": 15,
        "monthly_audio_minutes": 300,
        "daily_llm_tokens": 200_000,
        "monthly_storage_gb": 1,
        "max_api_keys": 3,
        "max_users": 3,
        "rate_limit_rps": 5.0,
        "rate_limit_burst": 10,
        "ws_connections_per_key": 5,
    },
    "starter": {
        "max_concurrent_sessions": 10,
        "max_sessions_per_day": 2000,
        "max_session_minutes": 30,
        "monthly_audio_minutes": 3000,
        "daily_llm_tokens": 1_000_000,
        "monthly_storage_gb": 10,
        "max_api_keys": 10,
        "max_users": 10,
        "rate_limit_rps": 10.0,
        "rate_limit_burst": 20,
        "ws_connections_per_key": 20,
    },
    "pro": dict(DEFAULT_QUOTA_POLICY),
    "enterprise": {
        "max_concurrent_sessions": 500,
        "max_sessions_per_day": 0,
        "max_session_minutes": 240,
        "monthly_audio_minutes": 0,
        "daily_llm_tokens": 0,
        "monthly_storage_gb": 2000,
        "max_api_keys": 200,
        "max_users": 1000,
        "rate_limit_rps": 200.0,
        "rate_limit_burst": 400,
        "ws_connections_per_key": 500,
    },
}


def plan_policy(plan: str) -> dict[str, Any]:
    """返回套餐基线（未知套餐回落 free）。"""
    return dict(PLAN_POLICIES.get((plan or "free").strip().lower(), PLAN_POLICIES["free"]))


def merge_policy(*layers: Mapping[str, Any] | None) -> dict[str, Any]:
    """按顺序合并策略层，仅保留 :data:`POLICY_KEYS` 中的键。"""
    merged: dict[str, Any] = dict(DEFAULT_QUOTA_POLICY)
    for layer in layers:
        if not layer:
            continue
        for key, value in layer.items():
            if key not in DEFAULT_QUOTA_POLICY:
                continue
            if value is None:
                continue
            try:
                merged[key] = float(value) if isinstance(value, float) else int(value) if isinstance(value, (int, bool)) else float(value)
            except (TypeError, ValueError):
                continue
    return merged


def current_period(period: str = "monthly", now: datetime | None = None) -> tuple[datetime, datetime]:
    """返回当前计费周期的 ``[start, end)``。支持 daily / weekly / monthly。"""
    moment = now or utc_now()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    kind = (period or "monthly").strip().lower()

    if kind == "daily":
        start = moment.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + timedelta(days=1)
    if kind == "weekly":
        start = (moment - timedelta(days=moment.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return start, start + timedelta(days=7)

    start = moment.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    days_in_month = monthrange(start.year, start.month)[1]
    return start, start + timedelta(days=days_in_month)


@dataclass(slots=True)
class QuotaPolicy:
    """解析后的配额策略（不可变语义，读多写少）。"""

    tenant_id: str = ""
    plan: str = "free"
    enforced: bool = True
    on_exceed: str = "block"
    period: str = "monthly"
    period_start: datetime | None = None
    period_end: datetime | None = None
    values: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_QUOTA_POLICY))
    used: dict[str, float] = field(default_factory=dict)

    # ---- 便捷访问器 ----

    def limit(self, key: str) -> float:
        """返回上限；``<=0`` 表示不限制。"""
        try:
            return float(self.values.get(key, DEFAULT_QUOTA_POLICY.get(key, 0)) or 0)
        except (TypeError, ValueError):
            return 0.0

    def unlimited(self, key: str) -> bool:
        return self.limit(key) <= 0

    def consumed(self, key: str) -> float:
        try:
            return float(self.used.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def remaining(self, key: str) -> float:
        """剩余额度；不限制时返回 ``float('inf')``。"""
        if self.unlimited(key):
            return float("inf")
        return max(0.0, self.limit(key) - self.consumed(key))

    def would_exceed(self, key: str, delta: float = 0.0) -> bool:
        if not self.enforced or self.unlimited(key):
            return False
        return self.consumed(key) + float(delta) > self.limit(key)

    @property
    def max_concurrent_sessions(self) -> int:
        return int(self.limit("max_concurrent_sessions"))

    @property
    def max_sessions_per_day(self) -> int:
        return int(self.limit("max_sessions_per_day"))

    @property
    def max_session_minutes(self) -> int:
        return int(self.limit("max_session_minutes"))

    @property
    def rate_limit_rps(self) -> float:
        return float(self.limit("rate_limit_rps"))

    @property
    def rate_limit_burst(self) -> int:
        value = int(self.limit("rate_limit_burst"))
        return value if value > 0 else max(1, int(self.rate_limit_rps * 2))

    @property
    def ws_connections_per_key(self) -> int:
        return int(self.limit("ws_connections_per_key"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "plan": self.plan,
            "enforced": self.enforced,
            "on_exceed": self.on_exceed,
            "period": self.period,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "limits": dict(self.values),
            "used": dict(self.used),
            "remaining": {
                key: (None if self.unlimited(key) else self.remaining(key))
                for key in POLICY_KEYS
            },
        }


async def ensure_quota_row(session: AsyncSession, tenant_id: str, *, plan: str = "") -> Quota:
    """确保租户有一行 ``quotas`` 记录，没有就按套餐基线创建。"""
    existing = (
        await session.execute(select(Quota).where(Quota.tenant_id == tenant_id))
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    resolved_plan = plan
    if not resolved_plan:
        tenant = (
            await session.execute(select(Tenant).where(Tenant.id == tenant_id))
        ).scalar_one_or_none()
        resolved_plan = str(getattr(tenant, "plan", "free") or "free")

    start, end = current_period("monthly")
    row = Quota(
        tenant_id=tenant_id,
        policy=plan_policy(resolved_plan),
        used={},
        period="monthly",
        period_start=start,
        period_end=end,
        enforced=True,
        on_exceed="block",
    )
    session.add(row)
    await session.flush()
    return row


async def resolve_policy(
    session: AsyncSession, tenant_id: str, *, auto_create: bool = True
) -> QuotaPolicy:
    """解析租户的最终生效配额策略，并在周期翻页时重置用量。"""
    tenant = (
        await session.execute(select(Tenant).where(Tenant.id == tenant_id))
    ).scalar_one_or_none()
    plan = str(getattr(tenant, "plan", "free") or "free")
    tenant_overrides: Mapping[str, Any] = {}
    if tenant is not None and isinstance(tenant.settings, Mapping):
        raw = tenant.settings.get("quota")
        if isinstance(raw, Mapping):
            tenant_overrides = raw

    row: Quota | None
    if auto_create:
        row = await ensure_quota_row(session, tenant_id, plan=plan)
    else:
        row = (
            await session.execute(select(Quota).where(Quota.tenant_id == tenant_id))
        ).scalar_one_or_none()

    now = utc_now()
    period = str(getattr(row, "period", "monthly") or "monthly")
    start, end = current_period(period, now)

    if row is not None:
        # 周期翻页：重置用量快照
        if row.period_end is None or row.period_end <= now:
            row.used = {}
            row.period_start = start
            row.period_end = end
            await session.flush()
        else:
            start = row.period_start or start
            end = row.period_end

    values = merge_policy(
        plan_policy(plan),
        tenant_overrides,
        getattr(row, "policy", None),
    )
    used_raw = dict(getattr(row, "used", None) or {})
    used: dict[str, float] = {}
    for key, value in used_raw.items():
        try:
            used[str(key)] = float(value)
        except (TypeError, ValueError):
            continue

    return QuotaPolicy(
        tenant_id=tenant_id,
        plan=plan,
        enforced=bool(getattr(row, "enforced", True)),
        on_exceed=str(getattr(row, "on_exceed", "block") or "block"),
        period=period,
        period_start=start,
        period_end=end,
        values=values,
        used=used,
    )

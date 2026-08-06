"""配额执行器：会话创建前置检查 + 运行中超额熔断。

三个使用时机：

1. ``POST /v1/sessions`` 或 WS 握手前 —— :meth:`QuotaEnforcer.check_session_start`
2. 会话运行中每 N 秒巡检   —— :meth:`QuotaEnforcer.check_runtime`
3. 计量落点后累加用量       —— :meth:`QuotaEnforcer.consume`

``on_exceed`` 支持三种处置：

* ``block``   —— 抛 ``QuotaError``，直接拒绝（默认）
* ``degrade`` —— 不抛异常，返回 ``degraded=True``，由编排层降级（关视频/换便宜模型）
* ``notify``  —— 仅记录并触发 webhook，不影响调用
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Mapping

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import QuotaError
from omnistream.db.models.api_key import ApiKey
from omnistream.db.models.quota import Quota
from omnistream.db.models.session import LIVE_STATES, Session as SessionModel
from omnistream.db.models.user import User
from omnistream.db.types import utc_now
from omnistream.platform.quota.policy import QuotaPolicy, current_period, resolve_policy

__all__ = ["QuotaCheck", "QuotaEnforcer", "QuotaSnapshot"]


@dataclass(slots=True)
class QuotaCheck:
    """单次检查结果。"""

    ok: bool = True
    degraded: bool = False
    reason: str = ""
    metric: str = ""
    limit: float = 0.0
    used: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "degraded": self.degraded,
            "reason": self.reason,
            "metric": self.metric,
            "limit": self.limit,
            "used": self.used,
        }


@dataclass(slots=True)
class QuotaSnapshot:
    """对外展示的配额快照。"""

    tenant_id: str
    plan: str
    enforced: bool
    on_exceed: str
    period_start: str = ""
    period_end: str = ""
    limits: dict[str, Any] = field(default_factory=dict)
    used: dict[str, float] = field(default_factory=dict)
    live_sessions: int = 0
    sessions_today: int = 0
    api_keys: int = 0
    users: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "plan": self.plan,
            "enforced": self.enforced,
            "on_exceed": self.on_exceed,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "limits": dict(self.limits),
            "used": dict(self.used),
            "current": {
                "live_sessions": self.live_sessions,
                "sessions_today": self.sessions_today,
                "api_keys": self.api_keys,
                "users": self.users,
            },
        }


class QuotaEnforcer:
    """租户配额执行器。构造后请先 ``await load()``（或用 :meth:`create`）。"""

    def __init__(self, session: AsyncSession, tenant_id: str) -> None:
        if not tenant_id:
            raise QuotaError("配额检查必须提供 tenant_id", code=ErrorCode.INVALID_REQUEST)
        self.session = session
        self.tenant_id = tenant_id
        self.policy: QuotaPolicy = QuotaPolicy(tenant_id=tenant_id)
        self._loaded = False

    @classmethod
    async def create(cls, session: AsyncSession, tenant_id: str) -> "QuotaEnforcer":
        enforcer = cls(session, tenant_id)
        await enforcer.load()
        return enforcer

    async def load(self) -> QuotaPolicy:
        """加载/刷新策略。"""
        self.policy = await resolve_policy(self.session, self.tenant_id)
        self._loaded = True
        return self.policy

    async def _ensure_loaded(self) -> None:
        if not self._loaded:
            await self.load()

    # ---------------- 实时计数 ----------------

    async def count_live_sessions(self) -> int:
        """当前处于活跃状态的会话数。"""
        stmt = (
            select(func.count())
            .select_from(SessionModel)
            .where(
                SessionModel.tenant_id == self.tenant_id,
                SessionModel.state.in_(tuple(LIVE_STATES)),
            )
        )
        return int((await self.session.execute(stmt)).scalar_one() or 0)

    async def count_sessions_today(self) -> int:
        """今日已创建会话数。"""
        start, _ = current_period("daily")
        stmt = (
            select(func.count())
            .select_from(SessionModel)
            .where(
                SessionModel.tenant_id == self.tenant_id,
                SessionModel.created_at >= start,
            )
        )
        return int((await self.session.execute(stmt)).scalar_one() or 0)

    async def count_api_keys(self) -> int:
        stmt = (
            select(func.count())
            .select_from(ApiKey)
            .where(
                ApiKey.tenant_id == self.tenant_id,
                ApiKey.deleted_at.is_(None),
                ApiKey.revoked_at.is_(None),
            )
        )
        return int((await self.session.execute(stmt)).scalar_one() or 0)

    async def count_users(self) -> int:
        stmt = (
            select(func.count())
            .select_from(User)
            .where(User.tenant_id == self.tenant_id, User.deleted_at.is_(None))
        )
        return int((await self.session.execute(stmt)).scalar_one() or 0)

    # ---------------- 检查 ----------------

    def _reject(self, check: QuotaCheck) -> QuotaCheck:
        """按 ``on_exceed`` 决定抛异常还是降级返回。"""
        if not self.policy.enforced or self.policy.on_exceed == "notify":
            return QuotaCheck(
                ok=True,
                degraded=False,
                reason=check.reason,
                metric=check.metric,
                limit=check.limit,
                used=check.used,
            )
        if self.policy.on_exceed == "degrade":
            return QuotaCheck(
                ok=True,
                degraded=True,
                reason=check.reason,
                metric=check.metric,
                limit=check.limit,
                used=check.used,
            )
        raise QuotaError(
            check.reason,
            code=(
                ErrorCode.CONCURRENCY_LIMIT
                if check.metric == "max_concurrent_sessions"
                else ErrorCode.QUOTA_EXCEEDED
            ),
            detail={
                "metric": check.metric,
                "limit": check.limit,
                "used": check.used,
                "tenant_id": self.tenant_id,
            },
        )

    async def check_session_start(self, *, requested_minutes: int = 0) -> QuotaCheck:
        """创建会话前的全部前置检查。"""
        await self._ensure_loaded()
        policy = self.policy

        if not policy.enforced:
            return QuotaCheck(ok=True)

        limit = policy.max_concurrent_sessions
        if limit > 0:
            live = await self.count_live_sessions()
            if live >= limit:
                return self._reject(
                    QuotaCheck(
                        ok=False,
                        reason=f"并发会话数已达上限 {limit}",
                        metric="max_concurrent_sessions",
                        limit=float(limit),
                        used=float(live),
                    )
                )

        daily = policy.max_sessions_per_day
        if daily > 0:
            today = await self.count_sessions_today()
            if today >= daily:
                return self._reject(
                    QuotaCheck(
                        ok=False,
                        reason=f"今日会话数已达上限 {daily}",
                        metric="max_sessions_per_day",
                        limit=float(daily),
                        used=float(today),
                    )
                )

        minutes_limit = policy.limit("monthly_audio_minutes")
        if minutes_limit > 0:
            used = policy.consumed("monthly_audio_minutes")
            if used >= minutes_limit:
                return self._reject(
                    QuotaCheck(
                        ok=False,
                        reason=f"本月音频分钟数已用尽（{minutes_limit:.0f} 分钟）",
                        metric="monthly_audio_minutes",
                        limit=minutes_limit,
                        used=used,
                    )
                )

        tokens_limit = policy.limit("daily_llm_tokens")
        if tokens_limit > 0 and policy.consumed("daily_llm_tokens") >= tokens_limit:
            return self._reject(
                QuotaCheck(
                    ok=False,
                    reason=f"今日 token 额度已用尽（{tokens_limit:.0f}）",
                    metric="daily_llm_tokens",
                    limit=tokens_limit,
                    used=policy.consumed("daily_llm_tokens"),
                )
            )

        if requested_minutes > 0 and policy.max_session_minutes > 0:
            if requested_minutes > policy.max_session_minutes:
                return self._reject(
                    QuotaCheck(
                        ok=False,
                        reason=f"单次会话时长上限 {policy.max_session_minutes} 分钟",
                        metric="max_session_minutes",
                        limit=float(policy.max_session_minutes),
                        used=float(requested_minutes),
                    )
                )

        return QuotaCheck(ok=True)

    async def check_api_key_creation(self) -> QuotaCheck:
        """签发 API Key 前的数量检查。"""
        await self._ensure_loaded()
        limit = int(self.policy.limit("max_api_keys"))
        if limit <= 0 or not self.policy.enforced:
            return QuotaCheck(ok=True)
        current = await self.count_api_keys()
        if current >= limit:
            return self._reject(
                QuotaCheck(
                    ok=False,
                    reason=f"API Key 数量已达上限 {limit}",
                    metric="max_api_keys",
                    limit=float(limit),
                    used=float(current),
                )
            )
        return QuotaCheck(ok=True)

    async def check_user_creation(self) -> QuotaCheck:
        """新增成员前的数量检查。"""
        await self._ensure_loaded()
        limit = int(self.policy.limit("max_users"))
        if limit <= 0 or not self.policy.enforced:
            return QuotaCheck(ok=True)
        current = await self.count_users()
        if current >= limit:
            return self._reject(
                QuotaCheck(
                    ok=False,
                    reason=f"成员数量已达上限 {limit}",
                    metric="max_users",
                    limit=float(limit),
                    used=float(current),
                )
            )
        return QuotaCheck(ok=True)

    async def check_runtime(self, session_row: SessionModel) -> QuotaCheck:
        """会话运行中巡检：单次时长与本周期额度。"""
        await self._ensure_loaded()
        policy = self.policy
        if not policy.enforced:
            return QuotaCheck(ok=True)

        max_minutes = policy.max_session_minutes
        if max_minutes > 0 and session_row.started_at is not None:
            elapsed = utc_now() - session_row.started_at
            if elapsed > timedelta(minutes=max_minutes):
                return self._reject(
                    QuotaCheck(
                        ok=False,
                        reason=f"会话已超过单次时长上限 {max_minutes} 分钟",
                        metric="max_session_minutes",
                        limit=float(max_minutes),
                        used=elapsed.total_seconds() / 60.0,
                    )
                )

        for metric in ("monthly_audio_minutes", "daily_llm_tokens", "monthly_storage_gb"):
            limit = policy.limit(metric)
            if limit > 0 and policy.consumed(metric) >= limit:
                return self._reject(
                    QuotaCheck(
                        ok=False,
                        reason=f"{metric} 额度已用尽",
                        metric=metric,
                        limit=limit,
                        used=policy.consumed(metric),
                    )
                )
        return QuotaCheck(ok=True)

    # ---------------- 用量累加 ----------------

    async def consume(self, metric: str, delta: float) -> float:
        """累加用量到 ``quotas.used`` 并返回累加后的值。"""
        if delta == 0:
            return self.policy.consumed(metric)
        row = (
            await self.session.execute(select(Quota).where(Quota.tenant_id == self.tenant_id))
        ).scalar_one_or_none()
        if row is None:
            await self.load()
            row = (
                await self.session.execute(
                    select(Quota).where(Quota.tenant_id == self.tenant_id)
                )
            ).scalar_one_or_none()
        if row is None:
            return self.policy.consumed(metric)

        total = row.add_used(metric, float(delta))
        row.updated_at = utc_now()
        await self.session.flush()
        self.policy.used[metric] = total
        return total

    async def consume_many(self, deltas: Mapping[str, float]) -> dict[str, float]:
        """批量累加，减少写库次数。"""
        row = (
            await self.session.execute(select(Quota).where(Quota.tenant_id == self.tenant_id))
        ).scalar_one_or_none()
        if row is None:
            await self.load()
            row = (
                await self.session.execute(
                    select(Quota).where(Quota.tenant_id == self.tenant_id)
                )
            ).scalar_one_or_none()
        if row is None:
            return {}

        result: dict[str, float] = {}
        for metric, delta in deltas.items():
            if not delta:
                continue
            result[metric] = row.add_used(metric, float(delta))
            self.policy.used[metric] = result[metric]
        if result:
            row.updated_at = utc_now()
            await self.session.flush()
        return result

    async def reset_period(self) -> QuotaPolicy:
        """强制重置当前周期用量（人工干预 / 定时任务）。"""
        row = (
            await self.session.execute(select(Quota).where(Quota.tenant_id == self.tenant_id))
        ).scalar_one_or_none()
        if row is not None:
            start, end = current_period(str(row.period or "monthly"))
            row.used = {}
            row.period_start = start
            row.period_end = end
            row.updated_at = utc_now()
            await self.session.flush()
        return await self.load()

    async def snapshot(self) -> QuotaSnapshot:
        """生成对外快照（控制台配额页用）。"""
        await self._ensure_loaded()
        policy = self.policy
        return QuotaSnapshot(
            tenant_id=self.tenant_id,
            plan=policy.plan,
            enforced=policy.enforced,
            on_exceed=policy.on_exceed,
            period_start=policy.period_start.isoformat() if policy.period_start else "",
            period_end=policy.period_end.isoformat() if policy.period_end else "",
            limits=dict(policy.values),
            used=dict(policy.used),
            live_sessions=await self.count_live_sessions(),
            sessions_today=await self.count_sessions_today(),
            api_keys=await self.count_api_keys(),
            users=await self.count_users(),
        )

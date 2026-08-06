"""会话 / 轮次 / 转写 / 视觉事件 / 工具调用的读写与分页查询。"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import ConflictError, NotFoundError
from omnistream.db.models.session import LIVE_STATES, Session as SessionORM
from omnistream.db.models.tool_call import ToolCall
from omnistream.db.models.transcript import Transcript
from omnistream.db.models.turn import Turn
from omnistream.db.models.vision_event import VisionEvent
from omnistream.db.repositories.base import PageResult, TenantScopedRepository, clamp_limit
from omnistream.db.types import utc_now

log = structlog.get_logger(__name__)


class SessionRepository(TenantScopedRepository[SessionORM]):
    """会话主表仓储，附带轮次/转写/视觉/工具调用的从属操作。"""

    model = SessionORM

    # ---------------- 会话 ----------------

    async def create_session(
        self,
        *,
        provider: str = "mock",
        model: str | None = None,
        config: dict[str, Any] | None = None,
        modalities: list[str] | None = None,
        end_user_id: str | None = None,
        api_key_id: str | None = None,
        user_id: str | None = None,
        client_ip: str | None = None,
        user_agent: str | None = None,
        metadata: dict[str, Any] | None = None,
        record_enabled: bool = False,
        redact_pii: bool = True,
        session_id: str | None = None,
    ) -> SessionORM:
        payload: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "state": "created",
            "config": config or {},
            "stats": {},
            "meta": metadata or {},
            "modalities": modalities or ["text", "audio"],
            "end_user_id": end_user_id,
            "api_key_id": api_key_id,
            "user_id": user_id,
            "client_ip": client_ip,
            "user_agent": user_agent,
            "record_enabled": record_enabled,
            "redact_pii": redact_pii,
        }
        if session_id:
            payload["id"] = session_id
        entity = await self.create(**payload)
        log.info("session_created", session_id=entity.id, provider=provider)
        return entity

    async def mark_active(self, session_id: str) -> SessionORM:
        entity = await self.require(session_id)
        entity.state = "active"
        entity.started_at = entity.started_at or utc_now()
        await self.session.flush()
        return entity

    async def close_session(
        self,
        session_id: str,
        *,
        stop_reason: str = "completed",
        state: str = "ended",
        stats: dict[str, Any] | None = None,
        error_code: str | None = None,
    ) -> SessionORM:
        entity = await self.require(session_id)
        now = utc_now()
        entity.state = state
        entity.stop_reason = stop_reason
        entity.ended_at = now
        entity.error_code = error_code
        if entity.started_at:
            entity.duration_ms = max(0, int((now - entity.started_at).total_seconds() * 1000))
        if stats:
            merged = dict(entity.stats or {})
            merged.update(stats)
            entity.stats = merged
            entity.turn_count = int(merged.get("turns", entity.turn_count) or 0)
            entity.barge_in_count = int(merged.get("barge_ins", entity.barge_in_count) or 0)
            entity.input_tokens = int(merged.get("input_tokens", entity.input_tokens) or 0)
            entity.output_tokens = int(merged.get("output_tokens", entity.output_tokens) or 0)
            entity.audio_in_ms = int(merged.get("audio_in_ms", entity.audio_in_ms) or 0)
            entity.audio_out_ms = int(merged.get("audio_out_ms", entity.audio_out_ms) or 0)
            entity.video_frames = int(merged.get("video_frames_sent", entity.video_frames) or 0)
            entity.avg_ttfa_ms = float(merged.get("avg_ttfa_ms", entity.avg_ttfa_ms) or 0.0)
        await self.session.flush()
        log.info("session_closed", session_id=session_id, stop_reason=stop_reason)
        return entity

    async def count_live(self) -> int:
        """当前占用并发额度的会话数。"""
        stmt = self.scoped(select(func.count()).select_from(SessionORM)).where(
            SessionORM.state.in_(tuple(LIVE_STATES))
        )
        result = await self.session.execute(stmt)
        return int(result.scalar_one() or 0)

    async def count_since(self, since: datetime) -> int:
        return await self.count(SessionORM.created_at >= since)

    async def list_sessions(
        self,
        *,
        state: str | None = None,
        provider: str | None = None,
        end_user_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> PageResult[SessionORM]:
        conditions = []
        if state:
            conditions.append(SessionORM.state == state)
        if provider:
            conditions.append(SessionORM.provider == provider)
        if end_user_id:
            conditions.append(SessionORM.end_user_id == end_user_id)
        if since:
            conditions.append(SessionORM.created_at >= since)
        if until:
            conditions.append(SessionORM.created_at <= until)
        return await self.paginate(*conditions, limit=limit, cursor=cursor)

    async def stale_live_sessions(self, idle_seconds: int = 3600) -> list[SessionORM]:
        """长时间未收尾的"僵尸"会话，供 maintenance 任务回收。"""
        threshold = utc_now() - timedelta(seconds=idle_seconds)
        stmt = (
            self.scoped()
            .where(SessionORM.state.in_(tuple(LIVE_STATES)))
            .where(SessionORM.created_at < threshold)
            .order_by(SessionORM.created_at.asc())
            .limit(500)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    # ---------------- 轮次 ----------------

    async def next_turn_index(self, session_id: str) -> int:
        stmt = (
            select(func.coalesce(func.max(Turn.index), -1))
            .where(Turn.session_id == session_id)
            .where(Turn.tenant_id == self.tenant_id)
        )
        result = await self.session.execute(stmt)
        return int(result.scalar_one() or -1) + 1

    async def create_turn(
        self,
        session_id: str,
        *,
        role: str = "user",
        index: int | None = None,
        text: str = "",
        state: str = "idle",
        turn_id: str | None = None,
    ) -> Turn:
        await self.require(session_id)
        turn_index = index if index is not None else await self.next_turn_index(session_id)
        existing = await self.session.execute(
            select(Turn)
            .where(Turn.tenant_id == self.tenant_id)
            .where(Turn.session_id == session_id)
            .where(Turn.index == turn_index)
        )
        if existing.scalar_one_or_none() is not None:
            raise ConflictError(
                f"turn index {turn_index} already exists in {session_id}",
                code=ErrorCode.CONFLICT,
            )
        turn = Turn(
            tenant_id=self.tenant_id,
            session_id=session_id,
            index=turn_index,
            role=role,
            state=state,
            text=text,
            started_at=utc_now(),
        )
        if turn_id:
            turn.id = turn_id
        self.session.add(turn)
        await self.session.flush()
        return turn

    async def get_turn(self, turn_id: str) -> Turn | None:
        stmt = select(Turn).where(Turn.tenant_id == self.tenant_id).where(Turn.id == turn_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def require_turn(self, turn_id: str) -> Turn:
        turn = await self.get_turn(turn_id)
        if turn is None:
            raise NotFoundError(f"turn not found: {turn_id}", code=ErrorCode.NOT_FOUND)
        return turn

    async def finish_turn(
        self,
        turn_id: str,
        *,
        text: str | None = None,
        state: str = "idle",
        stop_reason: str = "completed",
        metrics: dict[str, Any] | None = None,
        audio_ms: int | None = None,
        interrupted_at_ms: int | None = None,
    ) -> Turn:
        turn = await self.require_turn(turn_id)
        if text is not None:
            turn.text = text
        if audio_ms is not None:
            turn.audio_ms = audio_ms
        if metrics:
            merged = dict(turn.metrics or {})
            merged.update(metrics)
            turn.metrics = merged
        turn.state = state
        turn.stop_reason = stop_reason
        turn.interrupted_at_ms = interrupted_at_ms
        turn.ended_at = utc_now()
        turn.sync_metrics_columns()
        await self.session.flush()
        return turn

    async def list_turns(self, session_id: str, *, limit: int = 500) -> list[Turn]:
        stmt = (
            select(Turn)
            .where(Turn.tenant_id == self.tenant_id)
            .where(Turn.session_id == session_id)
            .order_by(Turn.index.asc())
            .limit(clamp_limit(limit) if limit <= 200 else limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    # ---------------- 转写 ----------------

    async def add_transcript(
        self,
        *,
        turn_id: str,
        session_id: str,
        text: str,
        speaker: str = "user",
        language: str = "zh-CN",
        words: list[dict[str, Any]] | None = None,
        start_ms: int = 0,
        end_ms: int = 0,
        confidence: float = 1.0,
        is_final: bool = True,
        source: str = "realtime",
        engine: str | None = None,
        redacted_text: str | None = None,
        pii_hits: list[dict[str, Any]] | None = None,
    ) -> Transcript:
        entity = Transcript(
            tenant_id=self.tenant_id,
            turn_id=turn_id,
            session_id=session_id,
            speaker=speaker,
            text=text,
            redacted_text=redacted_text,
            language=language,
            words=words or [],
            start_ms=start_ms,
            end_ms=end_ms,
            confidence=confidence,
            is_final=is_final,
            source=source,
            engine=engine,
            pii_hits=pii_hits or [],
        )
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def list_transcripts(
        self, session_id: str, *, final_only: bool = True, limit: int = 2000
    ) -> list[Transcript]:
        stmt = (
            select(Transcript)
            .where(Transcript.tenant_id == self.tenant_id)
            .where(Transcript.session_id == session_id)
        )
        if final_only:
            stmt = stmt.where(Transcript.is_final.is_(True))
        stmt = stmt.order_by(Transcript.start_ms.asc(), Transcript.id.asc()).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def low_confidence_transcripts(
        self, session_id: str, *, threshold: float = 0.7, limit: int = 500
    ) -> list[Transcript]:
        stmt = (
            select(Transcript)
            .where(Transcript.tenant_id == self.tenant_id)
            .where(Transcript.session_id == session_id)
            .where(Transcript.confidence < threshold)
            .where(Transcript.source == "realtime")
            .order_by(Transcript.start_ms.asc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    # ---------------- 视觉事件 ----------------

    async def add_vision_event(
        self,
        *,
        session_id: str,
        pts_ms: int,
        description: str,
        entities: list[dict[str, Any]] | None = None,
        frame_key: str | None = None,
        keyframe_score: float = 0.0,
        scene_change: float = 0.0,
        turn_id: str | None = None,
        model: str | None = None,
        latency_ms: float = 0.0,
    ) -> VisionEvent:
        entity = VisionEvent(
            tenant_id=self.tenant_id,
            session_id=session_id,
            turn_id=turn_id,
            pts_ms=pts_ms,
            description=description,
            entities=entities or [],
            frame_key=frame_key,
            keyframe_score=keyframe_score,
            scene_change=scene_change,
            model=model,
            latency_ms=latency_ms,
        )
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def list_vision_events(self, session_id: str, *, limit: int = 1000) -> list[VisionEvent]:
        stmt = (
            select(VisionEvent)
            .where(VisionEvent.tenant_id == self.tenant_id)
            .where(VisionEvent.session_id == session_id)
            .order_by(VisionEvent.pts_ms.asc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    # ---------------- 工具调用 ----------------

    async def add_tool_call(
        self,
        *,
        turn_id: str,
        session_id: str,
        name: str,
        arguments: dict[str, Any] | None = None,
        call_id: str | None = None,
        requires_confirmation: bool = False,
    ) -> ToolCall:
        entity = ToolCall(
            tenant_id=self.tenant_id,
            turn_id=turn_id,
            session_id=session_id,
            name=name,
            arguments=arguments or {},
            state="pending",
            started_at=utc_now(),
            requires_confirmation=requires_confirmation,
        )
        if call_id:
            entity.id = call_id
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def finish_tool_call(
        self,
        call_id: str,
        *,
        ok: bool,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        error_code: str | None = None,
        elapsed_ms: float = 0.0,
    ) -> ToolCall:
        stmt = select(ToolCall).where(ToolCall.tenant_id == self.tenant_id).where(ToolCall.id == call_id)
        found = (await self.session.execute(stmt)).scalar_one_or_none()
        if found is None:
            raise NotFoundError(f"tool_call not found: {call_id}", code=ErrorCode.NOT_FOUND)
        found.ok = ok
        found.result = result
        found.error = error
        found.error_code = error_code
        found.elapsed_ms = elapsed_ms
        found.state = "succeeded" if ok else "failed"
        found.finished_at = utc_now()
        await self.session.flush()
        return found

    async def list_tool_calls(self, session_id: str, *, limit: int = 500) -> list[ToolCall]:
        stmt = (
            select(ToolCall)
            .where(ToolCall.tenant_id == self.tenant_id)
            .where(ToolCall.session_id == session_id)
            .order_by(ToolCall.id.asc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    # ---------------- 聚合 ----------------

    async def latency_summary(self, *, since: datetime | None = None) -> dict[str, float]:
        """租户维度的延迟画像，供控制台概览卡片使用。"""
        stmt = select(
            func.count(Turn.id),
            func.avg(Turn.ttfa_ms),
            func.avg(Turn.ttft_ms),
            func.avg(Turn.e2e_ms),
            func.avg(Turn.barge_in_ms),
        ).where(Turn.tenant_id == self.tenant_id)
        if since:
            stmt = stmt.where(Turn.created_at >= since)
        row = (await self.session.execute(stmt)).one()
        return {
            "turns": float(row[0] or 0),
            "avg_ttfa_ms": float(row[1] or 0.0),
            "avg_ttft_ms": float(row[2] or 0.0),
            "avg_e2e_ms": float(row[3] or 0.0),
            "avg_barge_in_ms": float(row[4] or 0.0),
        }


def create_session_repo(session: AsyncSession, tenant_id: str) -> SessionRepository:
    return SessionRepository(session, tenant_id)


__all__ = ["SessionRepository", "create_session_repo"]

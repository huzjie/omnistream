"""轮次领域模型：单轮实体与全链路延迟画像。"""
from __future__ import annotations

from datetime import datetime

from pydantic import Field

from omnistream.core.clock import utcnow
from omnistream.protocol.enums import StopReason, TurnRole, TurnState
from omnistream.schemas.base import OmniModel


class TurnMetrics(OmniModel):
    """单轮延迟画像，全部毫秒。控制台瀑布图直接消费。"""
    speech_start_ms: float | None = None      # VAD 判定用户开口
    speech_end_ms: float | None = None        # 端点检测判定说完
    asr_first_partial_ms: float | None = None
    asr_final_ms: float | None = None
    provider_request_ms: float | None = None
    ttft_ms: float | None = None              # 首个文本 token
    ttfa_ms: float | None = None              # 首个音频字节（关键 SLO）
    e2e_ms: float | None = None               # 用户说完 → 助手出声
    barge_in_ms: float | None = None          # 用户开口 → 播放停止
    tool_ms: float | None = None
    memory_ms: float | None = None
    vision_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class Turn(OmniModel):
    id: str
    session_id: str
    index: int
    role: TurnRole
    state: TurnState = TurnState.IDLE
    text: str = ""
    audio_ms: int = 0
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None
    stop_reason: StopReason | None = None
    interrupted_at_ms: int | None = None
    metrics: TurnMetrics = Field(default_factory=TurnMetrics)
    tool_call_ids: list[str] = Field(default_factory=list)
    memory_injected_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 辅助视图（非契约）
# ---------------------------------------------------------------------------


class TurnSummary(OmniModel):
    """列表接口用的轻量轮次视图，不含全文与全部指标。"""

    id: str
    session_id: str
    index: int
    role: TurnRole
    state: TurnState
    preview: str = Field(default="", description="文本前 80 字")
    audio_ms: int = 0
    ttfa_ms: float | None = None
    e2e_ms: float | None = None
    interrupted: bool = False
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None

    @classmethod
    def of(cls, turn: Turn) -> "TurnSummary":
        return cls(
            id=turn.id,
            session_id=turn.session_id,
            index=turn.index,
            role=turn.role,
            state=turn.state,
            preview=turn.text[:80],
            audio_ms=turn.audio_ms,
            ttfa_ms=turn.metrics.ttfa_ms,
            e2e_ms=turn.metrics.e2e_ms,
            interrupted=turn.stop_reason is StopReason.INTERRUPTED,
            started_at=turn.started_at,
            ended_at=turn.ended_at,
        )

"""下行事件模型（服务端 → 客户端）。

与 :mod:`omnistream.protocol.client_events` 一样，本模块只依赖 core 与
protocol.enums，保证协议层可独立分发给 SDK 使用。
"""
from __future__ import annotations

import base64
from typing import Any, Union

from pydantic import BaseModel, ConfigDict, Field

from omnistream.protocol.enums import (
    AudioEncoding,
    MemoryLayer,
    ServerEventType,
    SessionState,
    StopReason,
    ToolCallState,
    TurnRole,
    TurnState,
)
from omnistream.protocol.envelope import Envelope

__all__ = [
    "SERVER_PAYLOAD_MODELS",
    "AudioDelta",
    "AudioDone",
    "BargeInDetected",
    "ErrorOut",
    "MemoryInjected",
    "MemorySilent",
    "MetricsTick",
    "PlaybackTruncated",
    "Pong",
    "RateLimitUpdated",
    "ResponseCancelled",
    "ResponseDone",
    "ServerPayload",
    "SessionClosed",
    "SessionCreated",
    "SessionUpdated",
    "SpeechStarted",
    "SpeechStopped",
    "TextDelta",
    "TextDone",
    "ToolCallDoneOut",
    "ToolCallOut",
    "TranscriptDeltaOut",
    "TranscriptDoneOut",
    "TurnCompleted",
    "TurnStarted",
    "TurnStateChanged",
    "VisionFrameAccepted",
    "VisionObservationOut",
    "dump_server_event",
]


class ServerModel(BaseModel):
    """下行负载基类。允许追加字段（向前兼容：老客户端忽略新字段即可）。"""

    model_config = ConfigDict(extra="ignore", populate_by_name=True, use_enum_values=False)


# ---------------------------------------------------------------- 会话
class SessionCreated(ServerModel):
    session_id: str
    state: SessionState = SessionState.ACTIVE
    provider: str = "mock"
    model: str | None = None
    protocol_version: str = "oms.v1"
    config: dict[str, Any] = Field(default_factory=dict)
    server_time_ms: int = 0
    expires_in_s: int = 0


class SessionUpdated(ServerModel):
    session_id: str
    config: dict[str, Any] = Field(default_factory=dict)
    changed: list[str] = Field(default_factory=list)


class SessionClosed(ServerModel):
    session_id: str
    state: SessionState = SessionState.ENDED
    reason: str = "completed"
    duration_ms: int = 0
    stats: dict[str, Any] = Field(default_factory=dict)


class Pong(ServerModel):
    nonce: str = ""
    server_time_ms: int = 0
    rtt_hint_ms: float | None = None


# ---------------------------------------------------------------- 轮次
class TurnStarted(ServerModel):
    turn_id: str
    index: int = 0
    role: TurnRole = TurnRole.ASSISTANT
    state: TurnState = TurnState.LISTENING


class TurnStateChanged(ServerModel):
    turn_id: str
    state: TurnState
    previous: TurnState | None = None
    reason: str = ""


class TurnCompleted(ServerModel):
    turn_id: str
    index: int = 0
    role: TurnRole = TurnRole.ASSISTANT
    text: str = ""
    stop_reason: StopReason = StopReason.COMPLETED
    metrics: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------- 语音端点
class SpeechStarted(ServerModel):
    turn_id: str | None = None
    audio_start_ms: int = 0
    energy_db: float = 0.0


class SpeechStopped(ServerModel):
    turn_id: str | None = None
    audio_end_ms: int = 0
    duration_ms: int = 0


# ---------------------------------------------------------------- 转写
class WordTimingOut(ServerModel):
    word: str
    start_ms: int = 0
    end_ms: int = 0
    confidence: float = 1.0


class TranscriptDeltaOut(ServerModel):
    turn_id: str
    text: str
    is_final: bool = False
    stability: float = 0.0
    start_ms: int = 0
    end_ms: int = 0
    speaker: str = "user"
    language: str = "zh-CN"
    words: list[WordTimingOut] = Field(default_factory=list)


class TranscriptDoneOut(ServerModel):
    turn_id: str
    text: str
    language: str = "zh-CN"
    start_ms: int = 0
    end_ms: int = 0
    speaker: str = "user"
    confidence: float = 1.0


# ---------------------------------------------------------------- 生成
class TextDelta(ServerModel):
    turn_id: str
    delta: str
    index: int = 0


class TextDone(ServerModel):
    turn_id: str
    text: str


class AudioDelta(ServerModel):
    """音频增量。高频场景建议走二进制帧，本模型用于 JSON 通道与录制回放。"""

    turn_id: str
    audio: str = Field(description="base64 编码的 PCM 数据")
    pts_ms: int = 0
    sample_rate: int = 24000
    channels: int = 1
    encoding: AudioEncoding = AudioEncoding.PCM16

    @classmethod
    def from_bytes(cls, turn_id: str, data: bytes, *, pts_ms: int = 0,
                   sample_rate: int = 24000, channels: int = 1,
                   encoding: AudioEncoding = AudioEncoding.PCM16) -> AudioDelta:
        """由原始 PCM 字节构造。"""
        return cls(
            turn_id=turn_id,
            audio=base64.b64encode(data).decode("ascii"),
            pts_ms=pts_ms,
            sample_rate=sample_rate,
            channels=channels,
            encoding=encoding,
        )


class AudioDone(ServerModel):
    turn_id: str
    total_ms: int = 0
    total_bytes: int = 0


class ResponseDone(ServerModel):
    turn_id: str
    stop_reason: StopReason = StopReason.COMPLETED
    text: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)


class ResponseCancelled(ServerModel):
    turn_id: str
    reason: StopReason = StopReason.INTERRUPTED
    truncated_at_ms: int = 0


# ---------------------------------------------------------------- 打断
class BargeInDetected(ServerModel):
    turn_id: str
    detected_at_ms: int = 0
    energy_db: float = 0.0
    latency_ms: float = 0.0
    confidence: float = 1.0


class PlaybackTruncated(ServerModel):
    turn_id: str
    truncated_at_ms: int = 0
    dropped_frames: int = 0
    dropped_ms: int = 0


# ---------------------------------------------------------------- 视觉
class VisualEntityOut(ServerModel):
    label: str
    confidence: float = 0.0
    bbox: list[float] = Field(default_factory=list, description="[x, y, w, h] 归一化坐标")


class VisionObservationOut(ServerModel):
    pts_ms: int = 0
    description: str = ""
    entities: list[VisualEntityOut] = Field(default_factory=list)
    scene_change: bool = False
    latency_ms: float = 0.0


class VisionFrameAccepted(ServerModel):
    pts_ms: int = 0
    accepted: bool = True
    reason: str = ""
    keyframe_score: float = 0.0
    fps_effective: float = 0.0


# ---------------------------------------------------------------- 工具
class ToolCallOut(ServerModel):
    call_id: str
    turn_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    state: ToolCallState = ToolCallState.PENDING
    requires_confirmation: bool = False
    executed_by: str = Field(default="client", description="client | server")


class ToolCallDoneOut(ServerModel):
    call_id: str
    turn_id: str
    name: str = ""
    state: ToolCallState = ToolCallState.SUCCEEDED
    ok: bool = True
    content: Any = None
    error: str | None = None
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------- 记忆
class MemoryInjected(ServerModel):
    turn_id: str
    memory_id: str
    layer: MemoryLayer = MemoryLayer.SESSION
    rendered: str = ""
    relevance: float = 0.0
    novelty: float = 0.0
    reason: str = ""
    budget_left: int = 0


class MemorySilent(ServerModel):
    """"保持沉默"也要上报 —— 这是记忆治理可解释性的关键。"""

    turn_id: str
    reason: str = ""
    top_candidate: str | None = None
    relevance: float = 0.0
    novelty: float = 0.0
    budget_left: int = 0


# ---------------------------------------------------------------- 观测
class MetricsTick(ServerModel):
    turn_id: str | None = None
    ttft_ms: float | None = None
    ttfa_ms: float | None = None
    e2e_ms: float | None = None
    barge_in_ms: float | None = None
    audio_in_ms: int = 0
    audio_out_ms: int = 0
    queue_depth: int = 0
    dropped_frames: int = 0


class RateLimitUpdated(ServerModel):
    limit: int = 0
    remaining: int = 0
    reset_at_ms: int = 0
    scope: str = "tenant"


class ErrorOut(ServerModel):
    code: str
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)
    recoverable: bool = False
    request_id: str | None = None


ServerPayload = Union[  # noqa: UP007 - 显式 Union 便于 TS SDK 生成判别联合
    SessionCreated, SessionUpdated, SessionClosed, Pong,
    TurnStarted, TurnStateChanged, TurnCompleted,
    SpeechStarted, SpeechStopped,
    TranscriptDeltaOut, TranscriptDoneOut,
    TextDelta, TextDone, AudioDelta, AudioDone, ResponseDone, ResponseCancelled,
    BargeInDetected, PlaybackTruncated,
    VisionObservationOut, VisionFrameAccepted,
    ToolCallOut, ToolCallDoneOut,
    MemoryInjected, MemorySilent,
    MetricsTick, RateLimitUpdated, ErrorOut,
]

#: 事件类型 -> 负载模型。新增下行事件必须在此登记（``schema_export`` 依赖它）。
SERVER_PAYLOAD_MODELS: dict[ServerEventType, type[ServerModel]] = {
    ServerEventType.SESSION_CREATED: SessionCreated,
    ServerEventType.SESSION_UPDATED: SessionUpdated,
    ServerEventType.SESSION_CLOSED: SessionClosed,
    ServerEventType.PONG: Pong,
    ServerEventType.TURN_STARTED: TurnStarted,
    ServerEventType.TURN_STATE: TurnStateChanged,
    ServerEventType.TURN_COMPLETED: TurnCompleted,
    ServerEventType.SPEECH_STARTED: SpeechStarted,
    ServerEventType.SPEECH_STOPPED: SpeechStopped,
    ServerEventType.TRANSCRIPT_DELTA: TranscriptDeltaOut,
    ServerEventType.TRANSCRIPT_DONE: TranscriptDoneOut,
    ServerEventType.RESPONSE_TEXT_DELTA: TextDelta,
    ServerEventType.RESPONSE_TEXT_DONE: TextDone,
    ServerEventType.RESPONSE_AUDIO_DELTA: AudioDelta,
    ServerEventType.RESPONSE_AUDIO_DONE: AudioDone,
    ServerEventType.RESPONSE_DONE: ResponseDone,
    ServerEventType.RESPONSE_CANCELLED: ResponseCancelled,
    ServerEventType.BARGE_IN: BargeInDetected,
    ServerEventType.PLAYBACK_TRUNCATED: PlaybackTruncated,
    ServerEventType.VISION_OBSERVATION: VisionObservationOut,
    ServerEventType.VISION_FRAME_ACCEPTED: VisionFrameAccepted,
    ServerEventType.TOOL_CALL: ToolCallOut,
    ServerEventType.TOOL_CALL_DONE: ToolCallDoneOut,
    ServerEventType.MEMORY_INJECTED: MemoryInjected,
    ServerEventType.MEMORY_SILENT: MemorySilent,
    ServerEventType.METRICS_TICK: MetricsTick,
    ServerEventType.RATE_LIMIT: RateLimitUpdated,
    ServerEventType.ERROR: ErrorOut,
}


def dump_server_event(
    event_type: ServerEventType,
    payload: ServerModel | dict[str, Any] | None = None,
    *,
    session_id: str | None = None,
    seq: int | None = None,
) -> Envelope:
    """把下行负载包装成 :class:`Envelope`。

    传入 ``dict`` 时会先用登记的模型做一次校验，避免服务端发出结构错误的事件。
    """
    if payload is None:
        body: dict[str, Any] = {}
    elif isinstance(payload, dict):
        model = SERVER_PAYLOAD_MODELS.get(event_type)
        body = (
            model.model_validate(payload).model_dump(mode="json", exclude_none=True)
            if model is not None
            else payload
        )
    else:
        body = payload.model_dump(mode="json", exclude_none=True)
    return Envelope(type=str(event_type), payload=body, session_id=session_id, seq=seq)

"""会话领域模型：会话配置（VAD/打断/记忆/视觉/工具）、会话实体与统计。"""
from __future__ import annotations

from datetime import datetime

from pydantic import Field

from omnistream.core.clock import utcnow
from omnistream.protocol.enums import Modality, SessionState
from omnistream.schemas.audio import AudioFormat
from omnistream.schemas.base import OmniModel
from omnistream.schemas.tool import ToolSpec
from omnistream.schemas.video import VideoSamplingPolicy


class VADConfig(OmniModel):
    backend: str = Field(default="auto", description="auto|silero|webrtc|energy|ensemble")
    threshold: float = 0.5
    min_speech_ms: int = 120
    min_silence_ms: int = 480
    prefix_padding_ms: int = 300
    adaptive_silence: bool = True


class BargeInConfig(OmniModel):
    enabled: bool = True
    mode: str = Field(default="immediate", description="immediate|finish_sentence|off")
    energy_threshold_db: float = -38.0
    min_duration_ms: int = 160
    cooldown_ms: int = 700


class MemoryConfig(OmniModel):
    enabled: bool = True
    layers: list[str] = Field(default_factory=lambda: ["session", "private_state",
                                                       "stable_knowledge", "procedural"])
    max_injections_per_turn: int = 1
    max_injections_per_session: int = 6
    relevance_threshold: float = 0.62
    novelty_threshold: float = 0.35


class SessionConfig(OmniModel):
    provider: str = "mock"
    model: str | None = None
    modalities: list[Modality] = Field(default_factory=lambda: [Modality.TEXT, Modality.AUDIO])
    instructions: str = "你是一个自然、简洁、可被随时打断的语音助手。"
    voice: str = "default"
    language: str = "zh-CN"
    input_audio: AudioFormat = Field(default_factory=AudioFormat)
    output_audio: AudioFormat = Field(default_factory=lambda: AudioFormat(sample_rate=24000))
    vad: VADConfig = Field(default_factory=VADConfig)
    barge_in: BargeInConfig = Field(default_factory=BargeInConfig)
    vision: VideoSamplingPolicy = Field(default_factory=VideoSamplingPolicy)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tools: list[ToolSpec] = Field(default_factory=list)
    tool_choice: str = "auto"
    temperature: float = 0.8
    max_turn_ms: int = 120_000
    idle_timeout_ms: int = 300_000
    proactive_enabled: bool = False
    proactive_silence_ms: int = 8_000
    record: bool = False
    redact_pii: bool = True


class SessionStats(OmniModel):
    turns: int = 0
    barge_ins: int = 0
    tool_calls: int = 0
    memory_injections: int = 0
    audio_in_ms: int = 0
    audio_out_ms: int = 0
    video_frames_sent: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    avg_ttfa_ms: float = 0.0
    p95_ttfa_ms: float = 0.0


class Session(OmniModel):
    id: str
    tenant_id: str
    config: SessionConfig
    state: SessionState = SessionState.CREATED
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    stop_reason: str | None = None
    client_ip: str | None = None
    user_agent: str | None = None
    stats: SessionStats = Field(default_factory=SessionStats)
    metadata: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 请求/响应 DTO（非契约，供 REST 层复用）
# ---------------------------------------------------------------------------


class SessionCreate(OmniModel):
    """``POST /v1/sessions`` 请求体。"""

    config: SessionConfig = Field(default_factory=SessionConfig)
    metadata: dict[str, str] = Field(default_factory=dict)
    #: 客户端自带的幂等键，重复提交返回同一会话
    idempotency_key: str | None = None


class SessionUpdate(OmniModel):
    """会话中途更新：只允许热更新的子集。"""

    instructions: str | None = None
    voice: str | None = None
    language: str | None = None
    temperature: float | None = None
    modalities: list[Modality] | None = None
    vad: VADConfig | None = None
    barge_in: BargeInConfig | None = None
    vision: VideoSamplingPolicy | None = None
    memory: MemoryConfig | None = None
    tools: list[ToolSpec] | None = None
    tool_choice: str | None = None
    proactive_enabled: bool | None = None

    def apply_to(self, config: SessionConfig) -> SessionConfig:
        """把非空字段合并进现有配置，返回新的配置对象。"""
        patch = self.model_dump(exclude_none=True)
        return config.model_copy(update=patch, deep=True)


class SessionToken(OmniModel):
    """短时效客户端连接凭证，避免把 API Key 下发到浏览器。"""

    session_id: str
    token: str
    ws_url: str
    expires_at: datetime
    ice_servers: list[dict[str, object]] = Field(default_factory=list)

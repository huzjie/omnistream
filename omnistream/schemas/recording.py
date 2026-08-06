"""录制与回放领域模型。"""
from __future__ import annotations

from datetime import datetime

from pydantic import Field

from omnistream.core.clock import utcnow
from omnistream.schemas.base import OmniModel


class RecordingSegment(OmniModel):
    """录制切片。长会话按时长切片，便于并发上传与断点续传。"""

    index: int = 0
    kind: str = Field(default="audio", description="audio|video|events")
    key: str = Field(description="对象存储 key")
    start_ms: int = 0
    end_ms: int = 0
    bytes: int = 0
    content_type: str = "audio/wav"
    checksum: str | None = Field(default=None, description="blake2b-128 十六进制")

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


class Recording(OmniModel):
    """一次会话的录制资产集合。"""

    id: str
    session_id: str
    tenant_id: str
    status: str = Field(default="recording", description="recording|completed|failed|deleted")
    audio_key: str | None = None
    video_key: str | None = None
    events_key: str | None = Field(default=None, description="全量事件 JSONL，用于精确回放")
    transcript_key: str | None = None
    segments: list[RecordingSegment] = Field(default_factory=list)
    duration_ms: int = 0
    bytes: int = 0
    sample_rate: int = 24000
    channels: int = 1
    encrypted: bool = False
    redacted: bool = Field(default=False, description="是否已按租户策略做 PII 脱敏")
    retention_until: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)
    completed_at: datetime | None = None
    error: str | None = None

    @property
    def expired(self) -> bool:
        return self.retention_until is not None and utcnow() >= self.retention_until

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000.0

    def segments_of(self, kind: str) -> list[RecordingSegment]:
        return sorted((s for s in self.segments if s.kind == kind), key=lambda s: s.index)


class ReplayEvent(OmniModel):
    """回放时间轴上的一个事件点。"""

    t_ms: int = 0
    type: str = ""
    turn_id: str | None = None
    label: str = ""
    payload: dict[str, object] = Field(default_factory=dict)


class ReplayManifest(OmniModel):
    """回放清单。控制台据此在时间轴上同步音频、字幕、指标与打断标记。"""

    session_id: str
    recording_id: str
    duration_ms: int = 0
    audio_url: str | None = None
    video_url: str | None = None
    transcript_url: str | None = None
    events: list[ReplayEvent] = Field(default_factory=list)
    barge_in_marks_ms: list[int] = Field(default_factory=list)
    turn_marks_ms: list[int] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = Field(default=None, description="签名 URL 过期时间")

    def add_event(self, event: ReplayEvent) -> "ReplayManifest":
        self.events.append(event)
        self.events.sort(key=lambda e: e.t_ms)
        return self

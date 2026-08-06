"""语音转写领域模型：增量、词级时间戳、分段与完整转写。"""
from __future__ import annotations

from datetime import datetime

from pydantic import Field

from omnistream.core.clock import utcnow
from omnistream.schemas.base import OmniModel


class WordTiming(OmniModel):
    """词级时间戳。用于控制台高亮、字幕对齐与打断点定位。"""

    word: str
    start_ms: int = 0
    end_ms: int = 0
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


class TranscriptDelta(OmniModel):
    """流式转写增量。``is_final=False`` 为可被覆盖的临时结果。"""

    turn_id: str = ""
    speaker: str = Field(default="user", description="user|assistant")
    text: str = ""
    is_final: bool = False
    #: 从本段语音起点算起的偏移
    start_ms: int = 0
    end_ms: int = 0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    language: str = "zh-CN"
    #: 增量在本段中的序号，客户端据此丢弃乱序包
    index: int = 0
    stability: float = Field(default=0.0, ge=0.0, le=1.0,
                             description="该 partial 被后续覆盖的概率的反面")


class TranscriptSegment(OmniModel):
    """一段已定稿的转写（通常对应一次连续说话）。"""

    id: str = ""
    turn_id: str = ""
    speaker: str = "user"
    text: str = ""
    language: str = "zh-CN"
    start_ms: int = 0
    end_ms: int = 0
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    words: list[WordTiming] = Field(default_factory=list)
    engine: str = Field(default="mock", description="产出该段的 ASR 引擎名")
    redacted: bool = Field(default=False, description="是否已做 PII 脱敏")

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def char_per_second(self) -> float:
        secs = self.duration_ms / 1000.0
        return len(self.text) / secs if secs > 0 else 0.0


class Transcript(OmniModel):
    """一次会话的完整转写。回放、质检、导出共用。"""

    session_id: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
    language: str = "zh-CN"
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @property
    def duration_ms(self) -> int:
        return max((s.end_ms for s in self.segments), default=0)

    @property
    def text(self) -> str:
        """带说话人前缀的可读全文。"""
        return "\n".join(f"[{s.speaker}] {s.text}" for s in self.segments if s.text)

    def by_speaker(self, speaker: str) -> list[TranscriptSegment]:
        return [s for s in self.segments if s.speaker == speaker]

    def append(self, segment: TranscriptSegment) -> "Transcript":
        self.segments.append(segment)
        self.updated_at = utcnow()
        return self

    def to_srt(self) -> str:
        """导出 SRT 字幕，便于人工复核与外部工具消费。"""
        def _ts(ms: int) -> str:
            h, rem = divmod(max(0, ms), 3_600_000)
            m, rem = divmod(rem, 60_000)
            s, milli = divmod(rem, 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"

        lines: list[str] = []
        for i, seg in enumerate(self.segments, start=1):
            lines.append(str(i))
            lines.append(f"{_ts(seg.start_ms)} --> {_ts(seg.end_ms)}")
            lines.append(f"{seg.speaker}: {seg.text}")
            lines.append("")
        return "\n".join(lines)

"""EndpointDetector —— 语音起点/终点判定。

判定规则（对齐 §4.1 时序图）::

    起点 SPEECH_START：
        连续语音累计 >= min_speech_ms  →  上报起点
        上报的 pts 会往前回溯 prefix_padding_ms，保证 ASR 不丢首音

    终点 SPEECH_END：
        进入语音后，连续静音累计 >= effective_silence_ms  →  上报终点
        effective_silence_ms = 自适应尾静音(语速/历史)  +  语义补偿(半句不截断)

检测器是**纯同步状态机**，不做任何 IO，可直接跑在音频回调路径上。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import structlog

from omnistream.duplex.endpointing.adaptive import AdaptiveSilence
from omnistream.duplex.endpointing.semantic import SemanticEndpointer
from omnistream.duplex.vad.base import VADResult
from omnistream.schemas.session import VADConfig

log = structlog.get_logger(__name__)


class EndpointEventKind(StrEnum):
    """端点事件类型。"""

    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass(slots=True, frozen=True)
class EndpointEvent:
    """端点检测输出。"""

    kind: EndpointEventKind
    pts_ms: int
    speech_start_pts_ms: int = 0
    speech_duration_ms: float = 0.0
    silence_ms: float = 0.0
    effective_silence_ms: int = 0
    reason: str = ""
    completeness: float = 0.0
    details: dict[str, float] = field(default_factory=dict)


class EndpointDetector:
    """语音端点检测器。"""

    def __init__(
        self,
        config: VADConfig | None = None,
        *,
        adaptive: AdaptiveSilence | None = None,
        semantic: SemanticEndpointer | None = None,
        language: str = "zh-CN",
    ) -> None:
        cfg = config or VADConfig()
        self.min_speech_ms = int(cfg.min_speech_ms)
        self.min_silence_ms = int(cfg.min_silence_ms)
        self.prefix_padding_ms = int(cfg.prefix_padding_ms)
        self.adaptive_enabled = bool(cfg.adaptive_silence)
        self.language = language

        self.adaptive = adaptive or AdaptiveSilence(base_ms=self.min_silence_ms)
        self.semantic = semantic or SemanticEndpointer()

        self._in_speech = False
        self._pending_speech_ms = 0.0
        self._pending_start_pts = 0
        self._speech_start_pts = 0
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._last_pts = 0
        self._partial_text = ""
        self._effective_silence_ms = self.min_silence_ms

    # ------------------------------------------------------------------
    # 输入
    # ------------------------------------------------------------------

    def set_partial_text(self, text: str) -> None:
        """喂入 ASR 增量文本，供语义端点与语速估计使用。"""
        self._partial_text = text or ""

    def process(self, result: VADResult) -> EndpointEvent | None:
        """处理一帧 VAD 结果，返回端点事件（多数帧返回 None）。"""
        frame_ms = float(result.duration_ms) or 20.0
        self._last_pts = int(result.pts_ms)

        if result.is_speech:
            return self._on_speech_frame(frame_ms)
        return self._on_silence_frame(frame_ms)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _on_speech_frame(self, frame_ms: float) -> EndpointEvent | None:
        if self._in_speech:
            self._speech_ms += frame_ms
            self._silence_ms = 0.0
            return None

        if self._pending_speech_ms <= 0.0:
            self._pending_start_pts = self._last_pts
        self._pending_speech_ms += frame_ms

        if self._pending_speech_ms < self.min_speech_ms:
            return None

        self._in_speech = True
        self._speech_ms = self._pending_speech_ms
        self._silence_ms = 0.0
        self._pending_speech_ms = 0.0
        self._speech_start_pts = max(0, self._pending_start_pts - self.prefix_padding_ms)

        log.debug("endpoint_speech_started", pts_ms=self._speech_start_pts,
                  min_speech_ms=self.min_speech_ms)
        return EndpointEvent(
            kind=EndpointEventKind.SPEECH_START,
            pts_ms=self._speech_start_pts,
            speech_start_pts_ms=self._speech_start_pts,
            speech_duration_ms=self._speech_ms,
            reason="min_speech_reached",
            details={"prefix_padding_ms": float(self.prefix_padding_ms)},
        )

    def _on_silence_frame(self, frame_ms: float) -> EndpointEvent | None:
        if not self._in_speech:
            # 未成立的短促噪声，缓慢衰减，避免零散毛刺累积成起点
            self._pending_speech_ms = max(0.0, self._pending_speech_ms - frame_ms)
            return None

        self._silence_ms += frame_ms
        self._effective_silence_ms = self._compute_effective_silence()
        if self._silence_ms < self._effective_silence_ms:
            return None

        completeness = self.semantic.completeness(self._partial_text, language=self.language)
        event = EndpointEvent(
            kind=EndpointEventKind.SPEECH_END,
            pts_ms=self._last_pts,
            speech_start_pts_ms=self._speech_start_pts,
            speech_duration_ms=self._speech_ms,
            silence_ms=self._silence_ms,
            effective_silence_ms=self._effective_silence_ms,
            reason="silence_threshold_reached",
            completeness=completeness,
            details={"chars": float(len(self._partial_text))},
        )
        log.debug("endpoint_speech_stopped", pts_ms=self._last_pts,
                  speech_ms=round(self._speech_ms, 1), silence_ms=round(self._silence_ms, 1),
                  effective_silence_ms=self._effective_silence_ms, completeness=round(completeness, 3))
        self._reset_utterance()
        return event

    def _compute_effective_silence(self) -> int:
        if not self.adaptive_enabled:
            return self.min_silence_ms
        base = self.adaptive.compute(
            speech_duration_ms=self._speech_ms,
            partial_text=self._partial_text,
            language=self.language,
        )
        extra = self.semantic.extra_wait_ms(self._partial_text, language=self.language)
        return int(min(self.adaptive.max_ms, base + extra))

    def _reset_utterance(self) -> None:
        self._in_speech = False
        self._pending_speech_ms = 0.0
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._partial_text = ""

    def reset(self) -> None:
        """完全复位（新会话/新说话人）。"""
        self._reset_utterance()
        self._speech_start_pts = 0
        self._pending_start_pts = 0
        self._effective_silence_ms = self.min_silence_ms

    # ------------------------------------------------------------------
    # 只读状态
    # ------------------------------------------------------------------

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def speech_start_pts_ms(self) -> int:
        return self._speech_start_pts

    @property
    def current_speech_ms(self) -> float:
        return self._speech_ms

    @property
    def current_silence_ms(self) -> float:
        return self._silence_ms

    @property
    def effective_silence_ms(self) -> int:
        return self._effective_silence_ms

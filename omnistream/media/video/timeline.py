"""视频时间轴：与音频统一时基对齐。

问题
----
音频 pts 由**累计样本数**推出（见 :class:`~omnistream.media.audio.timeline.AudioTimeline`），
视频 pts 由**采集时刻**推出，两者的零点不一定重合：WebRTC 的音视频轨道各有独立
RTP 时间戳；WebSocket 链路里浏览器的 ``performance.now()`` 与音频上下文时钟也可能
差几十毫秒。

如果不对齐，"用户举起杯子"这个视觉事件会被打上错误的 pts，指代消解（"刚才那个红色的"）
就会检索到错的时间窗。

做法
----
:class:`VideoTimeline` 维护一个 ``offset_ms``，把视频 pts 平移到音频时基上。
偏移量的来源优先级：

1. 显式 :meth:`sync_to_audio`（信令层能拿到 RTP 映射时最准）。
2. 首帧对齐 —— 用第一帧到达时的音频 pts 作为锚点。
3. 零偏移 —— 两端本来就同源（回环、WebSocket 单通道）。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from omnistream.media.constants import CLOCK_DRIFT_WARN_MS
from omnistream.media.video.frame import RawVideoFrame

__all__ = ["VideoTimeline", "VideoTimelineStats"]


@dataclass(slots=True)
class VideoTimelineStats:
    """视频时间轴统计。"""

    frames: int = 0
    out_of_order: int = 0
    duplicates: int = 0
    gaps: int = 0
    gap_ms_total: float = 0.0
    resyncs: int = 0
    drift_warnings: int = 0
    intervals: deque[float] = field(default_factory=lambda: deque(maxlen=64), repr=False)

    @property
    def measured_fps(self) -> float:
        """按最近若干帧间隔估算的实际帧率。"""
        if not self.intervals:
            return 0.0
        average = sum(self.intervals) / len(self.intervals)
        return 1000.0 / average if average > 0 else 0.0

    def as_dict(self) -> dict[str, float]:
        """转为可序列化字典。"""
        return {
            "frames": float(self.frames),
            "out_of_order": float(self.out_of_order),
            "duplicates": float(self.duplicates),
            "gaps": float(self.gaps),
            "gap_ms_total": round(self.gap_ms_total, 2),
            "resyncs": float(self.resyncs),
            "drift_warnings": float(self.drift_warnings),
            "measured_fps": round(self.measured_fps, 3),
        }


class VideoTimeline:
    """把视频帧 pts 归一到统一（音频）时基。"""

    def __init__(
        self,
        *,
        offset_ms: int = 0,
        gap_threshold_ms: float = 1500.0,
        drift_warn_ms: float = CLOCK_DRIFT_WARN_MS,
        drop_out_of_order: bool = True,
    ) -> None:
        """初始化时间轴。

        Args:
            offset_ms: 初始偏移（视频 pts + offset = 统一 pts）。
            gap_threshold_ms: 帧间隔超过该值记为一次空洞（画面卡死/切后台）。
            drift_warn_ms: 与音频时基偏差超过该值时告警。
            drop_out_of_order: 是否丢弃乱序帧（视频乱序没有补救价值）。
        """
        self.offset_ms = int(offset_ms)
        self.gap_threshold_ms = max(0.0, float(gap_threshold_ms))
        self.drift_warn_ms = float(drift_warn_ms)
        self.drop_out_of_order = bool(drop_out_of_order)

        self._anchored = offset_ms != 0
        self._last_pts: int | None = None
        self._first_pts: int | None = None
        self.stats = VideoTimelineStats()

    # ------------------------------------------------------------------
    # 只读视图
    # ------------------------------------------------------------------

    @property
    def anchored(self) -> bool:
        """是否已完成与音频时基的锚定。"""
        return self._anchored

    @property
    def last_pts_ms(self) -> int | None:
        """最近一帧的统一 pts。"""
        return self._last_pts

    @property
    def span_ms(self) -> int:
        """已覆盖的时间跨度。"""
        if self._first_pts is None or self._last_pts is None:
            return 0
        return int(self._last_pts - self._first_pts)

    # ------------------------------------------------------------------
    # 对齐控制
    # ------------------------------------------------------------------

    def sync_to_audio(self, *, video_pts_ms: int, audio_pts_ms: int) -> int:
        """用一对已知同时刻的 pts 建立偏移。

        Returns:
            新的 ``offset_ms``。
        """
        self.offset_ms = int(audio_pts_ms) - int(video_pts_ms)
        self._anchored = True
        self.stats.resyncs += 1
        return self.offset_ms

    def anchor_first_frame(self, *, video_pts_ms: int, audio_pts_ms: int) -> int:
        """首帧锚定（仅在尚未锚定时生效）。"""
        if self._anchored:
            return self.offset_ms
        return self.sync_to_audio(video_pts_ms=video_pts_ms, audio_pts_ms=audio_pts_ms)

    def to_unified(self, video_pts_ms: int) -> int:
        """把视频原生 pts 换算成统一 pts。"""
        return int(video_pts_ms) + self.offset_ms

    def to_video(self, unified_pts_ms: int) -> int:
        """把统一 pts 换算回视频原生 pts。"""
        return int(unified_pts_ms) - self.offset_ms

    # ------------------------------------------------------------------
    # 帧处理
    # ------------------------------------------------------------------

    def observe_audio(self, audio_pts_ms: int) -> float:
        """观测当前音频 pts，返回音视频偏差（毫秒）。

        偏差 = 视频最新统一 pts - 音频 pts。正值表示视频跑在音频前面。
        """
        if self._last_pts is None:
            return 0.0
        drift = float(self._last_pts - audio_pts_ms)
        if abs(drift) >= self.drift_warn_ms:
            self.stats.drift_warnings += 1
        return drift

    def align(self, frame: RawVideoFrame) -> RawVideoFrame | None:
        """对齐一帧的 pts。

        Returns:
            pts 已归一的帧；乱序且 ``drop_out_of_order=True`` 时返回 ``None``。
        """
        unified = self.to_unified(frame.pts_ms)

        if self._last_pts is not None:
            delta = unified - self._last_pts
            if delta == 0:
                self.stats.duplicates += 1
                if self.drop_out_of_order:
                    return None
            elif delta < 0:
                self.stats.out_of_order += 1
                if self.drop_out_of_order:
                    return None
            else:
                self.stats.intervals.append(float(delta))
                if delta > self.gap_threshold_ms:
                    self.stats.gaps += 1
                    self.stats.gap_ms_total += float(delta)

        if self._first_pts is None:
            self._first_pts = unified
        self._last_pts = unified
        self.stats.frames += 1

        if unified == frame.pts_ms:
            return frame
        frame.pts_ms = unified
        return frame

    def reset(self, *, keep_offset: bool = True) -> None:
        """重置时间轴。

        Args:
            keep_offset: 是否保留已建立的音视频偏移（重协商通常要保留）。
        """
        if not keep_offset:
            self.offset_ms = 0
            self._anchored = False
        self._last_pts = None
        self._first_pts = None
        self.stats = VideoTimelineStats()

    def snapshot(self) -> dict[str, float]:
        """导出统计快照。"""
        data = self.stats.as_dict()
        data["offset_ms"] = float(self.offset_ms)
        data["anchored"] = 1.0 if self._anchored else 0.0
        data["span_ms"] = float(self.span_ms)
        data["last_pts_ms"] = float(self._last_pts if self._last_pts is not None else -1)
        return data

    def __repr__(self) -> str:
        """调试用文本表示。"""
        return (
            f"VideoTimeline(offset={self.offset_ms:+d}ms, frames={self.stats.frames}, "
            f"fps={self.stats.measured_fps:.2f}, gaps={self.stats.gaps})"
        )

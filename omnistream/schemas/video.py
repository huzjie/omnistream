"""视频领域模型：帧、采样策略、关键帧评分结果。

视频在 OmniStream 中是**低帧率感知输入**（默认 1fps），不是流媒体转发：
上行帧经关键帧筛选后交给 VLM 产出场景描述，再以文本进入对话上下文。
"""
from __future__ import annotations

from pydantic import Field, field_validator

from omnistream.core.clock import now_ns
from omnistream.core.constants import (
    DEFAULT_JPEG_QUALITY,
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    MAX_VIDEO_FPS,
)
from omnistream.protocol.enums import VideoEncoding
from omnistream.schemas.base import OmniModel


class VideoFrame(OmniModel):
    """单帧视频。``data`` 默认为 JPEG 字节，JSON 序列化时自动 base64。"""

    pts_ms: int = Field(description="相对会话起点的呈现时间戳（毫秒）")
    data: bytes = Field(repr=False)
    width: int = DEFAULT_VIDEO_WIDTH
    height: int = DEFAULT_VIDEO_HEIGHT
    encoding: VideoEncoding = VideoEncoding.JPEG
    capture_ts_ns: int = Field(default_factory=now_ns)
    keyframe_score: float = Field(default=0.0, ge=0.0, le=1.0,
                                  description="与上一帧的差异度，越大越值得送 VLM")
    source: str = Field(default="camera", description="camera|screen|file|upload")
    facing: str | None = Field(default=None, description="user|environment")

    @field_validator("width", "height")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("视频宽高必须为正整数")
        return v

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1_000_000.0


class VideoSamplingPolicy(OmniModel):
    """视觉输入采样策略。挂在 ``SessionConfig.vision`` 上，可会话中途更新。"""

    enabled: bool = False
    fps: float = Field(default=DEFAULT_VIDEO_FPS, gt=0.0, le=MAX_VIDEO_FPS)
    max_width: int = DEFAULT_VIDEO_WIDTH
    max_height: int = DEFAULT_VIDEO_HEIGHT
    jpeg_quality: int = Field(default=DEFAULT_JPEG_QUALITY, ge=10, le=100)
    keyframe_only: bool = Field(default=True, description="仅在画面显著变化时送 VLM")
    keyframe_threshold: float = Field(default=0.18, ge=0.0, le=1.0)
    #: 两次 VLM 调用之间的最小间隔，防止画面抖动导致成本飙升
    min_interval_ms: int = Field(default=1500, ge=0)
    #: 场景描述缓存窗口，同一场景内复用描述
    describe_cache_ms: int = Field(default=4000, ge=0)
    max_frames_per_turn: int = Field(default=4, ge=1)
    #: 是否在无人说话时也持续观察（"边看边听"的看）
    observe_while_idle: bool = True

    @property
    def interval_ms(self) -> int:
        """按 fps 换算的采样间隔（毫秒）。"""
        return int(1000.0 / self.fps)

    def should_sample(self, last_sent_pts_ms: int, pts_ms: int) -> bool:
        """基于时间间隔判断该帧是否进入候选（关键帧判定另行叠加）。"""
        return pts_ms - last_sent_pts_ms >= max(self.interval_ms, self.min_interval_ms)

    def should_describe(self, frame: VideoFrame) -> bool:
        """在通过时间闸门后，判断是否值得调用 VLM。"""
        if not self.keyframe_only:
            return True
        return frame.keyframe_score >= self.keyframe_threshold


class VideoStats(OmniModel):
    """视觉链路统计，供控制台与计量使用。"""

    frames_received: int = 0
    frames_sampled: int = 0
    frames_described: int = 0
    frames_dropped: int = 0
    bytes_received: int = 0
    avg_keyframe_score: float = 0.0
    last_pts_ms: int = 0

    def observe(self, frame: VideoFrame, *, sampled: bool = False,
                described: bool = False) -> "VideoStats":
        self.frames_received += 1
        self.bytes_received += frame.size_bytes
        self.last_pts_ms = frame.pts_ms
        if sampled:
            self.frames_sampled += 1
        else:
            self.frames_dropped += 1
        if described:
            self.frames_described += 1
        n = self.frames_received
        self.avg_keyframe_score += (frame.keyframe_score - self.avg_keyframe_score) / n
        return self

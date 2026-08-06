"""媒体层配置：采样率、帧长、抖动缓冲、WebRTC/TURN、视频采样与码率。"""
from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from omnistream.core.constants import (
    DEFAULT_FRAME_MS,
    DEFAULT_INPUT_SAMPLE_RATE,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_OUTPUT_SAMPLE_RATE,
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    MAX_VIDEO_FPS,
    SUPPORTED_SAMPLE_RATES,
)


class MediaSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OMS_MEDIA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------ 音频
    input_sample_rate: int = DEFAULT_INPUT_SAMPLE_RATE
    output_sample_rate: int = DEFAULT_OUTPUT_SAMPLE_RATE
    channels: int = Field(default=1, ge=1, le=2)
    frame_ms: int = Field(default=DEFAULT_FRAME_MS, ge=10, le=60)
    #: 上行音频编码：pcm16 最通用；opus 省带宽但需 opuslib
    input_encoding: str = Field(default="pcm16", description="pcm16|opus|mulaw")
    output_encoding: str = Field(default="pcm16", description="pcm16|opus|mulaw")
    opus_bitrate: int = Field(default=24_000, ge=6_000, le=128_000)
    opus_complexity: int = Field(default=5, ge=0, le=10)

    # ------------------------------------------------------------ 抖动缓冲
    jitter_target_ms: int = Field(default=60, ge=0, le=500,
                                  description="目标缓冲深度，越小越低延迟越易卡顿")
    jitter_max_ms: int = Field(default=240, ge=20, le=2000)
    jitter_adaptive: bool = True
    #: 播放缓冲低于此值时插入舒适噪声而非静音，避免"断线感"
    comfort_noise_dbfs: float = Field(default=-62.0, le=0.0)

    # ------------------------------------------------------------ 环形缓冲
    ring_buffer_seconds: float = Field(default=30.0, gt=0.0, le=600.0)

    # ------------------------------------------------------------ 增益 / 处理
    agc_enabled: bool = True
    agc_target_dbfs: float = Field(default=-20.0, le=0.0)
    agc_max_gain_db: float = Field(default=18.0, ge=0.0)
    denoise_enabled: bool = False
    dc_offset_removal: bool = True

    # ------------------------------------------------------------ WebRTC / TURN
    webrtc_enabled: bool = False
    stun_servers: list[str] = Field(default_factory=lambda: ["stun:stun.l.google.com:19302"])
    turn_url: str | None = None
    turn_username: str | None = None
    turn_password: str | None = Field(default=None, repr=False)
    turn_ttl_s: int = Field(default=3600, ge=60)
    ice_gather_timeout_s: float = Field(default=5.0, gt=0.0)

    # ------------------------------------------------------------ 视频
    video_enabled: bool = False
    video_fps: float = Field(default=DEFAULT_VIDEO_FPS, gt=0.0, le=MAX_VIDEO_FPS)
    video_max_width: int = Field(default=DEFAULT_VIDEO_WIDTH, ge=64)
    video_max_height: int = Field(default=DEFAULT_VIDEO_HEIGHT, ge=64)
    video_jpeg_quality: int = Field(default=DEFAULT_JPEG_QUALITY, ge=10, le=100)
    video_min_bitrate: int = Field(default=150_000, ge=10_000)
    video_max_bitrate: int = Field(default=1_500_000, ge=50_000)
    video_keyframe_threshold: float = Field(default=0.18, ge=0.0, le=1.0)

    # ------------------------------------------------------------ 录制
    record_enabled: bool = False
    record_format: str = Field(default="wav", description="wav|flac|opus")
    record_segment_seconds: int = Field(default=300, ge=10)

    @field_validator("input_sample_rate", "output_sample_rate")
    @classmethod
    def _sr(cls, v: int) -> int:
        if v not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"不支持的采样率 {v}，可选 {SUPPORTED_SAMPLE_RATES}")
        return v

    @property
    def input_frame_samples(self) -> int:
        return self.input_sample_rate * self.frame_ms // 1000

    @property
    def input_frame_bytes(self) -> int:
        return self.input_frame_samples * 2 * self.channels

    @property
    def output_frame_samples(self) -> int:
        return self.output_sample_rate * self.frame_ms // 1000

    @property
    def output_frame_bytes(self) -> int:
        return self.output_frame_samples * 2 * self.channels

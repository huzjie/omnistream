"""音频领域模型：格式描述、最小传输单元 AudioChunk、统计量。"""
from __future__ import annotations

import numpy as np
from pydantic import Field, field_validator

from omnistream.core.clock import now_ns
from omnistream.protocol.enums import AudioEncoding
from omnistream.schemas.base import OmniModel


class AudioFormat(OmniModel):
    encoding: AudioEncoding = AudioEncoding.PCM16
    sample_rate: int = 16000
    channels: int = 1

    @field_validator("sample_rate")
    @classmethod
    def _sr(cls, v: int) -> int:
        if v not in (8000, 16000, 22050, 24000, 44100, 48000):
            raise ValueError(f"不支持的采样率: {v}")
        return v

    @property
    def bytes_per_sample(self) -> int:
        return 4 if self.encoding == AudioEncoding.PCM_F32 else 2


class AudioChunk(OmniModel):
    """流经全链路的最小音频单元，默认 20ms。"""

    pts_ms: int = Field(description="相对会话起点的呈现时间戳（毫秒）")
    data: bytes = Field(repr=False)
    sample_rate: int = 16000
    channels: int = 1
    encoding: AudioEncoding = AudioEncoding.PCM16
    capture_ts_ns: int = Field(default_factory=now_ns, description="本地单调时钟采集时刻")
    speaker: str | None = None

    @property
    def num_samples(self) -> int:
        width = 4 if self.encoding == AudioEncoding.PCM_F32 else 2
        return len(self.data) // (width * self.channels)

    @property
    def duration_ms(self) -> float:
        return self.num_samples * 1000.0 / self.sample_rate

    def to_ndarray(self) -> np.ndarray:
        if self.encoding == AudioEncoding.PCM16:
            arr = np.frombuffer(self.data, dtype="<i2").astype(np.float32) / 32768.0
        elif self.encoding == AudioEncoding.PCM_F32:
            arr = np.frombuffer(self.data, dtype="<f4").copy()
        else:
            raise ValueError(f"{self.encoding} 需先解码为 PCM")
        return arr.reshape(-1, self.channels) if self.channels > 1 else arr

    @classmethod
    def from_ndarray(cls, arr: np.ndarray, *, pts_ms: int, sample_rate: int = 16000,
                     channels: int = 1) -> "AudioChunk":
        pcm = np.clip(arr.reshape(-1), -1.0, 1.0)
        data = (pcm * 32767.0).astype("<i2").tobytes()
        return cls(pts_ms=pts_ms, data=data, sample_rate=sample_rate,
                   channels=channels, encoding=AudioEncoding.PCM16)


# ---------------------------------------------------------------------------
# 统计量（非契约，供媒体层/控制台消费）
# ---------------------------------------------------------------------------


class AudioStats(OmniModel):
    """一段音频流的滚动统计，用于控制台电平表与质量诊断。"""

    chunks: int = 0
    samples: int = 0
    duration_ms: float = 0.0
    rms: float = 0.0
    peak: float = 0.0
    dbfs: float = -120.0
    clipped_samples: int = 0
    silence_ms: float = 0.0
    dropped_chunks: int = 0
    late_chunks: int = 0
    last_pts_ms: int = 0

    def observe(self, chunk: AudioChunk) -> "AudioStats":
        """吸收一个音频块并就地更新统计量，返回自身便于链式调用。"""
        arr = chunk.to_ndarray().reshape(-1)
        n = int(arr.size)
        if n == 0:
            return self
        if chunk.pts_ms < self.last_pts_ms:
            self.late_chunks += 1
        self.chunks += 1
        self.samples += n
        self.duration_ms += chunk.duration_ms
        self.last_pts_ms = chunk.pts_ms
        block_rms = float(np.sqrt(np.mean(np.square(arr, dtype=np.float64))))
        block_peak = float(np.max(np.abs(arr)))
        # 能量域加权平均，保证长流统计稳定
        prev_weight = (self.samples - n) / self.samples
        self.rms = float(np.sqrt(prev_weight * self.rms**2 + (n / self.samples) * block_rms**2))
        self.peak = max(self.peak, block_peak)
        self.dbfs = 20.0 * float(np.log10(self.rms)) if self.rms > 1e-9 else -120.0
        self.clipped_samples += int(np.count_nonzero(np.abs(arr) >= 0.999))
        if block_rms < 10 ** (-50.0 / 20.0):
            self.silence_ms += chunk.duration_ms
        return self

    def reset(self) -> None:
        """清零所有累计量。"""
        for name, field in type(self).model_fields.items():
            setattr(self, name, field.get_default(call_default_factory=True))

"""声学回声消除（AEC）抽象接口。

全双工的物理前提是"麦克风听不见自己"。浏览器侧（``getUserMedia`` 的
``echoCancellation: true``）已经做得相当好，所以默认实现是
:class:`~omnistream.duplex.aec.passthrough.PassthroughAEC`；
服务端混音、电话网关、硬件终端等场景才需要真正的 AEC。

统一接口::

    aec.push_reference(playback_chunk)          # 播放侧持续写入参考信号
    clean = aec.process(mic_chunk)              # 采集侧逐帧消除

``process`` 必须是**同步、无阻塞 IO**：它位于 20ms 音频主链路上，
任何 await 都会引入抖动。
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from omnistream.protocol.enums import AudioEncoding
from omnistream.schemas.audio import AudioChunk

EPS: Final[float] = 1e-12

#: 默认帧长与采样率，与 VAD 保持一致。
DEFAULT_SAMPLE_RATE: Final[int] = 16000
DEFAULT_FRAME_MS: Final[int] = 20


class AECUnavailable(RuntimeError):
    """AEC 后端依赖缺失时由构造函数抛出，供工厂降级到 passthrough。"""


@dataclass(slots=True)
class AECStats:
    """AEC 运行统计，供 /metrics 与调参使用。"""

    frames_processed: int = 0
    reference_frames: int = 0
    #: 参考信号缺失（欠载）的帧数——数值大说明播放侧没有正确回灌
    reference_underruns: int = 0
    #: 平滑后的回声抑制比（Echo Return Loss Enhancement，dB），越大越好
    erle_db: float = 0.0
    #: 最近一帧的麦克风能量（dBFS）
    mic_db: float = -120.0
    #: 最近一帧消除后的能量（dBFS）
    clean_db: float = -120.0
    errors: int = 0

    def as_dict(self) -> dict[str, float]:
        return {
            "frames_processed": float(self.frames_processed),
            "reference_frames": float(self.reference_frames),
            "reference_underruns": float(self.reference_underruns),
            "erle_db": round(self.erle_db, 2),
            "mic_db": round(self.mic_db, 2),
            "clean_db": round(self.clean_db, 2),
            "errors": float(self.errors),
        }


def energy_dbfs(samples: np.ndarray) -> float:
    """一维 float32 [-1,1] 数组的 RMS dBFS。"""
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)) + EPS))
    if rms <= EPS:
        return -120.0
    return float(max(-120.0, 20.0 * math.log10(rms)))


class BaseAEC(abc.ABC):
    """回声消除抽象基类。"""

    #: 注册名，用于配置选择
    name: str = "base"

    def __init__(self, *, sample_rate: int = DEFAULT_SAMPLE_RATE,
                 frame_ms: int = DEFAULT_FRAME_MS, channels: int = 1) -> None:
        self.sample_rate = int(sample_rate)
        self.frame_ms = int(frame_ms)
        self.channels = int(channels)
        self.stats = AECStats()
        self._erle_alpha = 0.05

    # ------------------------------------------------------------------
    # 尺寸
    # ------------------------------------------------------------------

    @property
    def frame_samples(self) -> int:
        """单帧样本数（单声道计）。"""
        return int(self.sample_rate * self.frame_ms / 1000)

    # ------------------------------------------------------------------
    # 子类实现
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def process(self, mic: AudioChunk, ref: AudioChunk | None = None) -> AudioChunk:
        """消除回声，返回干净的麦克风帧。

        Args:
            mic: 麦克风采集帧。
            ref: 对齐后的参考帧；``None`` 时实现应从内部参考缓冲取。

        Returns:
            与 ``mic`` 同 pts、同采样率的干净音频帧。
        """

    def push_reference(self, chunk: AudioChunk) -> None:
        """写入播放侧参考信号。默认实现只计数（passthrough 用）。"""
        self.stats.reference_frames += 1

    # ------------------------------------------------------------------
    # 通用能力
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """后端依赖是否就绪。子类按需覆盖。"""
        return True

    def reset(self) -> None:
        """清空自适应滤波器状态与参考缓冲。"""
        self.stats = AECStats()

    def close(self) -> None:
        """释放原生资源，幂等。"""
        return None

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def to_mono_float(chunk: AudioChunk) -> np.ndarray:
        """AudioChunk → 一维 float32 单声道数组。"""
        arr = chunk.to_ndarray()
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        return np.asarray(arr, dtype=np.float32).reshape(-1)

    def _rebuild(self, mic: AudioChunk, samples: np.ndarray) -> AudioChunk:
        """用处理后的样本重建 AudioChunk，保持 pts/采样率/说话人不变。"""
        clean = AudioChunk.from_ndarray(
            np.clip(samples, -1.0, 1.0),
            pts_ms=mic.pts_ms,
            sample_rate=mic.sample_rate,
            channels=1,
        )
        clean.capture_ts_ns = mic.capture_ts_ns
        clean.speaker = mic.speaker
        return clean

    def _account(self, mic_samples: np.ndarray, clean_samples: np.ndarray) -> None:
        """更新 ERLE 与能量统计。"""
        self.stats.frames_processed += 1
        mic_db = energy_dbfs(mic_samples)
        clean_db = energy_dbfs(clean_samples)
        self.stats.mic_db = mic_db
        self.stats.clean_db = clean_db
        gain = mic_db - clean_db
        if math.isfinite(gain):
            self.stats.erle_db = ((1.0 - self._erle_alpha) * self.stats.erle_db
                                  + self._erle_alpha * gain)

    def _ensure_pcm(self, chunk: AudioChunk) -> None:
        """确保输入是可直接运算的 PCM 编码。"""
        if chunk.encoding not in (AudioEncoding.PCM16, AudioEncoding.PCM_F32):
            raise AECUnavailable(f"AEC 需要 PCM 输入，收到 {chunk.encoding}")

    def __repr__(self) -> str:
        return (f"<{type(self).__name__} name={self.name} sr={self.sample_rate} "
                f"frame={self.frame_ms}ms erle={self.stats.erle_db:.1f}dB>")


__all__ = [
    "DEFAULT_FRAME_MS",
    "DEFAULT_SAMPLE_RATE",
    "EPS",
    "AECStats",
    "AECUnavailable",
    "BaseAEC",
    "energy_dbfs",
]

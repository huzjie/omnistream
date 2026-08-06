"""VAD 抽象基类与结果模型。

所有 VAD 后端统一实现 :class:`BaseVAD`，输入一帧 :class:`~omnistream.schemas.audio.AudioChunk`，
输出一个 :class:`VADResult`。约定：

* ``process_frame`` 必须是**同步且无 IO 的纯计算**（barge-in 路径不允许挂起）。
  需要重型推理的后端（Silero ONNX）单帧耗时应 < 1ms，否则请自行在上游做 ``to_thread``。
* 后端必须容忍任意帧长；不满足自身要求时应内部缓冲或重采样，绝不抛异常中断链路。
* ``reset()`` 清空内部状态（新一轮会话/新说话人时调用）。
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from omnistream.schemas.audio import AudioChunk

#: 数值下限，避免 log(0)
EPS: Final[float] = 1e-10

#: 全链路默认帧长（毫秒）
DEFAULT_FRAME_MS: Final[int] = 20

#: 全链路默认采样率
DEFAULT_SAMPLE_RATE: Final[int] = 16000


def rms_dbfs(samples: np.ndarray) -> float:
    """计算一段 [-1, 1] 浮点样本的 RMS dBFS 值。

    Args:
        samples: 一维 float32/float64 数组。

    Returns:
        dBFS 值，静音返回 -120.0。
    """
    if samples.size == 0:
        return -120.0
    mean_square = float(np.mean(np.square(samples, dtype=np.float64)))
    if mean_square <= EPS:
        return -120.0
    return float(20.0 * math.log10(math.sqrt(mean_square) + EPS))


def zero_crossing_rate(samples: np.ndarray) -> float:
    """过零率，用于区分清音/摩擦音与稳态噪声。"""
    if samples.size < 2:
        return 0.0
    signs = np.signbit(samples)
    return float(np.count_nonzero(signs[1:] != signs[:-1]) / (samples.size - 1))


def spectral_flatness(samples: np.ndarray) -> float:
    """谱平坦度（Wiener entropy）。接近 1 = 类白噪声；接近 0 = 强谐波（语音）。"""
    if samples.size < 32:
        return 1.0
    windowed = samples * np.hanning(samples.size)
    spectrum = np.abs(np.fft.rfft(windowed)) + EPS
    geo_mean = float(np.exp(np.mean(np.log(spectrum))))
    arith_mean = float(np.mean(spectrum))
    if arith_mean <= EPS:
        return 1.0
    return float(min(1.0, geo_mean / arith_mean))


@dataclass(slots=True, frozen=True)
class VADResult:
    """单帧 VAD 判定结果。"""

    is_speech: bool
    prob: float
    pts_ms: int = 0
    energy_db: float = -120.0
    backend: str = "unknown"
    duration_ms: float = float(DEFAULT_FRAME_MS)
    noise_floor_db: float = -60.0
    details: dict[str, float] = field(default_factory=dict)

    @property
    def snr_db(self) -> float:
        """当前帧相对噪声基底的信噪比。"""
        return self.energy_db - self.noise_floor_db


class BaseVAD(abc.ABC):
    """语音活动检测抽象基类。"""

    #: 注册名，用于 ``create_vad("energy")``
    name: str = "base"

    def __init__(self, *, sample_rate: int = DEFAULT_SAMPLE_RATE, frame_ms: int = DEFAULT_FRAME_MS,
                 threshold: float = 0.5) -> None:
        self.sample_rate = int(sample_rate)
        self.frame_ms = int(frame_ms)
        self.threshold = float(threshold)
        self._frames_seen = 0
        self._speech_frames = 0

    # ------------------------------------------------------------------
    # 子类必须实现
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def process_frame(self, chunk: AudioChunk) -> VADResult:
        """判定单帧是否为语音。实现必须同步、无阻塞 IO。"""

    # ------------------------------------------------------------------
    # 通用能力
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """清空内部状态。子类覆盖时应调用 ``super().reset()``。"""
        self._frames_seen = 0
        self._speech_frames = 0

    def close(self) -> None:
        """释放后端资源（ONNX session 等）。默认无操作，幂等。"""
        return None

    @property
    def speech_ratio(self) -> float:
        """自上次 reset 起的语音帧占比，用于观测 VAD 触发率。"""
        if self._frames_seen == 0:
            return 0.0
        return self._speech_frames / self._frames_seen

    def _account(self, is_speech: bool) -> None:
        self._frames_seen += 1
        if is_speech:
            self._speech_frames += 1

    @staticmethod
    def to_mono_float(chunk: AudioChunk) -> np.ndarray:
        """把任意声道数的 AudioChunk 转成一维 float32 单声道数组。"""
        arr = chunk.to_ndarray()
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        return np.asarray(arr, dtype=np.float32).reshape(-1)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name} sr={self.sample_rate} frame={self.frame_ms}ms>"


class VADUnavailable(RuntimeError):
    """后端依赖缺失时由具体实现的构造函数抛出，供工厂降级捕获。

    注意：这是**工厂内部**的控制流异常，不会外泄到业务层；
    工厂在所有后端均不可用时才会抛 ``omnistream.core.errors.DuplexError``。
    """

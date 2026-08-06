"""音频数学工具：电平、增益、包络、重采样辅助计算。

约定（全平台统一，别在别处再造轮子）：

* **内部浮点表示** 一律是 ``float32``，取值范围 ``[-1.0, 1.0]``；
* **线上传输表示** 一律是 ``pcm16``（小端、有符号 16 位）；
* 所有函数都是**纯函数**，不修改入参（除非函数名里带 ``_inplace``）。

这些函数会被 VAD、AEC、barge-in、mock TTS 高频调用（20ms 一帧、每秒 50 次、
每路会话独立），因此实现上刻意避免了任何 Python 级逐样本循环。
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from omnistream.core.constants import BYTES_PER_PCM16_SAMPLE

__all__ = [
    "EPSILON",
    "PCM16_MAX",
    "apply_gain",
    "bytes_to_duration_ms",
    "clip",
    "crossfade",
    "db_to_linear",
    "dbfs",
    "duration_ms_to_samples",
    "fade_in",
    "fade_out",
    "float32_to_pcm16",
    "frame_signal",
    "linear_to_db",
    "mix",
    "normalize_peak",
    "pcm16_to_float32",
    "peak",
    "rms",
    "samples_to_duration_ms",
    "snr_db",
    "zero_crossing_rate",
]

#: 避免 log(0) 的下限，对应 -200 dBFS，远低于任何真实底噪。
EPSILON: float = 1e-10
#: pcm16 满量程。
PCM16_MAX: int = 32767
_PCM16_SCALE: float = 32768.0


# --------------------------------------------------------------------------- 格式转换
def pcm16_to_float32(data: bytes | bytearray | memoryview | NDArray[np.int16]) -> NDArray[np.float32]:
    """pcm16（小端字节流或 int16 数组）→ ``float32`` ``[-1, 1)``。

    Args:
        data: 原始 PCM16 字节流，或已经解析好的 ``int16`` ndarray。

    Returns:
        与输入等长的 ``float32`` 一维数组。空输入返回空数组。
    """
    if isinstance(data, np.ndarray):
        samples = data.astype(np.int16, copy=False)
    else:
        raw = bytes(data)
        if len(raw) % BYTES_PER_PCM16_SAMPLE:
            raw = raw[: len(raw) - (len(raw) % BYTES_PER_PCM16_SAMPLE)]
        samples = np.frombuffer(raw, dtype="<i2")
    return (samples.astype(np.float32) / _PCM16_SCALE).astype(np.float32, copy=False)


def float32_to_pcm16(samples: NDArray[np.float32]) -> bytes:
    """``float32`` ``[-1, 1]`` → pcm16 小端字节流（超范围先硬限幅）。"""
    if samples.size == 0:
        return b""
    clipped = np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)
    return (clipped * PCM16_MAX).astype("<i2").tobytes()


def samples_to_duration_ms(n_samples: int, sample_rate: int) -> float:
    """采样点数 → 毫秒。"""
    if sample_rate <= 0:
        return 0.0
    return n_samples * 1000.0 / sample_rate


def duration_ms_to_samples(duration_ms: float, sample_rate: int) -> int:
    """毫秒 → 采样点数（四舍五入到最近整数）。"""
    return int(round(duration_ms * sample_rate / 1000.0))


def bytes_to_duration_ms(n_bytes: int, sample_rate: int, channels: int = 1) -> float:
    """pcm16 字节数 → 毫秒。"""
    denom = sample_rate * channels * BYTES_PER_PCM16_SAMPLE
    if denom <= 0:
        return 0.0
    return n_bytes * 1000.0 / denom


# --------------------------------------------------------------------------- 电平
def rms(samples: NDArray[np.float32]) -> float:
    """均方根电平（线性）。空输入返回 0。"""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def peak(samples: NDArray[np.float32]) -> float:
    """峰值绝对电平（线性）。空输入返回 0。"""
    if samples.size == 0:
        return 0.0
    return float(np.max(np.abs(samples)))


def linear_to_db(value: float) -> float:
    """线性幅度 → dB（``20·log10``）。"""
    return 20.0 * math.log10(max(abs(value), EPSILON))


def db_to_linear(db: float) -> float:
    """dB → 线性幅度。"""
    return float(10.0 ** (db / 20.0))


def dbfs(samples: NDArray[np.float32]) -> float:
    """帧的 RMS dBFS。满量程正弦约 -3 dBFS，静音返回 -200。"""
    return linear_to_db(rms(samples))


def snr_db(signal: NDArray[np.float32], noise: NDArray[np.float32]) -> float:
    """信噪比（dB）。噪声为零时返回 ``+200``（视为无穷大但可比较）。"""
    signal_power = float(np.mean(np.square(signal, dtype=np.float64))) if signal.size else 0.0
    noise_power = float(np.mean(np.square(noise, dtype=np.float64))) if noise.size else 0.0
    if noise_power <= EPSILON:
        return 200.0
    return 10.0 * math.log10(max(signal_power, EPSILON) / noise_power)


def zero_crossing_rate(samples: NDArray[np.float32]) -> float:
    """过零率 ``[0, 1]``：清音（擦音）显著高于浊音，是 VAD 的廉价特征。"""
    if samples.size < 2:
        return 0.0
    signs = np.signbit(samples)
    return float(np.count_nonzero(signs[1:] != signs[:-1]) / (samples.size - 1))


# --------------------------------------------------------------------------- 增益与包络
def clip(samples: NDArray[np.float32], limit: float = 1.0) -> NDArray[np.float32]:
    """硬限幅到 ``[-limit, limit]``。"""
    return np.clip(samples, -abs(limit), abs(limit)).astype(np.float32, copy=False)


def apply_gain(samples: NDArray[np.float32], gain_db: float) -> NDArray[np.float32]:
    """施加 dB 增益并硬限幅，避免下游 pcm16 转换时溢出回绕。"""
    if gain_db == 0.0:
        return samples
    return clip(samples * np.float32(db_to_linear(gain_db)))


def normalize_peak(samples: NDArray[np.float32], target_dbfs: float = -1.0) -> NDArray[np.float32]:
    """峰值归一化到 ``target_dbfs``。全静音帧原样返回。"""
    current = peak(samples)
    if current <= EPSILON:
        return samples
    return clip(samples * np.float32(db_to_linear(target_dbfs) / current))


def fade_in(samples: NDArray[np.float32], fade_samples: int) -> NDArray[np.float32]:
    """线性淡入前 ``fade_samples`` 个采样，消除起播 pop 声。"""
    if fade_samples <= 0 or samples.size == 0:
        return samples
    n = min(fade_samples, samples.size)
    out = samples.astype(np.float32, copy=True)
    out[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    return out


def fade_out(samples: NDArray[np.float32], fade_samples: int) -> NDArray[np.float32]:
    """线性淡出末 ``fade_samples`` 个采样。barge-in 打断播放时必须调用，
    否则用户会听到明显的「咔哒」截断噪声。"""
    if fade_samples <= 0 or samples.size == 0:
        return samples
    n = min(fade_samples, samples.size)
    out = samples.astype(np.float32, copy=True)
    out[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return out


def crossfade(a: NDArray[np.float32], b: NDArray[np.float32],
              overlap: int) -> NDArray[np.float32]:
    """把 ``b`` 交叉淡入接到 ``a`` 后面，重叠 ``overlap`` 个采样。

    拼接 TTS 分片、切换说话人音色时用，避免相位跳变导致的爆音。
    """
    if overlap <= 0 or a.size == 0 or b.size == 0:
        return np.concatenate([a, b]).astype(np.float32, copy=False)
    n = min(overlap, a.size, b.size)
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    head = a[:-n]
    blend = a[-n:] * (1.0 - ramp) + b[:n] * ramp
    tail = b[n:]
    return np.concatenate([head, blend, tail]).astype(np.float32, copy=False)


def mix(*tracks: NDArray[np.float32], normalize: bool = True) -> NDArray[np.float32]:
    """混音若干等长/不等长轨道（短的自动零填充）。

    Args:
        tracks: 待混合的浮点音轨。
        normalize: 为 True 时按轨道数衰减，避免叠加溢出。

    Returns:
        混合后的 ``float32`` 数组；无输入时返回空数组。
    """
    usable = [t for t in tracks if t.size > 0]
    if not usable:
        return np.zeros(0, dtype=np.float32)
    length = max(t.size for t in usable)
    acc = np.zeros(length, dtype=np.float32)
    for track in usable:
        acc[: track.size] += track
    if normalize and len(usable) > 1:
        acc /= np.float32(len(usable))
    return clip(acc)


def frame_signal(samples: NDArray[np.float32], frame_len: int,
                 hop: int | None = None) -> NDArray[np.float32]:
    """把一维信号切成 ``(n_frames, frame_len)`` 的二维帧矩阵（末尾零填充）。

    Args:
        samples: 一维输入信号。
        frame_len: 每帧采样点数，必须 > 0。
        hop: 帧移，默认等于 ``frame_len``（不重叠）。

    Returns:
        形状 ``(n_frames, frame_len)`` 的 ``float32`` 数组。
    """
    if frame_len <= 0:
        raise ValueError("frame_len must be positive")
    step = hop if hop and hop > 0 else frame_len
    if samples.size == 0:
        return np.zeros((0, frame_len), dtype=np.float32)
    n_frames = max(1, 1 + math.ceil(max(0, samples.size - frame_len) / step))
    padded_len = (n_frames - 1) * step + frame_len
    padded = np.zeros(padded_len, dtype=np.float32)
    padded[: samples.size] = samples
    indices = np.arange(frame_len)[None, :] + step * np.arange(n_frames)[:, None]
    return padded[indices].astype(np.float32, copy=False)

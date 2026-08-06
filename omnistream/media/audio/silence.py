"""静音生成、静音判定与舒适噪声。

与 VAD 的分工
-------------
本模块只做**纯能量层面**的静音判断，属于媒体层的基础设施：补齐播放队列的空洞、
生成保活帧、裁剪素材首尾。真正的语音活动检测（含模型 VAD）在 WS4 的
``omnistream.duplex.vad`` 中，两者不要混用。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omnistream.media.audio import pcm
from omnistream.media.constants import (
    DEFAULT_CHANNELS,
    DEFAULT_FRAME_MS,
    DEFAULT_INPUT_SAMPLE_RATE,
    SILENCE_THRESHOLD_DBFS,
)
from omnistream.protocol.enums import AudioEncoding
from omnistream.schemas.audio import AudioChunk

__all__ = [
    "SilenceDetector",
    "SilenceStats",
    "comfort_noise",
    "comfort_noise_chunk",
    "find_silence_spans",
    "is_silent",
    "pad_to_samples",
    "silence",
    "silence_bytes",
    "silence_chunk",
    "silence_ratio",
    "trim_trailing_silence",
]

_EPS: float = 1e-12


def silence(num_samples: int, channels: int = DEFAULT_CHANNELS) -> np.ndarray:
    """生成 interleaved 静音样本数组。"""
    return np.zeros(max(0, int(num_samples)) * max(1, int(channels)), dtype=np.float32)


def silence_bytes(
    duration_ms: float,
    *,
    sample_rate: int = DEFAULT_INPUT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    encoding: AudioEncoding = AudioEncoding.PCM16,
) -> bytes:
    """生成指定时长的静音裸字节。"""
    frames = int(round(sample_rate * max(0.0, duration_ms) / 1000.0))
    width = 4 if encoding == AudioEncoding.PCM_F32 else 2
    return b"\x00" * (frames * width * max(1, channels))


def silence_chunk(
    *,
    pts_ms: int,
    duration_ms: float = DEFAULT_FRAME_MS,
    sample_rate: int = DEFAULT_INPUT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    encoding: AudioEncoding = AudioEncoding.PCM16,
    speaker: str | None = None,
) -> AudioChunk:
    """构造一个静音 :class:`AudioChunk`（补洞 / 保活 / 对齐用）。"""
    return AudioChunk(
        pts_ms=int(pts_ms),
        data=silence_bytes(
            duration_ms, sample_rate=sample_rate, channels=channels, encoding=encoding
        ),
        sample_rate=sample_rate,
        channels=max(1, channels),
        encoding=encoding,
        speaker=speaker,
    )


def comfort_noise(
    num_samples: int,
    *,
    level_dbfs: float = -62.0,
    channels: int = DEFAULT_CHANNELS,
    seed: int | None = None,
) -> np.ndarray:
    """生成低电平舒适噪声。

    绝对静音在长时间通话中会让人误以为掉线，插入极低电平的高斯噪声可以显著
    改善主观体验，同时不会触发下游 VAD（默认 -62 dBFS 远低于门限）。

    Args:
        num_samples: 每声道样本数。
        level_dbfs: 噪声 RMS 电平。
        channels: 声道数。
        seed: 随机种子，给定后结果可复现（压测对拍用）。

    Returns:
        interleaved float32 数组。
    """
    total = max(0, int(num_samples)) * max(1, int(channels))
    if total == 0:
        return np.zeros(0, dtype=np.float32)
    rng = np.random.default_rng(seed)
    amplitude = pcm.db_to_linear(level_dbfs)
    noise = rng.standard_normal(total).astype(np.float32) * amplitude
    return np.clip(noise, -1.0, 1.0).astype(np.float32)


def comfort_noise_chunk(
    *,
    pts_ms: int,
    duration_ms: float = DEFAULT_FRAME_MS,
    sample_rate: int = DEFAULT_INPUT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    level_dbfs: float = -62.0,
    seed: int | None = None,
) -> AudioChunk:
    """构造一个舒适噪声 :class:`AudioChunk`。"""
    frames = int(round(sample_rate * max(0.0, duration_ms) / 1000.0))
    samples = comfort_noise(frames, level_dbfs=level_dbfs, channels=channels, seed=seed)
    return pcm.chunk_from_float32(
        samples, pts_ms=int(pts_ms), sample_rate=sample_rate, channels=max(1, channels)
    )


def is_silent(arr: np.ndarray, threshold_dbfs: float = SILENCE_THRESHOLD_DBFS) -> bool:
    """按 RMS 判断一段样本是否为静音。"""
    if arr.size == 0:
        return True
    return pcm.dbfs(arr) < threshold_dbfs


def silence_ratio(
    arr: np.ndarray,
    *,
    sample_rate: int = DEFAULT_INPUT_SAMPLE_RATE,
    window_ms: float = 20.0,
    threshold_dbfs: float = SILENCE_THRESHOLD_DBFS,
) -> float:
    """按窗口统计静音占比，返回 ``0.0 ~ 1.0``。"""
    if arr.size == 0:
        return 1.0
    window = max(1, int(round(sample_rate * window_ms / 1000.0)))
    count = arr.size // window
    if count == 0:
        return 1.0 if is_silent(arr, threshold_dbfs) else 0.0
    view = arr[: count * window].reshape(count, window)
    energy = np.sqrt(np.mean(np.square(view, dtype=np.float64), axis=1))
    limit = pcm.db_to_linear(threshold_dbfs)
    return float(np.count_nonzero(energy < limit)) / float(count)


def find_silence_spans(
    arr: np.ndarray,
    *,
    sample_rate: int = DEFAULT_INPUT_SAMPLE_RATE,
    window_ms: float = 20.0,
    threshold_dbfs: float = SILENCE_THRESHOLD_DBFS,
    min_span_ms: float = 200.0,
) -> list[tuple[float, float]]:
    """找出所有长度超过 ``min_span_ms`` 的静音区间。

    Returns:
        ``[(start_ms, end_ms), ...]``，按时间升序。
    """
    if arr.size == 0:
        return []
    window = max(1, int(round(sample_rate * window_ms / 1000.0)))
    count = arr.size // window
    if count == 0:
        return []
    view = arr[: count * window].reshape(count, window)
    energy = np.sqrt(np.mean(np.square(view, dtype=np.float64), axis=1))
    limit = pcm.db_to_linear(threshold_dbfs)
    quiet = energy < limit

    spans: list[tuple[float, float]] = []
    run_start = -1
    for index in range(count):
        if quiet[index] and run_start < 0:
            run_start = index
        elif not quiet[index] and run_start >= 0:
            spans.append((run_start * window_ms, index * window_ms))
            run_start = -1
    if run_start >= 0:
        spans.append((run_start * window_ms, count * window_ms))
    return [span for span in spans if (span[1] - span[0]) >= min_span_ms]


def pad_to_samples(arr: np.ndarray, target_samples: int) -> np.ndarray:
    """右侧补零到目标长度；已超长则原样返回。"""
    target = max(0, int(target_samples))
    if arr.size >= target:
        return arr
    return np.concatenate((arr, np.zeros(target - arr.size, dtype=np.float32)))


def trim_trailing_silence(
    arr: np.ndarray,
    *,
    threshold_dbfs: float = SILENCE_THRESHOLD_DBFS,
    window: int = 160,
) -> np.ndarray:
    """裁掉尾部静音（保留首部，用于 TTS 输出瘦身）。"""
    if arr.size == 0:
        return arr
    limit = pcm.db_to_linear(threshold_dbfs)
    count = max(1, arr.size // window)
    view = arr[: count * window].reshape(count, window)
    energy = np.sqrt(np.mean(np.square(view, dtype=np.float64), axis=1))
    active = np.flatnonzero(energy > limit)
    if active.size == 0:
        return np.zeros(0, dtype=np.float32)
    return arr[: min(arr.size, (int(active[-1]) + 1) * window)]


@dataclass(slots=True)
class SilenceStats:
    """静音检测器的滚动统计。"""

    total_ms: float = 0.0
    silence_ms: float = 0.0
    active_ms: float = 0.0
    current_run_ms: float = 0.0
    longest_silence_ms: float = 0.0
    transitions: int = 0

    def as_dict(self) -> dict[str, float]:
        """转为可序列化字典。"""
        ratio = self.silence_ms / self.total_ms if self.total_ms > 0 else 0.0
        return {
            "total_ms": round(self.total_ms, 2),
            "silence_ms": round(self.silence_ms, 2),
            "active_ms": round(self.active_ms, 2),
            "current_run_ms": round(self.current_run_ms, 2),
            "longest_silence_ms": round(self.longest_silence_ms, 2),
            "transitions": float(self.transitions),
            "silence_ratio": round(ratio, 4),
        }


class SilenceDetector:
    """带滞回的流式静音检测器。

    与 :class:`~omnistream.media.audio.noise_gate.NoiseGate` 的区别：噪声门会**改写**
    样本，这里只做**判定**，不动数据。上行链路用它来决定"是否值得往 ASR 送"。
    """

    def __init__(
        self,
        *,
        threshold_dbfs: float = SILENCE_THRESHOLD_DBFS,
        hysteresis_db: float = 4.0,
        sample_rate: int = DEFAULT_INPUT_SAMPLE_RATE,
    ) -> None:
        """初始化检测器。

        Args:
            threshold_dbfs: 静音门限。
            hysteresis_db: 滞回宽度，避免门限附近抖动。
            sample_rate: 采样率，用于把样本数换算成毫秒。
        """
        self.threshold_dbfs = float(threshold_dbfs)
        self.hysteresis_db = max(0.0, float(hysteresis_db))
        self.sample_rate = max(1, int(sample_rate))
        self._silent = True
        self.stats = SilenceStats()

    @property
    def silent(self) -> bool:
        """当前是否处于静音态。"""
        return self._silent

    @property
    def silence_ms(self) -> float:
        """当前连续静音时长（毫秒）；非静音态返回 0。"""
        return self.stats.current_run_ms if self._silent else 0.0

    def observe(self, arr: np.ndarray, *, channels: int = 1) -> bool:
        """吸收一段样本，返回该段是否被判为静音。"""
        if arr.size == 0:
            return self._silent
        frames = arr.size / max(1, channels)
        duration_ms = frames * 1000.0 / self.sample_rate
        level = pcm.dbfs(arr)

        # 滞回：静音态需要超过 threshold + hysteresis 才认为开始说话
        if self._silent:
            now_silent = level < (self.threshold_dbfs + self.hysteresis_db)
        else:
            now_silent = level < self.threshold_dbfs

        if now_silent != self._silent:
            self.stats.transitions += 1
            self.stats.current_run_ms = 0.0
            self._silent = now_silent

        self.stats.total_ms += duration_ms
        self.stats.current_run_ms += duration_ms
        if now_silent:
            self.stats.silence_ms += duration_ms
            self.stats.longest_silence_ms = max(
                self.stats.longest_silence_ms, self.stats.current_run_ms
            )
        else:
            self.stats.active_ms += duration_ms
        return now_silent

    def observe_chunk(self, chunk: AudioChunk) -> bool:
        """吸收一个 :class:`AudioChunk`。"""
        self.sample_rate = chunk.sample_rate
        return self.observe(pcm.to_float32(chunk), channels=chunk.channels)

    def reset(self) -> None:
        """重置状态与统计。"""
        self._silent = True
        self.stats = SilenceStats()

    def snapshot(self) -> dict[str, float]:
        """导出统计快照。"""
        data = self.stats.as_dict()
        data["silent"] = 1.0 if self._silent else 0.0
        data["threshold_dbfs"] = self.threshold_dbfs
        return data

    def __repr__(self) -> str:
        """调试用文本表示。"""
        state = "silent" if self._silent else "active"
        return f"SilenceDetector({state}, run={self.stats.current_run_ms:.0f}ms)"

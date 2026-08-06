"""噪声门 + 谱减法降噪（可选启用）。

两级处理，都是纯 numpy 实现：

**1. 噪声门（NoiseGate）**
    带滞回的电平门：打开阈值 ``-45 dBFS``、关闭阈值 ``-52 dBFS``（滞回 7dB 防抖），
    带 hold 时间与 attack/release 斜坡，关闭时输出按斜坡渐隐而不是硬切（防咔哒）。

**2. 谱减法（SpectralDenoiser）**
    * 用最小值统计法（minimum statistics）在滑动窗口内估计噪声功率谱
    * 过减因子 ``alpha`` + 频谱下限 ``beta`` 防止音乐噪声
    * 50% 重叠 Hann 窗 STFT / ISTFT，保证 overlap-add 完美重构

默认只启用噪声门；谱减法需显式 ``enable_denoise=True``（会引入 ~16ms 额外延迟）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omnistream.media.audio import pcm
from omnistream.media.constants import (
    NOISE_GATE_ATTACK_MS,
    NOISE_GATE_CLOSE_DBFS,
    NOISE_GATE_HOLD_MS,
    NOISE_GATE_OPEN_DBFS,
    NOISE_GATE_RELEASE_MS,
)
from omnistream.protocol.enums import AudioEncoding
from omnistream.schemas.audio import AudioChunk

__all__ = ["NoiseGate", "NoiseGateStats", "NoiseSuppressor", "SpectralDenoiser"]


@dataclass(slots=True)
class NoiseGateStats:
    """噪声门统计。"""

    frames: int = 0
    open_frames: int = 0
    closed_frames: int = 0
    noise_floor_dbfs: float = -60.0

    def as_dict(self) -> dict[str, float]:
        """转成扁平字典。"""
        return {
            "frames": float(self.frames),
            "open_frames": float(self.open_frames),
            "closed_frames": float(self.closed_frames),
            "open_ratio": round(self.open_frames / max(1, self.frames), 4),
            "noise_floor_dbfs": round(self.noise_floor_dbfs, 2),
        }


class NoiseGate:
    """带滞回与斜坡的电平噪声门。"""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        open_dbfs: float = NOISE_GATE_OPEN_DBFS,
        close_dbfs: float = NOISE_GATE_CLOSE_DBFS,
        hold_ms: float = NOISE_GATE_HOLD_MS,
        attack_ms: float = NOISE_GATE_ATTACK_MS,
        release_ms: float = NOISE_GATE_RELEASE_MS,
        floor_gain: float = 0.06,
        adaptive: bool = True,
    ) -> None:
        """
        Args:
            sample_rate: 采样率。
            open_dbfs: 门开启阈值。
            close_dbfs: 门关闭阈值（必须低于 open，形成滞回）。
            hold_ms: 电平跌破关闭阈值后仍保持开启的时长。
            attack_ms: 开门斜坡时长。
            release_ms: 关门斜坡时长。
            floor_gain: 关门后的残留增益（不完全静音，保留房间感）。
            adaptive: 是否根据长期最小电平自适应调整阈值。
        """
        self.sample_rate = sample_rate
        self.open_dbfs = open_dbfs
        self.close_dbfs = min(close_dbfs, open_dbfs - 2.0)
        self.hold_ms = hold_ms
        self.attack_ms = max(0.5, attack_ms)
        self.release_ms = max(0.5, release_ms)
        self.floor_gain = float(np.clip(floor_gain, 0.0, 1.0))
        self.adaptive = adaptive
        self._is_open = False
        self._hold_left_ms = 0.0
        self._gain = self.floor_gain
        self._noise_floor_dbfs = -60.0
        self.stats = NoiseGateStats()

    @property
    def is_open(self) -> bool:
        """门当前是否打开。"""
        return self._is_open

    def process(self, arr: np.ndarray) -> np.ndarray:
        """处理一帧样本。"""
        if arr.size == 0:
            return arr
        self.stats.frames += 1
        frame_ms = arr.size * 1000.0 / max(1, self.sample_rate)
        level = pcm.dbfs(arr)

        if self.adaptive:
            # 只在电平低于当前底噪 + 6dB 时更新底噪估计（跟踪最小统计量）
            if level < self._noise_floor_dbfs + 6.0:
                self._noise_floor_dbfs = 0.98 * self._noise_floor_dbfs + 0.02 * level
            else:
                self._noise_floor_dbfs += 0.02  # 缓慢上抬，避免永久锁死在极低值
            self._noise_floor_dbfs = float(np.clip(self._noise_floor_dbfs, -90.0, -20.0))
            self.stats.noise_floor_dbfs = self._noise_floor_dbfs
            open_th = max(self.open_dbfs, self._noise_floor_dbfs + 9.0)
            close_th = max(self.close_dbfs, self._noise_floor_dbfs + 4.0)
        else:
            open_th, close_th = self.open_dbfs, self.close_dbfs

        if level >= open_th:
            self._is_open = True
            self._hold_left_ms = self.hold_ms
        elif level < close_th:
            self._hold_left_ms -= frame_ms
            if self._hold_left_ms <= 0.0:
                self._is_open = False
        else:
            self._hold_left_ms -= frame_ms
            if self._hold_left_ms <= 0.0:
                self._is_open = False

        if self._is_open:
            self.stats.open_frames += 1
            target = 1.0
            tau_ms = self.attack_ms
        else:
            self.stats.closed_frames += 1
            target = self.floor_gain
            tau_ms = self.release_ms

        steps = max(1, int(self.sample_rate * tau_ms / 1000.0))
        span = min(arr.size, steps)
        ramp = np.full(arr.size, target, dtype=np.float32)
        ramp[:span] = np.linspace(self._gain, target, span, dtype=np.float32)
        self._gain = target
        return (arr * ramp).astype(np.float32)

    def process_chunk(self, chunk: AudioChunk) -> AudioChunk:
        """处理一个 chunk。"""
        if chunk.encoding not in (AudioEncoding.PCM16, AudioEncoding.PCM_F32):
            return chunk
        samples = pcm.bytes_to_float32(chunk.data, chunk.encoding)
        return AudioChunk(
            pts_ms=chunk.pts_ms,
            data=pcm.float32_to_bytes(self.process(samples), chunk.encoding),
            sample_rate=chunk.sample_rate,
            channels=chunk.channels,
            encoding=chunk.encoding,
            capture_ts_ns=chunk.capture_ts_ns,
            speaker=chunk.speaker,
        )

    def reset(self) -> None:
        """复位门状态。"""
        self._is_open = False
        self._hold_left_ms = 0.0
        self._gain = self.floor_gain
        self._noise_floor_dbfs = -60.0

    def snapshot(self) -> dict[str, float]:
        """统计快照。"""
        return self.stats.as_dict()


class SpectralDenoiser:
    """基于最小值统计的谱减法降噪。

    50% 重叠 Hann 窗 STFT，逐帧执行::

        |Y| = max(|X| - alpha * |N|, beta * |X|)
        y   = ISTFT(|Y| * exp(j * phase(X)))

    噪声谱 ``|N|`` 用滑动最小值 + 上偏系数估计，无需人工标注静音段。
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        frame_size: int = 512,
        alpha: float = 2.0,
        beta: float = 0.08,
        noise_window: int = 40,
    ) -> None:
        """
        Args:
            sample_rate: 采样率。
            frame_size: FFT 窗长（样本），建议 2 的幂。
            alpha: 过减因子，越大降噪越狠但越容易产生音乐噪声。
            beta: 频谱下限系数，保底残留避免过减。
            noise_window: 最小值统计的窗口帧数。
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop = frame_size // 2
        self.alpha = alpha
        self.beta = beta
        self.noise_window = max(4, noise_window)
        self._window = np.hanning(frame_size + 1)[:frame_size].astype(np.float32)
        self._input_tail = np.zeros(0, dtype=np.float32)
        self._overlap = np.zeros(self.hop, dtype=np.float32)
        self._history: list[np.ndarray] = []
        self._noise_mag: np.ndarray | None = None

    def _update_noise(self, magnitude: np.ndarray) -> np.ndarray:
        self._history.append(magnitude)
        if len(self._history) > self.noise_window:
            self._history.pop(0)
        stacked = np.stack(self._history, axis=0)
        minimum = np.min(stacked, axis=0)
        # 最小值统计存在系统性低估，乘以经验补偿因子 1.5
        estimate = minimum * 1.5
        if self._noise_mag is None:
            self._noise_mag = estimate
        else:
            self._noise_mag = 0.9 * self._noise_mag + 0.1 * estimate
        return self._noise_mag

    def process(self, arr: np.ndarray) -> np.ndarray:
        """流式处理，输出长度与输入基本一致（首帧有 ``hop`` 样本延迟）。"""
        if arr.size == 0:
            return arr
        buffer = np.concatenate((self._input_tail, arr.astype(np.float32)))
        out_parts: list[np.ndarray] = []
        pos = 0
        while pos + self.frame_size <= buffer.size:
            segment = buffer[pos : pos + self.frame_size] * self._window
            spectrum = np.fft.rfft(segment)
            magnitude = np.abs(spectrum)
            noise = self._update_noise(magnitude)
            cleaned = np.maximum(magnitude - self.alpha * noise, self.beta * magnitude)
            restored = np.fft.irfft(
                cleaned * np.exp(1j * np.angle(spectrum)), n=self.frame_size
            ).astype(np.float32)
            restored *= self._window
            head = restored[: self.hop] + self._overlap
            self._overlap = restored[self.hop :]
            out_parts.append(head)
            pos += self.hop
        self._input_tail = buffer[pos:]
        if not out_parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(out_parts).astype(np.float32)

    def reset(self) -> None:
        """清空 STFT 状态与噪声估计。"""
        self._input_tail = np.zeros(0, dtype=np.float32)
        self._overlap = np.zeros(self.hop, dtype=np.float32)
        self._history.clear()
        self._noise_mag = None


class NoiseSuppressor:
    """噪声门 + 可选谱减法的组合入口（管线直接用这个）。"""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        enable_gate: bool = True,
        enable_denoise: bool = False,
        gate: NoiseGate | None = None,
        denoiser: SpectralDenoiser | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.enable_gate = enable_gate
        self.enable_denoise = enable_denoise
        self.gate = gate or NoiseGate(sample_rate=sample_rate)
        self.denoiser = denoiser or SpectralDenoiser(sample_rate=sample_rate)

    def process_chunk(self, chunk: AudioChunk) -> AudioChunk:
        """按配置依次执行降噪与门控。"""
        if chunk.encoding not in (AudioEncoding.PCM16, AudioEncoding.PCM_F32):
            return chunk
        samples = pcm.bytes_to_float32(chunk.data, chunk.encoding)
        if self.enable_denoise:
            denoised = self.denoiser.process(samples)
            # 谱减法有 hop 延迟，长度不足时补零对齐，保持帧长恒定
            if denoised.size < samples.size:
                denoised = np.concatenate(
                    (denoised, np.zeros(samples.size - denoised.size, dtype=np.float32))
                )
            samples = denoised[: samples.size]
        if self.enable_gate:
            samples = self.gate.process(samples)
        return AudioChunk(
            pts_ms=chunk.pts_ms,
            data=pcm.float32_to_bytes(samples, chunk.encoding),
            sample_rate=chunk.sample_rate,
            channels=chunk.channels,
            encoding=chunk.encoding,
            capture_ts_ns=chunk.capture_ts_ns,
            speaker=chunk.speaker,
        )

    def reset(self) -> None:
        """复位全部内部状态。"""
        self.gate.reset()
        self.denoiser.reset()

    def snapshot(self) -> dict[str, float]:
        """统计快照。"""
        return self.gate.snapshot()

"""自动增益控制（AGC）与峰值限幅。

目标：把用户麦克风电平稳定在 ASR 友好的 ``-18 dBFS`` 附近，
既避免过小（识别率骤降）也避免过载削波（产生谐波失真）。

算法
----
1. **电平检测**：每帧计算 RMS → dBFS。
2. **平滑**：用不同的 attack / release 时间常数做单极点平滑，
   电平上升快（attack 12ms，防爆音）、下降慢（release 220ms，防抽气）。
3. **增益计算**：``gain_db = target_dbfs - smoothed_dbfs``，
   裁剪到 ``[AGC_MIN_GAIN_DB, AGC_MAX_GAIN_DB]``。
4. **增益平滑**：对增益本身再做一次单极点平滑，帧内用线性插值渐变，消除阶跃咔哒。
5. **限幅**：软拐点限幅器把峰值压到 ``-1 dBFS`` 以下（tanh 软饱和，无硬削波）。
6. **门控**：低于噪声门限（默认 -55 dBFS）时不提升增益，避免把底噪放大。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omnistream.media.audio import pcm
from omnistream.media.constants import (
    AGC_ATTACK_MS,
    AGC_MAX_GAIN_DB,
    AGC_MIN_GAIN_DB,
    AGC_RELEASE_MS,
    AGC_TARGET_DBFS,
    LIMITER_CEILING_DBFS,
)
from omnistream.protocol.enums import AudioEncoding
from omnistream.schemas.audio import AudioChunk

__all__ = ["AutoGainControl", "GainStats", "Limiter", "apply_gain_db", "soft_limit"]


def apply_gain_db(arr: np.ndarray, gain_db: float) -> np.ndarray:
    """对样本施加固定分贝增益。"""
    return (arr * pcm.db_to_linear(gain_db)).astype(np.float32)


def soft_limit(arr: np.ndarray, ceiling_dbfs: float = LIMITER_CEILING_DBFS) -> np.ndarray:
    """软拐点限幅：拐点以下线性，以上 tanh 压缩，输出严格不超过 ceiling。

    Args:
        arr: 输入样本。
        ceiling_dbfs: 输出上限（dBFS）。

    Returns:
        限幅后的样本。
    """
    ceiling = pcm.db_to_linear(ceiling_dbfs)
    knee = ceiling * 0.7
    out = arr.astype(np.float32, copy=True)
    magnitude = np.abs(out)
    over = magnitude > knee
    if np.any(over):
        excess = (magnitude[over] - knee) / max(ceiling - knee, 1e-6)
        compressed = knee + (ceiling - knee) * np.tanh(excess)
        out[over] = np.sign(out[over]) * compressed
    return np.clip(out, -ceiling, ceiling).astype(np.float32)


@dataclass(slots=True)
class GainStats:
    """AGC 运行统计。"""

    frames: int = 0
    current_gain_db: float = 0.0
    input_dbfs: float = -120.0
    output_dbfs: float = -120.0
    limited_frames: int = 0
    gated_frames: int = 0

    def as_dict(self) -> dict[str, float]:
        """转成扁平字典。"""
        return {
            "frames": float(self.frames),
            "gain_db": round(self.current_gain_db, 2),
            "input_dbfs": round(self.input_dbfs, 2),
            "output_dbfs": round(self.output_dbfs, 2),
            "limited_frames": float(self.limited_frames),
            "gated_frames": float(self.gated_frames),
        }


class Limiter:
    """独立的峰值限幅器（也可脱离 AGC 单独使用）。"""

    def __init__(self, ceiling_dbfs: float = LIMITER_CEILING_DBFS) -> None:
        self.ceiling_dbfs = ceiling_dbfs
        self.engaged = 0

    def process(self, arr: np.ndarray) -> np.ndarray:
        """限幅并统计触发次数。"""
        ceiling = pcm.db_to_linear(self.ceiling_dbfs)
        if arr.size and float(np.max(np.abs(arr))) > ceiling:
            self.engaged += 1
        return soft_limit(arr, self.ceiling_dbfs)


class AutoGainControl:
    """流式自动增益控制。

    Example::

        agc = AutoGainControl(sample_rate=16000)
        chunk = agc.process_chunk(chunk)
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        target_dbfs: float = AGC_TARGET_DBFS,
        max_gain_db: float = AGC_MAX_GAIN_DB,
        min_gain_db: float = AGC_MIN_GAIN_DB,
        attack_ms: float = AGC_ATTACK_MS,
        release_ms: float = AGC_RELEASE_MS,
        noise_floor_dbfs: float = -55.0,
        ceiling_dbfs: float = LIMITER_CEILING_DBFS,
        enabled: bool = True,
    ) -> None:
        """
        Args:
            sample_rate: 采样率，用于把时间常数换算成单极点系数。
            target_dbfs: 目标 RMS 电平。
            max_gain_db: 最大提升。
            min_gain_db: 最大衰减（负值）。
            attack_ms: 电平上升平滑时间常数。
            release_ms: 电平下降平滑时间常数。
            noise_floor_dbfs: 低于此电平判定为底噪，冻结增益。
            ceiling_dbfs: 限幅上限。
            enabled: 关闭时 process 为纯直通（仍统计电平）。
        """
        self.sample_rate = sample_rate
        self.target_dbfs = target_dbfs
        self.max_gain_db = max_gain_db
        self.min_gain_db = min_gain_db
        self.noise_floor_dbfs = noise_floor_dbfs
        self.enabled = enabled
        self._attack_ms = max(1.0, attack_ms)
        self._release_ms = max(1.0, release_ms)
        self._env_dbfs = -60.0
        self._gain_db = 0.0
        self._prev_gain_linear = 1.0
        self._limiter = Limiter(ceiling_dbfs)
        self.stats = GainStats()

    def _coeff(self, tau_ms: float, num_samples: int) -> float:
        """把时间常数换算成单极点系数（按本帧样本数计算）。"""
        frame_ms = num_samples * 1000.0 / max(1, self.sample_rate)
        return float(np.exp(-frame_ms / tau_ms))

    def process(self, arr: np.ndarray) -> np.ndarray:
        """处理一帧 ``float32`` 样本。"""
        if arr.size == 0:
            return arr
        self.stats.frames += 1
        level = pcm.dbfs(arr)
        self.stats.input_dbfs = level

        coeff = self._coeff(
            self._attack_ms if level > self._env_dbfs else self._release_ms, arr.size
        )
        self._env_dbfs = coeff * self._env_dbfs + (1.0 - coeff) * level

        if not self.enabled:
            self.stats.output_dbfs = level
            return arr

        if self._env_dbfs < self.noise_floor_dbfs:
            # 底噪：保持当前增益，不放大噪声
            self.stats.gated_frames += 1
            desired_gain_db = self._gain_db
        else:
            desired_gain_db = self.target_dbfs - self._env_dbfs
            desired_gain_db = float(
                np.clip(desired_gain_db, self.min_gain_db, self.max_gain_db)
            )

        gain_coeff = self._coeff(self._release_ms * 0.5, arr.size)
        self._gain_db = gain_coeff * self._gain_db + (1.0 - gain_coeff) * desired_gain_db
        self.stats.current_gain_db = self._gain_db

        target_linear = pcm.db_to_linear(self._gain_db)
        ramp = np.linspace(self._prev_gain_linear, target_linear, arr.size, dtype=np.float32)
        self._prev_gain_linear = target_linear

        boosted = (arr * ramp).astype(np.float32)
        before = self._limiter.engaged
        out = self._limiter.process(boosted)
        if self._limiter.engaged != before:
            self.stats.limited_frames += 1
        self.stats.output_dbfs = pcm.dbfs(out)
        return out

    def process_chunk(self, chunk: AudioChunk) -> AudioChunk:
        """处理一个 :class:`AudioChunk`（仅支持裸 PCM）。"""
        if chunk.encoding not in (AudioEncoding.PCM16, AudioEncoding.PCM_F32):
            return chunk
        samples = pcm.bytes_to_float32(chunk.data, chunk.encoding)
        processed = self.process(samples)
        return AudioChunk(
            pts_ms=chunk.pts_ms,
            data=pcm.float32_to_bytes(processed, chunk.encoding),
            sample_rate=chunk.sample_rate,
            channels=chunk.channels,
            encoding=chunk.encoding,
            capture_ts_ns=chunk.capture_ts_ns,
            speaker=chunk.speaker,
        )

    def reset(self) -> None:
        """重置内部包络与增益状态。"""
        self._env_dbfs = -60.0
        self._gain_db = 0.0
        self._prev_gain_linear = 1.0

    def snapshot(self) -> dict[str, float]:
        """返回统计快照。"""
        return self.stats.as_dict()

    def __repr__(self) -> str:
        return f"<AutoGainControl gain={self._gain_db:.1f}dB env={self._env_dbfs:.1f}dBFS>"

"""纯能量阈值 VAD —— 零第三方依赖（仅 numpy）的兜底实现。

算法（真实可用，非玩具）::

    1. 帧能量 RMS → dBFS
    2. 自适应噪声基底：快降慢升的非对称 EMA，跟踪环境底噪
    3. 动态阈值 = max(绝对阈值, 噪声基底 + SNR 余量)
    4. 迟滞（hysteresis）：起始阈值高、维持阈值低 6dB，避免边界抖动
    5. 谱平坦度 + 过零率联合校验：拒绝风扇/白噪声一类的稳态宽带噪声
    6. 概率输出：sigmoid((energy_db - 阈值) / 斜率)，供 Ensemble 加权投票

在 ``mock_mode`` 与"无模型权重"环境下这是默认后端，能保证全双工链路端到端跑通。
"""
from __future__ import annotations

import math

import numpy as np

from omnistream.duplex.vad.base import (
    DEFAULT_FRAME_MS,
    DEFAULT_SAMPLE_RATE,
    BaseVAD,
    VADResult,
    rms_dbfs,
    spectral_flatness,
    zero_crossing_rate,
)
from omnistream.schemas.audio import AudioChunk


class EnergyVAD(BaseVAD):
    """自适应能量阈值 VAD。"""

    name = "energy"

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_ms: int = DEFAULT_FRAME_MS,
        threshold: float = 0.5,
        absolute_threshold_db: float = -45.0,
        snr_margin_db: float = 9.0,
        release_margin_db: float = 6.0,
        noise_attack: float = 0.35,
        noise_release: float = 0.008,
        min_noise_db: float = -90.0,
        max_noise_db: float = -25.0,
        flatness_max: float = 0.62,
        slope_db: float = 6.0,
    ) -> None:
        super().__init__(sample_rate=sample_rate, frame_ms=frame_ms, threshold=threshold)
        self.absolute_threshold_db = float(absolute_threshold_db)
        self.snr_margin_db = float(snr_margin_db)
        self.release_margin_db = float(release_margin_db)
        self.noise_attack = float(noise_attack)
        self.noise_release = float(noise_release)
        self.min_noise_db = float(min_noise_db)
        self.max_noise_db = float(max_noise_db)
        self.flatness_max = float(flatness_max)
        self.slope_db = max(1.0, float(slope_db))

        self._noise_db: float = -60.0
        self._in_speech: bool = False
        self._warmup_frames: int = 0

    # ------------------------------------------------------------------

    def reset(self) -> None:
        super().reset()
        self._noise_db = -60.0
        self._in_speech = False
        self._warmup_frames = 0

    def process_frame(self, chunk: AudioChunk) -> VADResult:
        samples = self.to_mono_float(chunk)
        energy_db = rms_dbfs(samples)
        flatness = spectral_flatness(samples)
        zcr = zero_crossing_rate(samples)

        self._update_noise_floor(energy_db)
        enter_db = max(self.absolute_threshold_db, self._noise_db + self.snr_margin_db)
        exit_db = enter_db - self.release_margin_db
        active_db = exit_db if self._in_speech else enter_db

        # 稳态宽带噪声抑制：谱非常平坦 + 过零率极高 → 判为噪声
        noise_like = flatness > self.flatness_max and zcr > 0.45
        loud_enough = energy_db >= active_db

        is_speech = bool(loud_enough and not noise_like)
        # 预热期（前 5 帧）先建立噪声基底，不吐 speech，避免开麦瞬间误触发
        if self._warmup_frames < 5:
            self._warmup_frames += 1
            is_speech = False

        self._in_speech = is_speech
        prob = self._probability(energy_db, enter_db, noise_like)
        self._account(is_speech)

        return VADResult(
            is_speech=is_speech,
            prob=prob,
            pts_ms=int(chunk.pts_ms),
            energy_db=energy_db,
            backend=self.name,
            duration_ms=float(chunk.duration_ms) or float(self.frame_ms),
            noise_floor_db=self._noise_db,
            details={"flatness": flatness, "zcr": zcr, "enter_db": enter_db, "exit_db": exit_db},
        )

    # ------------------------------------------------------------------

    def _update_noise_floor(self, energy_db: float) -> None:
        """非对称 EMA：底噪下降跟得快、上升跟得慢，避免被语音抬高。"""
        if energy_db < self._noise_db:
            alpha = self.noise_attack
        else:
            alpha = self.noise_release
        self._noise_db += alpha * (energy_db - self._noise_db)
        self._noise_db = min(self.max_noise_db, max(self.min_noise_db, self._noise_db))

    def _probability(self, energy_db: float, enter_db: float, noise_like: bool) -> float:
        margin = (energy_db - enter_db) / self.slope_db
        prob = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, margin))))
        if noise_like:
            prob *= 0.35
        return float(min(1.0, max(0.0, prob)))

    @property
    def noise_floor_db(self) -> float:
        return self._noise_db


def peak_dbfs(samples: np.ndarray) -> float:
    """峰值 dBFS，barge-in 检测器用它做快速能量闸门。"""
    if samples.size == 0:
        return -120.0
    peak = float(np.max(np.abs(samples)))
    if peak <= 1e-10:
        return -120.0
    return float(20.0 * math.log10(peak))

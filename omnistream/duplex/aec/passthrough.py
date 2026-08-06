"""直通 AEC（默认实现）。

**什么时候用**：浏览器 / 移动端 SDK 已经在采集侧开启了 WebRTC AEC
（``getUserMedia({audio: {echoCancellation: true}})``），服务端再做一次
只会引入额外延迟和音质损伤。OmniStream 的默认链路属于这一类。

直通并不等于什么都不做——它仍然：

* 维护参考信号缓冲，使得"是否有回声风险"这件事可被观测；
* 统计麦克风/参考的能量比，估算**回声泄漏风险分**，
  barge-in 检测器可以据此临时抬高能量门槛（自适应防自激）；
* 可选开启 ``residual_suppression``：一个非常轻量的频域残余抑制器，
  只在"助手正在播放且麦克风能量与参考高度相关"时做温和衰减，
  作为浏览器 AEC 偶发失效时的安全网。默认关闭。
"""
from __future__ import annotations

import numpy as np
import structlog

from omnistream.duplex.aec.base import (
    DEFAULT_FRAME_MS,
    DEFAULT_SAMPLE_RATE,
    EPS,
    BaseAEC,
    energy_dbfs,
)
from omnistream.duplex.aec.reference import ReferenceSignal
from omnistream.schemas.audio import AudioChunk

log = structlog.get_logger(__name__)


class PassthroughAEC(BaseAEC):
    """不改变信号的 AEC，附带回声风险观测与可选残余抑制。"""

    name = "passthrough"

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_ms: int = DEFAULT_FRAME_MS,
        channels: int = 1,
        track_reference: bool = True,
        residual_suppression: bool = False,
        suppression_db: float = 9.0,
        risk_threshold: float = 0.55,
    ) -> None:
        super().__init__(sample_rate=sample_rate, frame_ms=frame_ms, channels=channels)
        self.track_reference = bool(track_reference)
        self.residual_suppression = bool(residual_suppression)
        self.suppression_db = float(suppression_db)
        self.risk_threshold = float(risk_threshold)

        self.reference = ReferenceSignal(sample_rate=self.sample_rate) if track_reference else None
        self._risk = 0.0
        self._frames_suppressed = 0

    # ------------------------------------------------------------------
    # 参考信号
    # ------------------------------------------------------------------

    def push_reference(self, chunk: AudioChunk) -> None:
        """把播放帧写入参考缓冲。"""
        super().push_reference(chunk)
        if self.reference is not None:
            self.reference.push(chunk)

    # ------------------------------------------------------------------
    # 主处理
    # ------------------------------------------------------------------

    def process(self, mic: AudioChunk, ref: AudioChunk | None = None) -> AudioChunk:
        """直通返回麦克风帧（可选残余抑制）。"""
        self._ensure_pcm(mic)
        mic_samples = self.to_mono_float(mic)

        ref_samples = self._resolve_reference(mic_samples.size, ref)
        self._risk = self._echo_risk(mic_samples, ref_samples)

        if not self.residual_suppression or self._risk < self.risk_threshold:
            self._account(mic_samples, mic_samples)
            return mic

        gain = float(10.0 ** (-abs(self.suppression_db) / 20.0))
        # 风险越高衰减越接近 suppression_db，避免"一刀切"损伤双讲。
        blend = min(1.0, (self._risk - self.risk_threshold) / max(EPS, 1.0 - self.risk_threshold))
        applied = 1.0 - blend * (1.0 - gain)
        cleaned = (mic_samples * applied).astype(np.float32)
        self._frames_suppressed += 1
        self._account(mic_samples, cleaned)
        return self._rebuild(mic, cleaned)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _resolve_reference(self, n: int, ref: AudioChunk | None) -> np.ndarray:
        if ref is not None:
            return self.to_mono_float(ref)
        if self.reference is None:
            return np.zeros(0, dtype=np.float32)
        samples = self.reference.take(n)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak <= 1e-6:
            self.stats.reference_underruns += 1
        return samples

    def _echo_risk(self, mic: np.ndarray, ref: np.ndarray) -> float:
        """回声泄漏风险 [0,1]：参考有能量、且与麦克风相关性高时接近 1。"""
        if mic.size == 0 or ref.size == 0:
            return 0.0
        n = min(mic.size, ref.size)
        m = mic[:n].astype(np.float64)
        r = ref[:n].astype(np.float64)
        ref_db = energy_dbfs(ref[:n])
        if ref_db < -55.0:
            return 0.0
        m_norm = float(np.linalg.norm(m))
        r_norm = float(np.linalg.norm(r))
        if m_norm < 1e-6 or r_norm < 1e-6:
            return 0.0
        corr = abs(float(np.dot(m, r)) / (m_norm * r_norm))
        # 能量项：参考越响，泄漏概率越大
        loudness = min(1.0, max(0.0, (ref_db + 55.0) / 40.0))
        return float(min(1.0, 0.65 * corr + 0.35 * loudness))

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------

    @property
    def echo_risk(self) -> float:
        """最近一帧的回声泄漏风险，barge-in 可用它临时抬高门槛。"""
        return self._risk

    def suggested_barge_in_margin_db(self) -> float:
        """基于风险给出的 barge-in 能量门槛补偿（dB，>=0）。"""
        if self._risk < self.risk_threshold:
            return 0.0
        return float(min(12.0, 12.0 * (self._risk - self.risk_threshold)
                         / max(EPS, 1.0 - self.risk_threshold)))

    def reset(self) -> None:
        super().reset()
        self._risk = 0.0
        self._frames_suppressed = 0
        if self.reference is not None:
            self.reference.reset()

    def stats_dict(self) -> dict[str, float]:
        data = self.stats.as_dict()
        data["echo_risk"] = round(self._risk, 3)
        data["frames_suppressed"] = float(self._frames_suppressed)
        if self.reference is not None:
            data.update({f"ref_{k}": v for k, v in self.reference.stats().items()})
        return data


__all__ = ["PassthroughAEC"]

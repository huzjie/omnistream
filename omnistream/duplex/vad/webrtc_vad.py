"""webrtcvad 后端（GMM 判决，CPU 占用极低）。

限制（由 libwebrtc 决定，本实现自动适配，调用方无需关心）：

* 采样率只支持 8000 / 16000 / 32000 / 48000
* 帧长只支持 10 / 20 / 30 ms
* 输入必须是单声道 int16 小端

不满足时本实现会做**线性重采样 + 内部缓冲重新分帧**，而不是抛异常。
依赖 ``webrtcvad`` 或 ``webrtcvad-wheels``；未安装时构造函数抛 :class:`VADUnavailable`，
由 :func:`omnistream.duplex.vad.create_vad` 自动降级到能量 VAD。
"""
from __future__ import annotations

from typing import Any, Final

import numpy as np

from omnistream.duplex.vad.base import (
    DEFAULT_FRAME_MS,
    DEFAULT_SAMPLE_RATE,
    BaseVAD,
    VADResult,
    VADUnavailable,
    rms_dbfs,
)
from omnistream.schemas.audio import AudioChunk

_SUPPORTED_RATES: Final[tuple[int, ...]] = (8000, 16000, 32000, 48000)
_SUPPORTED_FRAME_MS: Final[tuple[int, ...]] = (10, 20, 30)

_webrtcvad: Any = None
_IMPORT_ERROR: str = ""

try:  # pragma: no cover - 依赖可选
    import webrtcvad as _webrtcvad_mod

    _webrtcvad = _webrtcvad_mod
except Exception as exc:  # noqa: BLE001 - 任何导入失败都视为不可用
    _IMPORT_ERROR = str(exc)


def is_available() -> bool:
    """webrtcvad 是否可用。"""
    return _webrtcvad is not None


class WebRTCVAD(BaseVAD):
    """Google WebRTC VAD 封装。"""

    name = "webrtc"

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_ms: int = DEFAULT_FRAME_MS,
        threshold: float = 0.5,
        mode: int = 2,
    ) -> None:
        if _webrtcvad is None:
            raise VADUnavailable(f"webrtcvad 未安装（pip install webrtcvad-wheels）: {_IMPORT_ERROR}")
        super().__init__(sample_rate=sample_rate, frame_ms=frame_ms, threshold=threshold)
        self.mode = max(0, min(3, int(mode)))
        self._target_rate = self._nearest_rate(self.sample_rate)
        self._target_frame_ms = frame_ms if frame_ms in _SUPPORTED_FRAME_MS else 20
        self._samples_per_frame = self._target_rate * self._target_frame_ms // 1000
        self._vad = _webrtcvad.Vad(self.mode)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._last_prob = 0.0

    # ------------------------------------------------------------------

    @staticmethod
    def _nearest_rate(sample_rate: int) -> int:
        if sample_rate in _SUPPORTED_RATES:
            return sample_rate
        return min(_SUPPORTED_RATES, key=lambda r: abs(r - sample_rate))

    def reset(self) -> None:
        super().reset()
        self._buffer = np.zeros(0, dtype=np.float32)
        self._last_prob = 0.0
        self._vad = _webrtcvad.Vad(self.mode)

    def process_frame(self, chunk: AudioChunk) -> VADResult:
        samples = self.to_mono_float(chunk)
        energy_db = rms_dbfs(samples)
        resampled = self._resample(samples, int(chunk.sample_rate), self._target_rate)
        self._buffer = np.concatenate([self._buffer, resampled]) if self._buffer.size else resampled

        votes: list[bool] = []
        while self._buffer.size >= self._samples_per_frame:
            block = self._buffer[: self._samples_per_frame]
            self._buffer = self._buffer[self._samples_per_frame :]
            pcm = (np.clip(block, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            try:
                votes.append(bool(self._vad.is_speech(pcm, self._target_rate)))
            except Exception:  # noqa: BLE001 - 帧长/采样率异常时保守判为非语音
                votes.append(False)

        if votes:
            prob = sum(1.0 for v in votes if v) / len(votes)
            self._last_prob = prob
        else:
            # 帧不足一个 webrtc 帧，沿用上一次概率（避免空洞）
            prob = self._last_prob

        is_speech = prob >= self.threshold
        self._account(is_speech)
        return VADResult(
            is_speech=is_speech,
            prob=float(prob),
            pts_ms=int(chunk.pts_ms),
            energy_db=energy_db,
            backend=self.name,
            duration_ms=float(chunk.duration_ms) or float(self.frame_ms),
            noise_floor_db=-60.0,
            details={"mode": float(self.mode), "sub_frames": float(len(votes))},
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """轻量线性重采样。VAD 对相位不敏感，无需 soxr 级质量。"""
        if src_rate == dst_rate or samples.size == 0:
            return samples.astype(np.float32, copy=False)
        dst_len = max(1, int(round(samples.size * dst_rate / float(src_rate))))
        src_idx = np.linspace(0.0, samples.size - 1.0, num=dst_len, dtype=np.float64)
        return np.interp(src_idx, np.arange(samples.size, dtype=np.float64), samples).astype(np.float32)

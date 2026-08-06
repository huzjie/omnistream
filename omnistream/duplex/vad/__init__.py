"""VAD 子包：统一导出与工厂 :func:`create_vad`。

后端选择策略（``backend="auto"``）::

    silero (onnxruntime + 权重可用)
      └─ 失败 → webrtc (webrtcvad 已安装)
           └─ 失败 → energy (纯 numpy，永远可用)

这条降级链保证"无 GPU、无模型权重、无编译环境"的机器上全双工链路仍然端到端可跑。
"""
from __future__ import annotations

import structlog

from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import DuplexError
from omnistream.duplex.vad.base import (
    DEFAULT_FRAME_MS,
    DEFAULT_SAMPLE_RATE,
    BaseVAD,
    VADResult,
    VADUnavailable,
    rms_dbfs,
    spectral_flatness,
    zero_crossing_rate,
)
from omnistream.duplex.vad.energy_vad import EnergyVAD, peak_dbfs
from omnistream.duplex.vad.ensemble import VADEnsemble, WeightedVAD
from omnistream.duplex.vad.silero_vad import SileroVAD
from omnistream.duplex.vad.silero_vad import is_available as silero_available
from omnistream.duplex.vad.webrtc_vad import WebRTCVAD
from omnistream.duplex.vad.webrtc_vad import is_available as webrtc_available

log = structlog.get_logger(__name__)

#: 支持的后端名
BACKENDS: tuple[str, ...] = ("auto", "silero", "webrtc", "energy", "ensemble")

#: Ensemble 默认权重（Silero 主导，webrtc 辅助，energy 兜底）
ENSEMBLE_WEIGHTS: dict[str, float] = {"silero": 0.6, "webrtc": 0.25, "energy": 0.15}


def create_vad(
    backend: str = "auto",
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    frame_ms: int = DEFAULT_FRAME_MS,
    threshold: float = 0.5,
    webrtc_mode: int = 2,
    silero_model_path: str | None = None,
    allow_model_download: bool = True,
    strict: bool = False,
) -> BaseVAD:
    """按名创建 VAD 后端；不可用时自动降级。

    Args:
        backend: ``auto`` / ``silero`` / ``webrtc`` / ``energy`` / ``ensemble``。
        sample_rate: 输入采样率。
        frame_ms: 输入帧长（毫秒）。
        threshold: 语音判定概率阈值。
        webrtc_mode: webrtcvad 激进度 0~3。
        silero_model_path: Silero 权重路径，None 用默认缓存目录。
        allow_model_download: 允许首次运行自动下载 Silero 权重。
        strict: True 时指定后端不可用直接抛错，不降级。

    Returns:
        一个可用的 :class:`BaseVAD` 实例。

    Raises:
        DuplexError: ``strict=True`` 且指定后端不可用，或所有后端都不可用（理论不可能）。
    """
    name = (backend or "auto").strip().lower()
    if name not in BACKENDS:
        log.warning("vad_backend_unknown", backend=name, fallback="auto")
        name = "auto"

    if name == "energy":
        return EnergyVAD(sample_rate=sample_rate, frame_ms=frame_ms, threshold=threshold)

    if name == "ensemble":
        return _build_ensemble(
            sample_rate=sample_rate, frame_ms=frame_ms, threshold=threshold,
            webrtc_mode=webrtc_mode, silero_model_path=silero_model_path,
            allow_model_download=allow_model_download,
        )

    if name in ("silero", "auto"):
        try:
            return SileroVAD(
                sample_rate=sample_rate, frame_ms=frame_ms, threshold=threshold,
                model_path=silero_model_path, allow_download=allow_model_download,
            )
        except VADUnavailable as exc:
            if strict and name == "silero":
                raise DuplexError(str(exc), code=ErrorCode.VAD_BACKEND_MISSING) from exc
            log.warning("vad_silero_unavailable", error=str(exc), fallback="webrtc")

    if name in ("webrtc", "auto", "silero"):
        try:
            return WebRTCVAD(sample_rate=sample_rate, frame_ms=frame_ms,
                             threshold=threshold, mode=webrtc_mode)
        except VADUnavailable as exc:
            if strict and name == "webrtc":
                raise DuplexError(str(exc), code=ErrorCode.VAD_BACKEND_MISSING) from exc
            log.warning("vad_webrtc_unavailable", error=str(exc), fallback="energy")

    log.info("vad_backend_selected", backend="energy", requested=name)
    return EnergyVAD(sample_rate=sample_rate, frame_ms=frame_ms, threshold=threshold)


def _build_ensemble(
    *,
    sample_rate: int,
    frame_ms: int,
    threshold: float,
    webrtc_mode: int,
    silero_model_path: str | None,
    allow_model_download: bool,
) -> BaseVAD:
    """尽力组装所有可用后端；只剩一个时直接返回它，避免无谓开销。"""
    members: list[WeightedVAD] = []
    try:
        members.append(WeightedVAD(
            SileroVAD(sample_rate=sample_rate, frame_ms=frame_ms, threshold=threshold,
                      model_path=silero_model_path, allow_download=allow_model_download),
            ENSEMBLE_WEIGHTS["silero"],
        ))
    except VADUnavailable as exc:
        log.warning("vad_ensemble_skip_member", member="silero", error=str(exc))
    try:
        members.append(WeightedVAD(
            WebRTCVAD(sample_rate=sample_rate, frame_ms=frame_ms, threshold=threshold, mode=webrtc_mode),
            ENSEMBLE_WEIGHTS["webrtc"],
        ))
    except VADUnavailable as exc:
        log.warning("vad_ensemble_skip_member", member="webrtc", error=str(exc))
    members.append(WeightedVAD(
        EnergyVAD(sample_rate=sample_rate, frame_ms=frame_ms, threshold=threshold),
        ENSEMBLE_WEIGHTS["energy"],
    ))

    if len(members) == 1:
        log.info("vad_ensemble_degraded_to_single", backend=members[0].vad.name)
        return members[0].vad
    ensemble = VADEnsemble(members, sample_rate=sample_rate, frame_ms=frame_ms, threshold=threshold)
    log.info("vad_ensemble_ready", backends=ensemble.backend_names)
    return ensemble


def available_backends() -> dict[str, bool]:
    """返回各后端的可用性，供 ``/healthz`` 与 ``oms doctor`` 展示。"""
    return {
        "silero": silero_available(),
        "webrtc": webrtc_available(),
        "energy": True,
    }


__all__ = [
    "BACKENDS",
    "DEFAULT_FRAME_MS",
    "DEFAULT_SAMPLE_RATE",
    "ENSEMBLE_WEIGHTS",
    "BaseVAD",
    "EnergyVAD",
    "SileroVAD",
    "VADEnsemble",
    "VADResult",
    "VADUnavailable",
    "WebRTCVAD",
    "WeightedVAD",
    "available_backends",
    "create_vad",
    "peak_dbfs",
    "rms_dbfs",
    "silero_available",
    "spectral_flatness",
    "webrtc_available",
    "zero_crossing_rate",
]

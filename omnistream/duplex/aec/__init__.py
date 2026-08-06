"""回声消除（AEC）子包。

默认策略：**直通**。浏览器/移动端 SDK 已开启 WebRTC AEC 时，
服务端不应再叠加一层，否则只会引入额外延迟与音质损伤。
只有服务端混音、SIP/PSTN 网关、硬件终端这类"服务端拿到裸麦克风信号"的场景
才需要 :class:`SpeexAEC`。

工厂 :func:`create_aec` 提供依赖缺失时的自动降级：``speex → passthrough``。
"""
from __future__ import annotations

from typing import Final

import structlog

from omnistream.duplex.aec.base import (
    DEFAULT_FRAME_MS,
    DEFAULT_SAMPLE_RATE,
    AECStats,
    AECUnavailable,
    BaseAEC,
    energy_dbfs,
)
from omnistream.duplex.aec.passthrough import PassthroughAEC
from omnistream.duplex.aec.reference import (
    DEFAULT_INITIAL_DELAY_MS,
    DEFAULT_MAX_DELAY_MS,
    ReferenceSignal,
)
from omnistream.duplex.aec.speex_aec import SpeexAEC
from omnistream.duplex.aec.speex_aec import is_available as speex_available

log = structlog.get_logger(__name__)

#: 支持的后端名。
BACKENDS: Final[tuple[str, ...]] = ("auto", "passthrough", "speex", "none")


def create_aec(
    backend: str = "auto",
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    frame_ms: int = DEFAULT_FRAME_MS,
    channels: int = 1,
    filter_length_ms: int = 200,
    initial_delay_ms: int = DEFAULT_INITIAL_DELAY_MS,
    residual_suppression: bool = False,
) -> BaseAEC:
    """按名字创建 AEC 实例，依赖缺失时自动降级。

    Args:
        backend: ``auto`` / ``passthrough`` / ``speex`` / ``none``。
            ``auto`` 等价于 ``passthrough``（尊重客户端 AEC）。
        sample_rate: 采样率。
        frame_ms: 帧长（毫秒）。
        channels: 声道数。
        filter_length_ms: speex 滤波器拖尾长度。
        initial_delay_ms: 参考信号初始延迟猜测。
        residual_suppression: passthrough 是否开启轻量残余抑制。

    Returns:
        :class:`BaseAEC` 实例；永不返回 ``None``，永不抛依赖类异常。
    """
    name = (backend or "auto").strip().lower()

    if name == "speex":
        if speex_available():
            try:
                return SpeexAEC(sample_rate=sample_rate, frame_ms=frame_ms, channels=channels,
                                filter_length_ms=filter_length_ms,
                                initial_delay_ms=initial_delay_ms)
            except AECUnavailable as exc:
                log.warning("aec.backend_degraded", requested="speex",
                            fallback="passthrough", error=str(exc))
        else:
            log.warning("aec.backend_missing", requested="speex", fallback="passthrough")

    track_reference = name != "none"
    return PassthroughAEC(
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        channels=channels,
        track_reference=track_reference,
        residual_suppression=residual_suppression,
    )


def available_backends() -> dict[str, bool]:
    """各后端依赖是否就绪，供 ``/healthz`` 与 CLI 自检展示。"""
    return {"passthrough": True, "speex": speex_available(), "none": True}


__all__ = [
    "BACKENDS",
    "DEFAULT_FRAME_MS",
    "DEFAULT_INITIAL_DELAY_MS",
    "DEFAULT_MAX_DELAY_MS",
    "DEFAULT_SAMPLE_RATE",
    "AECStats",
    "AECUnavailable",
    "BaseAEC",
    "PassthroughAEC",
    "ReferenceSignal",
    "SpeexAEC",
    "available_backends",
    "create_aec",
    "energy_dbfs",
    "speex_available",
]

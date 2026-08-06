"""媒体层异常定义。

统一继承 :class:`omnistream.core.errors.MediaError`，错误码取自
:class:`omnistream.core.error_codes.ErrorCode` 的 40 段（媒体域）。

``TransportClosed`` 由 WS1 在 ``core/errors.py`` 中定义，这里只做再导出，
保证 ``from omnistream.media.errors import TransportClosed`` 与
``from omnistream.core.errors import TransportClosed`` 是**同一个类**，
异常捕获不会因导入路径不同而漏网。
"""
from __future__ import annotations

from typing import Any

from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import MediaError, TransportClosed

__all__ = [
    "ClockDriftError",
    "CodecError",
    "IceFailed",
    "MediaError",
    "MediaStalled",
    "OptionalDependencyMissing",
    "ResampleError",
    "SignalingError",
    "TransportClosed",
]


class CodecError(MediaError):
    """编解码失败（Opus / JPEG / H.264 / μ-law）。"""

    default_code = ErrorCode.CODEC_ERROR


class ResampleError(MediaError):
    """重采样失败或参数非法。"""

    default_code = ErrorCode.RESAMPLE_ERROR


class ClockDriftError(MediaError):
    """时间轴漂移超过容忍阈值，pts 已不可信。"""

    default_code = ErrorCode.CLOCK_DRIFT


class IceFailed(MediaError):
    """ICE 连接建立失败或候选交换异常。"""

    default_code = ErrorCode.ICE_FAILED


class MediaStalled(MediaError):
    """媒体链路静默超时（轨道断开 / 上行中断）。"""

    default_code = ErrorCode.MEDIA_STALLED


class SignalingError(MediaError):
    """SDP 协商流程异常（offer/answer 顺序错误、会话不存在等）。"""

    default_code = ErrorCode.PROTOCOL_MALFORMED


class OptionalDependencyMissing(MediaError):
    """可选重依赖（aiortc / av / PIL / soxr）未安装。

    这是**可预期**的降级信号，不是 bug：调用方应捕获它并切换到降级路径
    （WebSocket 传输 + PCM16 直通 + RGB24 原始帧）。
    """

    default_code = ErrorCode.NOT_IMPLEMENTED

    def __init__(self, package: str, feature: str, *, install_hint: str = "", **kw: Any) -> None:
        hint = install_hint or f"pip install {package}"
        super().__init__(
            f"optional dependency '{package}' is required for {feature}; install with: {hint}",
            **kw,
        )
        self.package = package
        self.feature = feature
        self.detail.setdefault("package", package)
        self.detail.setdefault("feature", feature)
        self.detail.setdefault("install_hint", hint)

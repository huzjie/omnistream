"""媒体层（WS3 — Realtime Media）：实时音视频接入、处理与自适应。

子包一览
--------
* ``audio`` —— 格式/重采样/分帧/抖动缓冲/播放队列/增益，纯 numpy 可运行。
* ``video`` —— 帧容器/场景检测/自适应抽帧/缩放/编解码/屏幕共享。
* ``transport`` —— 回环、WebSocket、WebRTC 三种链路的统一抽象。
* ``pipeline`` —— ingress / egress / video_ingress 三条管线与会话媒体上下文。
* ``adaptive`` —— 带宽估计、码率分配、质量档位、拥塞与策略。
* ``simulate`` —— 音视频信号源、网络损伤、脚本化用户，供 Demo 与压测使用。

依赖策略
--------
本包的**根模块**（``constants`` / ``errors`` / ``metrics``）不依赖任何可选第三方库，
任何环境下都可以 import。子包按需延迟加载（见 :func:`__getattr__`），
这样 ``from omnistream.media import constants`` 不会连带拉起 PyAV / aiortc。

可选重依赖（aiortc / av / opuslib / Pillow / soxr / prometheus_client）缺失时，
对应能力抛 :class:`~omnistream.media.errors.OptionalDependencyMissing`，
其余链路自动降级运行。
"""
from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING, Any

from omnistream.media import constants as constants
from omnistream.media import errors as errors
from omnistream.media import metrics as metrics
from omnistream.media.errors import (
    ClockDriftError,
    CodecError,
    IceFailed,
    MediaError,
    MediaStalled,
    OptionalDependencyMissing,
    ResampleError,
    SignalingError,
    TransportClosed,
)
from omnistream.media.metrics import PROMETHEUS_AVAILABLE

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查
    from omnistream.media import adaptive, audio, pipeline, simulate, transport, video

#: 延迟加载的子包名。
_LAZY_SUBPACKAGES: frozenset[str] = frozenset(
    {"adaptive", "audio", "pipeline", "simulate", "transport", "video"}
)

__all__ = [
    "PROMETHEUS_AVAILABLE",
    "ClockDriftError",
    "CodecError",
    "IceFailed",
    "MediaError",
    "MediaStalled",
    "OptionalDependencyMissing",
    "ResampleError",
    "SignalingError",
    "TransportClosed",
    "adaptive",
    "audio",
    "constants",
    "errors",
    "metrics",
    "pipeline",
    "simulate",
    "transport",
    "video",
]


def __getattr__(name: str) -> ModuleType | Any:
    """按需导入子包，避免根 import 拉起可选重依赖。

    Args:
        name: 属性名。

    Returns:
        对应的子模块对象。

    Raises:
        AttributeError: 名称不是已知子包。
    """
    if name in _LAZY_SUBPACKAGES:
        module = importlib.import_module(f"omnistream.media.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'omnistream.media' has no attribute {name!r}")


def __dir__() -> list[str]:
    """补全交互式提示。"""
    return sorted(set(__all__) | set(globals()))

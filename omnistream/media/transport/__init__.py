"""传输子包：回环 / WebSocket / WebRTC 三种链路的统一入口。

选型建议
--------
============  ==============  ==========================================
链路          额外依赖        适用场景
============  ==============  ==========================================
loopback      无              Demo、压测、CI、离线回放
websocket     无              生产默认。任何反向代理都能过，最省心
webrtc        aiortc          端到端时延敏感、弱网多、需要浏览器原生 AEC
============  ==============  ==========================================

:func:`create_transport` 在请求 ``webrtc`` 但 aiortc 缺失时会**自动降级**到
WebSocket（前提是给了 socket 对象），并在日志里留下 ``transport_downgraded``。
这条降级路径是有意为之：自托管用户装不上 aiortc 不应该导致整个服务不可用。
"""
from __future__ import annotations

from typing import Any

from omnistream.core.logging import get_logger
from omnistream.media.errors import OptionalDependencyMissing
from omnistream.media.transport.base import (
    BaseTransport,
    LinkImpairment,
    TransportConfig,
    TransportKind,
    TransportState,
)
from omnistream.media.transport.datachannel import (
    CHUNK_HEADER_SIZE,
    ChunkHeader,
    DataChannelCodec,
    Reassembler,
    ReassemblyStats,
    fragment,
)
from omnistream.media.transport.ice import (
    CandidateInfo,
    IceConfig,
    IceServer,
    build_ice_servers,
    ensure_connectable,
    ice_config_from_settings,
    parse_candidate,
    summarize_candidates,
    turn_credentials,
)
from omnistream.media.transport.loopback import LoopbackTransport
from omnistream.media.transport.stats import (
    StatsSample,
    TransportStats,
    TransportStatsCollector,
)
from omnistream.media.transport.webrtc import AIORTC_AVAILABLE, WebRtcTransport
from omnistream.media.transport.webrtc_signaling import (
    SIGNALING_REGISTRY,
    SessionDescription,
    SignalingRegistry,
    SignalingSession,
    SignalingState,
    prefer_codec,
    strip_video,
    tune_opus,
    validate_sdp,
)
from omnistream.media.transport.websocket import (
    CallableWebSocketAdapter,
    ClientWebSocketAdapter,
    ServerWebSocketAdapter,
    WebSocketAdapter,
    WebSocketTransport,
    adapt_websocket,
)

__all__ = [
    "AIORTC_AVAILABLE",
    "CHUNK_HEADER_SIZE",
    "SIGNALING_REGISTRY",
    "BaseTransport",
    "CallableWebSocketAdapter",
    "CandidateInfo",
    "ChunkHeader",
    "ClientWebSocketAdapter",
    "DataChannelCodec",
    "IceConfig",
    "IceServer",
    "LinkImpairment",
    "LoopbackTransport",
    "Reassembler",
    "ReassemblyStats",
    "ServerWebSocketAdapter",
    "SessionDescription",
    "SignalingRegistry",
    "SignalingSession",
    "SignalingState",
    "StatsSample",
    "TransportConfig",
    "TransportKind",
    "TransportState",
    "TransportStats",
    "TransportStatsCollector",
    "WebRtcTransport",
    "WebSocketAdapter",
    "WebSocketTransport",
    "adapt_websocket",
    "available_transports",
    "build_ice_servers",
    "create_transport",
    "ensure_connectable",
    "fragment",
    "ice_config_from_settings",
    "parse_candidate",
    "prefer_codec",
    "strip_video",
    "summarize_candidates",
    "tune_opus",
    "turn_credentials",
    "validate_sdp",
]

_log = get_logger(__name__)


def available_transports() -> list[str]:
    """列出当前环境真正可用的链路类型。

    Returns:
        形如 ``["loopback", "websocket", "webrtc"]`` 的列表。
    """
    kinds = [str(TransportKind.LOOPBACK), str(TransportKind.WEBSOCKET)]
    if AIORTC_AVAILABLE:
        kinds.append(str(TransportKind.WEBRTC))
    return kinds


def create_transport(
    kind: str | TransportKind = TransportKind.LOOPBACK,
    *,
    config: TransportConfig | None = None,
    websocket: Any = None,
    allow_downgrade: bool = True,
    **options: Any,
) -> BaseTransport:
    """按类型创建一条媒体链路。

    Args:
        kind: ``loopback`` / ``websocket`` / ``webrtc``。
        config: 链路配置，``None`` 时使用默认值。
        websocket: WebSocket 场景下的 socket 对象或适配器。
        allow_downgrade: 请求 webrtc 但 aiortc 缺失时是否降级到 WebSocket。
        **options: 透传给具体实现的额外参数
            （如 loopback 的 ``echo`` / ``impairment``，webrtc 的 ``ice_config``）。

    Returns:
        尚未 ``start()`` 的链路实例。

    Raises:
        ValueError: 未知链路类型，或 WebSocket 链路缺少 socket 对象。
        OptionalDependencyMissing: 请求 webrtc、aiortc 缺失且不允许降级。
    """
    try:
        resolved = TransportKind(str(kind))
    except ValueError as exc:
        raise ValueError(
            f"unknown transport kind {kind!r}; expected one of {available_transports()}"
        ) from exc

    if resolved is TransportKind.LOOPBACK:
        return LoopbackTransport(config, **options)

    if resolved is TransportKind.WEBSOCKET:
        if websocket is None:
            raise ValueError("websocket transport requires a websocket object")
        return WebSocketTransport(websocket, config, **options)

    if AIORTC_AVAILABLE:
        return WebRtcTransport(config, **options)

    if allow_downgrade and websocket is not None:
        _log.warning(
            "transport_downgraded",
            requested=str(resolved),
            fallback=str(TransportKind.WEBSOCKET),
            reason="aiortc_missing",
        )
        return WebSocketTransport(websocket, config)

    raise OptionalDependencyMissing(
        "aiortc", "webrtc transport", install_hint="pip install 'omnistream[webrtc]'"
    )

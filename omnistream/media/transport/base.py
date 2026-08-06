"""传输层抽象：会话媒体链路的统一契约。

三种实现共享同一套语义
----------------------
* :class:`~omnistream.media.transport.loopback.LoopbackTransport` —— 进程内回环，
  Demo / 压测 / 联调用，可注入网络损伤。
* :class:`~omnistream.media.transport.websocket.WebSocketTransport` —— 生产默认，
  文本走 :class:`~omnistream.protocol.envelope.Envelope`，媒体走 OMS1 二进制帧。
* :class:`~omnistream.media.transport.webrtc.WebRtcTransport` —— 低延迟可选，
  需要 aiortc；缺失时由 ``create_transport`` 自动降级到 WebSocket。

统一约定
--------
* **上行**（对端 → 平台）一律进有界流：音频 ``BLOCK`` 背压、视频 ``DROP_OLDEST``、
  事件 ``DROP_OLDEST``。任何实现都不允许自己开无界队列。
* **下行**（平台 → 对端）一律走 ``send_audio`` / ``send_video`` / ``send_event``。
* ``stop()`` 之后上行流必须自然结束，``recv_*`` 的 ``async for`` 会正常退出。
* pts 一律是**会话内相对毫秒**，由上游时间轴负责，传输层只透传不改写。

子类只需要实现四个钩子：``_open`` / ``_close`` / ``_send_binary`` / ``_send_text``。
"""
from __future__ import annotations

import abc
import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from omnistream.core.clock import now_ms, now_ns
from omnistream.core.ids import new_session_id
from omnistream.core.logging import get_logger
from omnistream.core.streams import BoundedStream, OverflowPolicy, StreamClosed
from omnistream.media.constants import (
    AUDIO_STREAM_MAXSIZE,
    DEFAULT_CHANNELS,
    DEFAULT_FRAME_MS,
    DEFAULT_INPUT_SAMPLE_RATE,
    EVENT_STREAM_MAXSIZE,
    MEDIA_STALL_TIMEOUT_MS,
    STATS_INTERVAL_MS,
    VIDEO_STREAM_MAXSIZE,
    WS_HEARTBEAT_INTERVAL_MS,
    WS_IDLE_TIMEOUT_MS,
    WS_SEND_TIMEOUT_MS,
)
from omnistream.media.errors import TransportClosed
from omnistream.media.metrics import media_stalls, transport_connections, transport_errors
from omnistream.media.transport.stats import TransportStats, TransportStatsCollector
from omnistream.protocol.binary import (
    KIND_AUDIO,
    KIND_VIDEO,
    decode_frame,
    encode_audio,
    encode_video,
    is_binary_frame,
)
from omnistream.protocol.enums import AudioEncoding, ServerEventType, VideoEncoding
from omnistream.protocol.envelope import Envelope
from omnistream.schemas.audio import AudioChunk
from omnistream.schemas.video import VideoFrame

__all__ = [
    "BaseTransport",
    "LinkImpairment",
    "TransportConfig",
    "TransportKind",
    "TransportState",
]

_log = get_logger(__name__)


class TransportKind(StrEnum):
    """链路类型。"""

    LOOPBACK = "loopback"
    WEBSOCKET = "websocket"
    WEBRTC = "webrtc"


class TransportState(StrEnum):
    """链路生命周期状态。"""

    IDLE = "idle"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"


@runtime_checkable
class LinkImpairment(Protocol):
    """网络损伤注入器的结构化协议。

    由 :mod:`omnistream.media.simulate.network` 实现；传输层只按结构调用，
    因此 ``simulate`` 子包对 ``transport`` 没有反向依赖。
    """

    def plan(self, payload: bytes) -> tuple[float, bool]:
        """为一个待发送分片规划投递结果。

        Args:
            payload: 待发送的字节。

        Returns:
            ``(delay_s, dropped)``：投递延迟（秒）与是否丢弃。
        """
        ...


@dataclass(slots=True)
class TransportConfig:
    """链路配置。所有时间参数单位均为毫秒。"""

    session_id: str = ""
    kind: TransportKind = TransportKind.LOOPBACK
    sample_rate: int = DEFAULT_INPUT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    frame_ms: int = DEFAULT_FRAME_MS
    audio_encoding: AudioEncoding = AudioEncoding.PCM16
    video_encoding: VideoEncoding = VideoEncoding.JPEG
    audio_maxsize: int = AUDIO_STREAM_MAXSIZE
    video_maxsize: int = VIDEO_STREAM_MAXSIZE
    event_maxsize: int = EVENT_STREAM_MAXSIZE
    heartbeat_interval_ms: int = WS_HEARTBEAT_INTERVAL_MS
    idle_timeout_ms: int = WS_IDLE_TIMEOUT_MS
    send_timeout_ms: int = WS_SEND_TIMEOUT_MS
    stall_timeout_ms: int = MEDIA_STALL_TIMEOUT_MS
    stats_interval_ms: int = STATS_INTERVAL_MS
    enable_video: bool = True
    enable_stats: bool = True
    enable_watchdog: bool = True

    def __post_init__(self) -> None:
        """补全会话 ID 并做基本收敛。"""
        if not self.session_id:
            self.session_id = new_session_id()
        self.sample_rate = max(1, int(self.sample_rate))
        self.channels = max(1, int(self.channels))
        self.frame_ms = max(1, int(self.frame_ms))
        self.audio_maxsize = max(1, int(self.audio_maxsize))
        self.video_maxsize = max(1, int(self.video_maxsize))
        self.event_maxsize = max(1, int(self.event_maxsize))

    def as_dict(self) -> dict[str, Any]:
        """导出为可序列化字典。"""
        return {
            "session_id": self.session_id,
            "kind": str(self.kind),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frame_ms": self.frame_ms,
            "audio_encoding": str(self.audio_encoding),
            "video_encoding": str(self.video_encoding),
            "heartbeat_interval_ms": self.heartbeat_interval_ms,
            "idle_timeout_ms": self.idle_timeout_ms,
            "stall_timeout_ms": self.stall_timeout_ms,
            "enable_video": self.enable_video,
        }


class BaseTransport(abc.ABC):
    """媒体链路抽象基类。

    Example::

        transport = LoopbackTransport(TransportConfig(session_id="sess_x"))
        await transport.start()
        async for chunk in transport.recv_audio():
            await transport.send_audio(chunk)      # 回声
        await transport.stop()
    """

    #: 子类覆盖，用于指标标签与日志。
    kind: TransportKind = TransportKind.LOOPBACK

    def __init__(self, config: TransportConfig | None = None) -> None:
        """
        Args:
            config: 链路配置，``None`` 时使用默认配置并自动分配 session_id。
        """
        self.config = config or TransportConfig(kind=self.kind)
        if config is not None and config.kind != self.kind:
            self.config.kind = self.kind
        self.session_id = self.config.session_id

        self.audio_in: BoundedStream[AudioChunk] = BoundedStream(
            maxsize=self.config.audio_maxsize,
            policy=OverflowPolicy.BLOCK,
            name=f"{self.kind}:audio_in",
        )
        self.video_in: BoundedStream[VideoFrame] = BoundedStream(
            maxsize=self.config.video_maxsize,
            policy=OverflowPolicy.DROP_OLDEST,
            name=f"{self.kind}:video_in",
        )
        self.events_in: BoundedStream[Envelope] = BoundedStream(
            maxsize=self.config.event_maxsize,
            policy=OverflowPolicy.DROP_OLDEST,
            name=f"{self.kind}:events_in",
        )

        self.stats = TransportStats(kind=str(self.kind), session_id=self.session_id)
        self._collector = TransportStatsCollector(
            self.stats,
            interval_ms=self.config.stats_interval_ms,
            name=f"stats:{self.kind}:{self.session_id}",
        )

        self._state: TransportState = TransportState.IDLE
        self._stalled = False
        self._seq = 0
        self._close_reason = ""
        self._watchdog_task: asyncio.Task[None] | None = None
        self._close_callbacks: list[Callable[[str], None]] = []
        self._stall_callbacks: list[Callable[[int], None]] = []
        self._send_lock = asyncio.Lock()
        self._log = _log.bind(session_id=self.session_id, transport=str(self.kind))

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    @property
    def state(self) -> TransportState:
        """当前链路状态。"""
        return self._state

    @property
    def connected(self) -> bool:
        """是否处于可收发状态。"""
        return self._state is TransportState.CONNECTED

    @property
    def closed(self) -> bool:
        """是否已关闭或失败。"""
        return self._state in (TransportState.CLOSED, TransportState.FAILED)

    @property
    def stalled(self) -> bool:
        """是否已被看门狗判定为静默。"""
        return self._stalled

    @property
    def close_reason(self) -> str:
        """关闭原因（尚未关闭时为空串）。"""
        return self._close_reason

    def next_seq(self) -> int:
        """取下一个下行事件序号（会话内单调递增）。"""
        self._seq += 1
        return self._seq

    # ------------------------------------------------------------------
    # 回调注册
    # ------------------------------------------------------------------

    def on_close(self, callback: Callable[[str], None]) -> None:
        """注册关闭回调，参数为关闭原因。"""
        self._close_callbacks.append(callback)

    def on_stall(self, callback: Callable[[int], None]) -> None:
        """注册静默回调，参数为静默毫秒数。"""
        self._stall_callbacks.append(callback)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """建立链路（幂等）。

        Raises:
            TransportClosed: 链路已关闭后再次 start。
        """
        if self._state is TransportState.CONNECTED:
            return
        if self.closed:
            raise TransportClosed(
                "transport already closed", detail={"session_id": self.session_id}
            )
        self._state = TransportState.CONNECTING
        try:
            await self._open()
        except Exception as exc:
            self._state = TransportState.FAILED
            self._close_reason = f"open_failed: {exc}"
            self.stats.on_error()
            try:
                transport_errors.labels(kind=str(self.kind), reason="open_failed").inc()
            except Exception:
                pass
            self._log.warning("transport_open_failed", error=str(exc))
            raise
        self._state = TransportState.CONNECTED
        try:
            transport_connections.labels(kind=str(self.kind)).inc()
        except Exception:
            pass
        if self.config.enable_stats:
            self._collector.start()
        if self.config.enable_watchdog and self.config.stall_timeout_ms > 0:
            self._watchdog_task = asyncio.create_task(
                self._watchdog(), name=f"watchdog:{self.kind}:{self.session_id}"
            )
        self._log.info("transport_started", config=self.config.as_dict())

    async def stop(self, reason: str = "normal") -> None:
        """关闭链路并让上行流自然结束（幂等）。

        Args:
            reason: 关闭原因，会透传给关闭回调与日志。
        """
        if self.closed or self._state is TransportState.CLOSING:
            return
        was_connected = self._state is TransportState.CONNECTED
        self._state = TransportState.CLOSING
        self._close_reason = reason

        task = self._watchdog_task
        self._watchdog_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        try:
            await self._close(reason)
        except Exception as exc:  # 关闭阶段的异常不再向上抛
            self._log.warning("transport_close_failed", error=str(exc))
        finally:
            await self._collector.stop()
            self.audio_in.close()
            self.video_in.close()
            self.events_in.close()
            self._state = TransportState.CLOSED
            if was_connected:
                try:
                    transport_connections.labels(kind=str(self.kind)).dec()
                except Exception:
                    pass
            for callback in list(self._close_callbacks):
                try:
                    callback(reason)
                except Exception as exc:
                    self._log.warning("transport_close_callback_failed", error=str(exc))
            self._log.info("transport_stopped", reason=reason, stats=self.stats.as_dict())

    async def __aenter__(self) -> "BaseTransport":
        """支持 ``async with``。"""
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """退出上下文时关闭链路。"""
        reason = "error" if exc_info and exc_info[0] is not None else "normal"
        await self.stop(reason)

    # ------------------------------------------------------------------
    # 抽象钩子
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def _open(self) -> None:
        """建立底层连接。失败时抛异常。"""

    @abc.abstractmethod
    async def _close(self, reason: str) -> None:
        """关闭底层连接。实现内部必须吞掉自己的清理异常。"""

    @abc.abstractmethod
    async def _send_binary(self, payload: bytes, *, media: str) -> None:
        """发送一帧 OMS1 二进制数据。

        Args:
            payload: 已编码的完整帧。
            media: ``audio`` 或 ``video``，仅用于统计标签。
        """

    @abc.abstractmethod
    async def _send_text(self, payload: bytes) -> None:
        """发送一条 UTF-8 JSON 文本消息（Envelope 序列化结果）。"""

    # ------------------------------------------------------------------
    # 下行发送
    # ------------------------------------------------------------------

    async def send_audio(self, chunk: AudioChunk) -> int:
        """下发一帧音频。

        Args:
            chunk: 音频帧，``pts_ms`` 为会话内相对时间。

        Returns:
            实际写出的字节数。

        Raises:
            TransportClosed: 链路不可用。
        """
        self._ensure_open()
        payload = encode_audio(
            self.session_id,
            int(chunk.pts_ms),
            chunk.data,
            sample_rate=int(chunk.sample_rate),
            channels=int(chunk.channels),
            encoding=str(chunk.encoding),
            seq=self.next_seq(),
        )
        await self._guarded_send(payload, media="audio")
        self.stats.on_sent(len(payload), media="audio")
        return len(payload)

    async def send_video(self, frame: VideoFrame) -> int:
        """下发一帧视频。

        Args:
            frame: 视频帧。

        Returns:
            实际写出的字节数；视频被禁用时返回 ``0``。
        """
        if not self.config.enable_video:
            return 0
        self._ensure_open()
        payload = encode_video(
            self.session_id,
            int(frame.pts_ms),
            frame.data,
            width=int(frame.width),
            height=int(frame.height),
            encoding=str(frame.encoding),
            keyframe_score=float(frame.keyframe_score),
        )
        await self._guarded_send(payload, media="video")
        self.stats.on_sent(len(payload), media="video")
        return len(payload)

    async def send_event(
        self,
        type_: str | ServerEventType,
        payload: dict[str, Any] | None = None,
        *,
        seq: int | None = None,
    ) -> int:
        """下发一条控制事件。

        Args:
            type_: 事件类型，取自 :class:`~omnistream.protocol.enums.ServerEventType`。
            payload: 事件负载。
            seq: 显式序号，``None`` 时自动分配。

        Returns:
            写出的字节数。
        """
        envelope = Envelope.server(
            str(type_),
            payload or {},
            session_id=self.session_id,
            seq=seq if seq is not None else self.next_seq(),
        )
        return await self.send_envelope(envelope)

    async def send_envelope(self, envelope: Envelope) -> int:
        """下发一个已构造好的 Envelope。"""
        self._ensure_open()
        raw = envelope.to_wire()
        await self._guarded_send_text(raw)
        self.stats.on_sent(len(raw), media="event")
        return len(raw)

    async def _guarded_send(self, payload: bytes, *, media: str) -> None:
        """带超时与串行化的二进制发送。"""
        timeout = self.config.send_timeout_ms / 1000.0
        try:
            async with self._send_lock:
                await asyncio.wait_for(self._send_binary(payload, media=media), timeout)
        except TimeoutError as exc:
            self.stats.on_error()
            try:
                transport_errors.labels(kind=str(self.kind), reason="send_timeout").inc()
            except Exception:
                pass
            raise TransportClosed(
                "media send timed out",
                detail={"session_id": self.session_id, "media": media, "timeout_s": timeout},
            ) from exc

    async def _guarded_send_text(self, payload: bytes) -> None:
        """带超时与串行化的文本发送。"""
        timeout = self.config.send_timeout_ms / 1000.0
        try:
            async with self._send_lock:
                await asyncio.wait_for(self._send_text(payload), timeout)
        except TimeoutError as exc:
            self.stats.on_error()
            try:
                transport_errors.labels(kind=str(self.kind), reason="send_timeout").inc()
            except Exception:
                pass
            raise TransportClosed(
                "event send timed out",
                detail={"session_id": self.session_id, "timeout_s": timeout},
            ) from exc

    def _ensure_open(self) -> None:
        """校验链路可用，否则抛 :class:`TransportClosed`。"""
        if self._state is not TransportState.CONNECTED:
            raise TransportClosed(
                f"transport not connected (state={self._state})",
                detail={"session_id": self.session_id, "state": str(self._state)},
            )

    # ------------------------------------------------------------------
    # 上行接收
    # ------------------------------------------------------------------

    async def recv_audio(self) -> AsyncIterator[AudioChunk]:
        """迭代上行音频帧，链路关闭后自然结束。"""
        async for chunk in self.audio_in:
            yield chunk

    async def recv_video(self) -> AsyncIterator[VideoFrame]:
        """迭代上行视频帧，链路关闭后自然结束。"""
        async for frame in self.video_in:
            yield frame

    async def recv_events(self) -> AsyncIterator[Envelope]:
        """迭代上行控制事件，链路关闭后自然结束。"""
        async for envelope in self.events_in:
            yield envelope

    # ------------------------------------------------------------------
    # 上行注入（供子类调用）
    # ------------------------------------------------------------------

    async def _emit_audio(self, chunk: AudioChunk) -> None:
        """把一帧上行音频送进有界流（满时背压）。"""
        try:
            await self.audio_in.put(chunk)
        except StreamClosed:
            return

    async def _emit_video(self, frame: VideoFrame) -> None:
        """把一帧上行视频送进有界流（满时丢最旧）。"""
        if not self.config.enable_video:
            return
        try:
            await self.video_in.put(frame)
        except StreamClosed:
            return

    async def _emit_event(self, envelope: Envelope) -> None:
        """把一条上行事件送进有界流。"""
        try:
            await self.events_in.put(envelope)
        except StreamClosed:
            return

    async def _handle_binary(self, raw: bytes) -> bool:
        """解析并分发一条上行二进制消息。

        Args:
            raw: 原始字节。

        Returns:
            是否成功识别并分发。
        """
        if not is_binary_frame(raw):
            self.stats.on_error()
            try:
                transport_errors.labels(kind=str(self.kind), reason="bad_magic").inc()
            except Exception:
                pass
            return False
        try:
            frame = decode_frame(raw)
        except Exception as exc:
            self.stats.on_error()
            try:
                transport_errors.labels(kind=str(self.kind), reason="decode_failed").inc()
            except Exception:
                pass
            self._log.warning("binary_frame_decode_failed", error=str(exc), nbytes=len(raw))
            return False

        header = frame.header
        if frame.kind == KIND_AUDIO:
            self.stats.on_recv(len(raw), media="audio")
            chunk = self._audio_from_header(header, frame.payload)
            if chunk is None:
                return False
            await self._emit_audio(chunk)
            return True
        if frame.kind == KIND_VIDEO:
            self.stats.on_recv(len(raw), media="video")
            video = self._video_from_header(header, frame.payload)
            if video is None:
                return False
            await self._emit_video(video)
            return True
        return False

    def _audio_from_header(self, header: dict[str, Any], payload: bytes) -> AudioChunk | None:
        """由 OMS1 头部构造 :class:`AudioChunk`。"""
        try:
            encoding = AudioEncoding(str(header.get("enc", AudioEncoding.PCM16)))
        except ValueError:
            self._log.warning("unknown_audio_encoding", enc=header.get("enc"))
            return None
        try:
            return AudioChunk(
                pts_ms=int(header.get("pts", 0)),
                data=payload,
                sample_rate=int(header.get("sr", self.config.sample_rate)),
                channels=int(header.get("ch", self.config.channels)),
                encoding=encoding,
                capture_ts_ns=now_ns(),
                speaker=header.get("spk"),
            )
        except Exception as exc:
            self._log.warning("audio_chunk_build_failed", error=str(exc))
            return None

    def _video_from_header(self, header: dict[str, Any], payload: bytes) -> VideoFrame | None:
        """由 OMS1 头部构造 :class:`VideoFrame`。"""
        try:
            encoding = VideoEncoding(str(header.get("enc", VideoEncoding.JPEG)))
        except ValueError:
            self._log.warning("unknown_video_encoding", enc=header.get("enc"))
            return None
        try:
            return VideoFrame(
                pts_ms=int(header.get("pts", 0)),
                data=payload,
                width=max(1, int(header.get("w", 1))),
                height=max(1, int(header.get("h", 1))),
                encoding=encoding,
                capture_ts_ns=now_ns(),
                keyframe_score=min(1.0, max(0.0, float(header.get("ks", 0.0)))),
                source=str(header.get("src", "camera")),
            )
        except Exception as exc:
            self._log.warning("video_frame_build_failed", error=str(exc))
            return None

    async def _handle_text(self, raw: str | bytes) -> bool:
        """解析并分发一条上行文本消息。

        Args:
            raw: JSON 文本或其 UTF-8 字节。

        Returns:
            是否成功解析。
        """
        data = raw.encode("utf-8") if isinstance(raw, str) else raw
        self.stats.on_recv(len(data), media="event")
        try:
            envelope = Envelope.from_wire(data)
        except Exception as exc:
            self.stats.on_error()
            try:
                transport_errors.labels(kind=str(self.kind), reason="bad_envelope").inc()
            except Exception:
                pass
            self._log.warning("envelope_decode_failed", error=str(exc))
            return False
        if envelope.session_id is None:
            envelope.session_id = self.session_id
        await self._emit_event(envelope)
        return True

    # ------------------------------------------------------------------
    # 看门狗
    # ------------------------------------------------------------------

    async def _watchdog(self) -> None:
        """周期检查上行静默时长，超时上报 stall。"""
        period = max(0.2, self.config.stall_timeout_ms / 4000.0)
        while not self.closed:
            try:
                await asyncio.sleep(period)
            except asyncio.CancelledError:
                return
            if self._state is not TransportState.CONNECTED:
                continue
            idle = self.stats.idle_ms
            if idle >= self.config.stall_timeout_ms:
                if not self._stalled:
                    self._stalled = True
                    try:
                        media_stalls.labels(media="link", reason="idle").inc()
                    except Exception:
                        pass
                    self._log.warning("media_stalled", idle_ms=idle)
                    for callback in list(self._stall_callbacks):
                        try:
                            callback(idle)
                        except Exception as exc:
                            self._log.warning("stall_callback_failed", error=str(exc))
            elif self._stalled:
                self._stalled = False
                self._log.info("media_recovered", idle_ms=idle)

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """导出链路快照，供 ``/v1/media/stats`` 与控制台使用。"""
        return {
            "kind": str(self.kind),
            "state": str(self._state),
            "session_id": self.session_id,
            "stalled": self._stalled,
            "close_reason": self._close_reason,
            "queued": {
                "audio_in": self.audio_in.size,
                "video_in": self.video_in.size,
                "events_in": self.events_in.size,
                "audio_dropped": self.audio_in.dropped,
                "video_dropped": self.video_in.dropped,
                "events_dropped": self.events_in.dropped,
            },
            "stats": self.stats.as_dict(),
            "config": self.config.as_dict(),
            "updated_ms": now_ms(),
        }

    def __repr__(self) -> str:
        """调试用文本表示。"""
        return (
            f"{type(self).__name__}(session={self.session_id}, "
            f"state={self._state}, sent={self.stats.bytes_sent}B, "
            f"recv={self.stats.bytes_recv}B)"
        )

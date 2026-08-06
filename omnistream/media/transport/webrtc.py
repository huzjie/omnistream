"""WebRTC 传输（aiortc 可选后端）。

为什么是可选的
--------------
aiortc 会连带拉起 PyAV / cryptography / pylibsrtp，在很多自托管环境里装不上或不想装。
所以本模块采用**可选导入**：``import`` 永远成功，只有真正构造
:class:`WebRtcTransport` 时才会因缺依赖抛
:class:`~omnistream.media.errors.OptionalDependencyMissing`，
上层（``create_transport`` / ``/v1/rtc/offer``）据此自动降级到 WebSocket。

链路分工
--------
* **音频** 走 RTP 音频轨（Opus，48kHz）—— 这是 WebRTC 相对 WebSocket 的核心收益：
  拥塞控制、FEC、乱序重排都由浏览器与 SRTP 栈处理。
* **视频** 与 **控制事件** 走 DataChannel —— 上行视频是低帧率感知输入（默认 0.5fps），
  用 DataChannel 传 JPEG 比开一路 VP8 编码器省得多，也避开了浏览器的自动降分辨率。
* DataChannel 消息统一经 :class:`~omnistream.media.transport.datachannel.DataChannelCodec`
  分片，避免撞上 SCTP 的单消息上限。
"""
from __future__ import annotations

import asyncio
import fractions
from typing import Any

import numpy as np

from omnistream.core.clock import monotonic_s
from omnistream.core.logging import get_logger
from omnistream.media.audio import pcm
from omnistream.media.audio.frame_slicer import FrameSlicer
from omnistream.media.audio.resampler import Resampler
from omnistream.media.constants import (
    DEFAULT_FRAME_MS,
    ICE_GATHER_TIMEOUT_MS,
    VLM_LONG_SIDE,
    WEBRTC_SAMPLE_RATE,
)
from omnistream.media.errors import IceFailed, OptionalDependencyMissing, TransportClosed
from omnistream.media.metrics import transport_errors
from omnistream.media.transport.base import BaseTransport, TransportConfig, TransportKind
from omnistream.media.transport.datachannel import DataChannelCodec
from omnistream.media.transport.ice import IceConfig, ice_config_from_settings
from omnistream.media.transport.webrtc_signaling import (
    SessionDescription,
    prefer_codec,
    tune_opus,
    validate_sdp,
)
from omnistream.media.video.downscaler import Downscaler
from omnistream.media.video.frame import PIL_AVAILABLE, RawVideoFrame
from omnistream.protocol.enums import AudioEncoding, VideoEncoding
from omnistream.schemas.audio import AudioChunk

__all__ = [
    "AIORTC_AVAILABLE",
    "WebRtcTransport",
    "is_available",
    "require_aiortc",
]

_log = get_logger(__name__)

try:  # pragma: no cover - 环境相关
    import av  # noqa: F401  (AudioFrame 构造需要)
    from aiortc import (
        RTCConfiguration,
        RTCIceServer,
        RTCPeerConnection,
        RTCSessionDescription,
    )
    from aiortc.mediastreams import MediaStreamError, MediaStreamTrack

    AIORTC_AVAILABLE = True
except ImportError:  # pragma: no cover - 环境相关
    av = None  # type: ignore[assignment]
    RTCConfiguration = RTCIceServer = RTCPeerConnection = None  # type: ignore[assignment]
    RTCSessionDescription = None  # type: ignore[assignment]
    MediaStreamError = Exception  # type: ignore[assignment,misc]
    MediaStreamTrack = object  # type: ignore[assignment,misc]
    AIORTC_AVAILABLE = False


def is_available() -> bool:
    """aiortc 是否可用。"""
    return AIORTC_AVAILABLE


def require_aiortc(feature: str) -> None:
    """确保 aiortc 已安装。

    Args:
        feature: 触发检查的能力名，写进错误信息。

    Raises:
        OptionalDependencyMissing: aiortc 未安装。
    """
    if not AIORTC_AVAILABLE:
        raise OptionalDependencyMissing(
            "aiortc", feature, install_hint="pip install 'omnistream[webrtc]'"
        )


_OUTBOUND_TRACK_CLS: Any = None


def _outbound_track_cls() -> Any:
    """惰性构造出栈音频轨类（继承 aiortc 的 MediaStreamTrack）。

    Returns:
        ``_OutboundAudioTrack`` 类对象。

    Raises:
        OptionalDependencyMissing: aiortc 未安装。
    """
    global _OUTBOUND_TRACK_CLS
    if _OUTBOUND_TRACK_CLS is not None:
        return _OUTBOUND_TRACK_CLS
    require_aiortc("webrtc outbound audio track")

    class _OutboundAudioTrack(MediaStreamTrack):  # type: ignore[misc,valid-type]
        """把 PlaybackQueue 的 PCM 按实时节拍推给 aiortc 的音频轨。

        缓冲为空时输出静音而不是阻塞 —— RTP 的时钟不等人，
        断流时补静音远比让整条轨卡住要好。
        """

        kind = "audio"

        def __init__(
            self,
            *,
            sample_rate: int = WEBRTC_SAMPLE_RATE,
            channels: int = 1,
            frame_ms: int = DEFAULT_FRAME_MS,
            max_buffer_bytes: int = 1 << 20,
        ) -> None:
            super().__init__()
            self.sample_rate = int(sample_rate)
            self.channels = max(1, int(channels))
            self.frame_ms = max(1, int(frame_ms))
            self.samples_per_frame = int(self.sample_rate * self.frame_ms / 1000)
            self.frame_bytes = self.samples_per_frame * 2 * self.channels
            self.max_buffer_bytes = int(max_buffer_bytes)
            self._buffer = bytearray()
            self._start: float | None = None
            self._timestamp = 0
            self._silence_frames = 0
            self._pushed_bytes = 0
            self._layout = "mono" if self.channels == 1 else "stereo"

        @property
        def pending_bytes(self) -> int:
            """缓冲中尚未发出的字节数。"""
            return len(self._buffer)

        @property
        def silence_frames(self) -> int:
            """因缓冲空而补出的静音帧数。"""
            return self._silence_frames

        def push(self, pcm16: bytes) -> int:
            """写入 PCM16 数据。

            Args:
                pcm16: 与本轨采样率/声道一致的小端 int16 字节。

            Returns:
                实际写入的字节数（缓冲溢出时会丢弃最旧数据）。
            """
            self._buffer.extend(pcm16)
            self._pushed_bytes += len(pcm16)
            overflow = len(self._buffer) - self.max_buffer_bytes
            if overflow > 0:
                del self._buffer[:overflow]
            return len(pcm16)

        def clear(self) -> int:
            """丢弃缓冲中的全部待发数据（barge-in 用）。"""
            count = len(self._buffer)
            self._buffer.clear()
            return count

        def _take(self) -> bytes:
            """取一帧数据，不足部分补静音。"""
            if len(self._buffer) >= self.frame_bytes:
                data = bytes(self._buffer[: self.frame_bytes])
                del self._buffer[: self.frame_bytes]
                return data
            data = bytes(self._buffer)
            self._buffer.clear()
            if len(data) < self.frame_bytes:
                self._silence_frames += 1
                data = data + b"\x00" * (self.frame_bytes - len(data))
            return data

        async def recv(self) -> Any:
            """按墙钟节拍产出一帧 ``av.AudioFrame``。"""
            if self._start is None:
                self._start = monotonic_s()
                self._timestamp = 0
            else:
                self._timestamp += self.samples_per_frame
                wait = self._start + self._timestamp / self.sample_rate - monotonic_s()
                if wait > 0:
                    await asyncio.sleep(wait)
            payload = self._take()
            array = np.frombuffer(payload, dtype="<i2").reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(array, format="s16", layout=self._layout)
            frame.sample_rate = self.sample_rate
            frame.pts = self._timestamp
            frame.time_base = fractions.Fraction(1, self.sample_rate)
            return frame

    _OUTBOUND_TRACK_CLS = _OutboundAudioTrack
    return _OUTBOUND_TRACK_CLS


class WebRtcTransport(BaseTransport):
    """基于 aiortc 的 WebRTC 链路（服务端为 answerer）。

    Example::

        transport = WebRtcTransport(TransportConfig(session_id=sid))
        await transport.start()
        answer = await transport.negotiate(offer_sdp)
        return answer.as_dict()
    """

    kind = TransportKind.WEBRTC

    def __init__(
        self,
        config: TransportConfig | None = None,
        *,
        ice_config: IceConfig | None = None,
        track_sample_rate: int = WEBRTC_SAMPLE_RATE,
        enable_video_track: bool = True,
        video_long_side: int = VLM_LONG_SIDE,
    ) -> None:
        """
        Args:
            config: 链路配置。
            ice_config: ICE 配置，``None`` 时从应用配置读取。
            track_sample_rate: 出栈音频轨采样率（Opus 原生 48kHz）。
            enable_video_track: 是否消费远端视频轨。
            video_long_side: 远端视频帧归一化后的长边像素。

        Raises:
            OptionalDependencyMissing: aiortc 未安装。
        """
        require_aiortc("webrtc transport")
        super().__init__(config)
        self.ice_config = ice_config or ice_config_from_settings(user_tag=self.session_id)
        self.track_sample_rate = int(track_sample_rate)
        self.enable_video_track = bool(enable_video_track)
        self.codec = DataChannelCodec()
        self.downscaler = Downscaler(long_side=video_long_side)

        self._pc: Any = None
        self._out_track: Any = None
        self._channel: Any = None
        self._consumers: set[asyncio.Task[None]] = set()
        self._out_resampler: Resampler | None = None
        self._out_resampler_rate = 0
        self._local_description: SessionDescription | None = None
        self._ice_state = "new"
        self._connection_state = "new"
        self._video_frames_encoded = 0

    # ------------------------------------------------------------------
    # 只读视图
    # ------------------------------------------------------------------

    @property
    def peer_connection(self) -> Any:
        """底层 ``RTCPeerConnection``（未 start 时为 ``None``）。"""
        return self._pc

    @property
    def local_description(self) -> SessionDescription | None:
        """本端 answer（协商完成后可用）。"""
        return self._local_description

    @property
    def ice_state(self) -> str:
        """ICE 连接状态。"""
        return self._ice_state

    @property
    def connection_state(self) -> str:
        """PeerConnection 状态。"""
        return self._connection_state

    # ------------------------------------------------------------------
    # 生命周期钩子
    # ------------------------------------------------------------------

    async def _open(self) -> None:
        """创建 PeerConnection、出栈轨与事件回调。"""
        servers = [
            RTCIceServer(
                urls=list(item.urls),
                username=item.username,
                credential=item.credential,
            )
            for item in self.ice_config.ice_servers
        ]
        self._pc = RTCPeerConnection(RTCConfiguration(iceServers=servers))
        self._out_track = _outbound_track_cls()(
            sample_rate=self.track_sample_rate,
            channels=self.config.channels,
            frame_ms=self.config.frame_ms,
        )
        self._pc.addTrack(self._out_track)
        self._wire_events()
        self._log.info(
            "webrtc_opened",
            ice_servers=len(servers),
            has_turn=self.ice_config.has_turn,
        )

    def _wire_events(self) -> None:
        """注册 aiortc 事件回调。"""
        pc = self._pc

        @pc.on("track")
        def _on_track(track: Any) -> None:
            self._log.info("webrtc_track_received", kind=track.kind)
            if track.kind == "audio":
                self._spawn(self._consume_audio(track), name="webrtc_audio")
            elif track.kind == "video" and self.enable_video_track:
                self._spawn(self._consume_video(track), name="webrtc_video")

        @pc.on("datachannel")
        def _on_datachannel(channel: Any) -> None:
            self._log.info("webrtc_datachannel_open", label=channel.label)
            self._channel = channel

            @channel.on("message")
            def _on_message(message: Any) -> None:
                self._spawn(self._on_channel_message(message), name="webrtc_dc_msg")

            @channel.on("close")
            def _on_channel_close() -> None:
                self._log.info("webrtc_datachannel_closed", label=channel.label)
                if self._channel is channel:
                    self._channel = None

        @pc.on("iceconnectionstatechange")
        def _on_ice_state() -> None:
            self._ice_state = str(pc.iceConnectionState)
            self._log.info("webrtc_ice_state", state=self._ice_state)
            if self._ice_state in ("failed", "closed"):
                try:
                    transport_errors.labels(kind=str(self.kind), reason="ice_failed").inc()
                except Exception:
                    pass
                self._spawn(self.stop(f"ice_{self._ice_state}"), name="webrtc_ice_stop")

        @pc.on("connectionstatechange")
        def _on_connection_state() -> None:
            self._connection_state = str(pc.connectionState)
            self._log.info("webrtc_connection_state", state=self._connection_state)
            if self._connection_state in ("failed", "closed"):
                self._spawn(
                    self.stop(f"pc_{self._connection_state}"), name="webrtc_pc_stop"
                )

    def _spawn(self, coro: Any, *, name: str) -> None:
        """启动一个受管的后台任务。"""
        task = asyncio.create_task(coro, name=f"{name}:{self.session_id}")
        self._consumers.add(task)
        task.add_done_callback(self._consumers.discard)

    async def _close(self, reason: str) -> None:
        """停轨、停任务、关 PeerConnection。"""
        current = asyncio.current_task()
        tasks = [task for task in self._consumers if task is not current]
        self._consumers.clear()
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        if self._out_track is not None:
            try:
                self._out_track.stop()
            except Exception:
                pass
        if self._channel is not None:
            try:
                self._channel.close()
            except Exception:
                pass
            self._channel = None
        if self._pc is not None:
            try:
                await self._pc.close()
            except Exception:
                pass
            self._pc = None

    # ------------------------------------------------------------------
    # 协商
    # ------------------------------------------------------------------

    async def negotiate(
        self,
        offer_sdp: str,
        *,
        type_: str = "offer",
        opus_bitrate_bps: int = 32_000,
        gather_timeout_ms: int = ICE_GATHER_TIMEOUT_MS,
    ) -> SessionDescription:
        """执行一次 offer/answer 协商。

        Args:
            offer_sdp: 远端 offer 的 SDP。
            type_: 描述类型，必须是 ``offer``。
            opus_bitrate_bps: 写入 answer 的 Opus 目标码率。
            gather_timeout_ms: ICE 收集超时。

        Returns:
            本端 answer。

        Raises:
            TransportClosed: 链路尚未 start。
            IceFailed: ICE 收集超时。
        """
        if self._pc is None:
            raise TransportClosed(
                "webrtc transport not started", detail={"session_id": self.session_id}
            )
        validate_sdp(offer_sdp, expect_type=type_)
        await self._pc.setRemoteDescription(
            RTCSessionDescription(sdp=offer_sdp, type=type_)
        )
        answer = await self._pc.createAnswer()
        try:
            await asyncio.wait_for(
                self._pc.setLocalDescription(answer),
                timeout=max(0.5, gather_timeout_ms / 1000.0),
            )
        except TimeoutError as exc:
            raise IceFailed(
                "ice gathering timed out",
                detail={"session_id": self.session_id, "timeout_ms": gather_timeout_ms},
            ) from exc
        sdp = self._pc.localDescription.sdp
        sdp = prefer_codec(sdp, "opus", media="audio")
        sdp = tune_opus(sdp, bitrate_bps=opus_bitrate_bps, frame_ms=self.config.frame_ms)
        self._local_description = SessionDescription(
            type=self._pc.localDescription.type, sdp=sdp
        )
        self._log.info("webrtc_answer_ready", sdp_bytes=len(sdp))
        return self._local_description

    async def add_remote_candidate(self, candidate: dict[str, Any]) -> bool:
        """追加一条 trickle ICE 候选。

        Args:
            candidate: ``{"candidate": ..., "sdpMid": ..., "sdpMLineIndex": ...}``。

        Returns:
            是否成功加入。
        """
        if self._pc is None:
            return False
        raw = str(candidate.get("candidate", "")).strip()
        if not raw:
            return False
        try:
            from aiortc.sdp import candidate_from_sdp

            parsed = candidate_from_sdp(raw.replace("candidate:", "", 1))
            parsed.sdpMid = candidate.get("sdpMid")
            index = candidate.get("sdpMLineIndex")
            parsed.sdpMLineIndex = int(index) if index is not None else None
            await self._pc.addIceCandidate(parsed)
            return True
        except Exception as exc:
            self._log.warning("webrtc_add_candidate_failed", error=str(exc))
            return False

    # ------------------------------------------------------------------
    # 上行消费
    # ------------------------------------------------------------------

    async def _consume_audio(self, track: Any) -> None:
        """把远端音频轨解成 20ms 的 :class:`AudioChunk`。"""
        slicer = FrameSlicer(
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            frame_ms=self.config.frame_ms,
        )
        resampler: Resampler | None = None
        src_rate = 0
        try:
            while True:
                frame = await track.recv()
                samples = frame.to_ndarray()
                flat = np.asarray(samples).reshape(-1)
                if flat.dtype != np.int16:
                    flat = np.clip(flat, -32768, 32767).astype(np.int16)
                mono = self._remix_int16(flat, int(getattr(frame.layout, "channels", 1)))
                rate = int(frame.sample_rate or self.track_sample_rate)
                if resampler is None or rate != src_rate:
                    src_rate = rate
                    resampler = Resampler(
                        src_rate, self.config.sample_rate, channels=self.config.channels
                    )
                payload = resampler.process(mono.tobytes())
                if not payload:
                    continue
                self.stats.on_recv(len(payload), media="audio")
                for chunk in slicer.push_bytes(payload):
                    await self._emit_audio(chunk)
        except (MediaStreamError, asyncio.CancelledError):
            return
        except Exception as exc:
            self.stats.on_error()
            self._log.warning("webrtc_audio_consume_failed", error=str(exc))
        finally:
            for chunk in slicer.flush(pad=False):
                try:
                    await self._emit_audio(chunk)
                except Exception:
                    break

    def _remix_int16(self, flat: np.ndarray, src_channels: int) -> np.ndarray:
        """把交错 int16 数据混成目标声道数。"""
        src_channels = max(1, src_channels)
        if src_channels == self.config.channels:
            return flat
        as_float = flat.astype(np.float32) / 32768.0
        remixed = pcm.remix(as_float, src_channels, self.config.channels)
        return np.clip(remixed * 32767.0, -32768, 32767).astype(np.int16)

    async def _consume_video(self, track: Any) -> None:
        """把远端视频轨降采样并编码为 :class:`~omnistream.schemas.video.VideoFrame`。"""
        encoding = VideoEncoding.JPEG if PIL_AVAILABLE else VideoEncoding.RGB24
        started = monotonic_s()
        try:
            while True:
                av_frame = await track.recv()
                elapsed_ms = int((monotonic_s() - started) * 1000)
                pixels = av_frame.to_ndarray(format="rgb24")
                raw = RawVideoFrame.from_ndarray(
                    pixels, pts_ms=elapsed_ms, source="camera"
                )
                scaled = self.downscaler.process(raw)
                video_frame = scaled.to_video_frame(encoding=encoding)
                self._video_frames_encoded += 1
                self.stats.on_recv(len(video_frame.data), media="video")
                await self._emit_video(video_frame)
        except (MediaStreamError, asyncio.CancelledError):
            return
        except Exception as exc:
            self.stats.on_error()
            self._log.warning("webrtc_video_consume_failed", error=str(exc))

    async def _on_channel_message(self, message: Any) -> None:
        """处理一条 DataChannel 消息（分片重组后再分发）。"""
        try:
            if isinstance(message, str):
                await self._handle_text(message)
                return
            payload = self.codec.decode(bytes(message))
            if payload is None:
                return
            await self._handle_binary(payload)
        except Exception as exc:
            self.stats.on_error()
            self._log.warning("webrtc_datachannel_message_failed", error=str(exc))

    # ------------------------------------------------------------------
    # 下行发送
    # ------------------------------------------------------------------

    async def send_audio(self, chunk: AudioChunk) -> int:
        """把一帧 TTS 音频推给出栈轨。

        与基类不同：WebRTC 的音频不走 OMS1 二进制帧，而是直接进 RTP 音频轨。

        Args:
            chunk: PCM16 或 PCM_F32 的音频帧。

        Returns:
            推入轨道的字节数。

        Raises:
            TransportClosed: 链路不可用。
        """
        self._ensure_open()
        if self._out_track is None:
            raise TransportClosed(
                "webrtc outbound track missing", detail={"session_id": self.session_id}
            )
        payload = self._to_track_pcm(chunk)
        written = self._out_track.push(payload)
        self.stats.on_sent(written, media="audio")
        return written

    def _to_track_pcm(self, chunk: AudioChunk) -> bytes:
        """把音频帧转成出栈轨要求的采样率与位深。"""
        if chunk.encoding not in (AudioEncoding.PCM16, AudioEncoding.PCM_F32):
            raise TransportClosed(
                f"webrtc track requires raw pcm, got {chunk.encoding}",
                detail={"session_id": self.session_id},
            )
        samples = pcm.bytes_to_float32(chunk.data, chunk.encoding)
        if chunk.channels != self.config.channels:
            samples = pcm.remix(samples, chunk.channels, self.config.channels)
        raw = pcm.float32_to_bytes(samples, AudioEncoding.PCM16)
        if chunk.sample_rate == self.track_sample_rate:
            return raw
        if self._out_resampler is None or self._out_resampler_rate != chunk.sample_rate:
            self._out_resampler_rate = int(chunk.sample_rate)
            self._out_resampler = Resampler(
                self._out_resampler_rate,
                self.track_sample_rate,
                channels=self.config.channels,
            )
        return self._out_resampler.process(raw)

    def cancel_playback(self) -> int:
        """清空出栈轨缓冲（barge-in 三路止损之一）。

        Returns:
            被丢弃的字节数。
        """
        if self._out_track is None:
            return 0
        return int(self._out_track.clear())

    async def _send_binary(self, payload: bytes, *, media: str) -> None:
        """通过 DataChannel 发送二进制帧（自动分片）。"""
        channel = self._channel
        if channel is None or getattr(channel, "readyState", "") != "open":
            self.stats.on_error()
            try:
                transport_errors.labels(kind=str(self.kind), reason="no_datachannel").inc()
            except Exception:
                pass
            raise TransportClosed(
                "webrtc datachannel not open",
                detail={"session_id": self.session_id, "media": media},
            )
        for piece in self.codec.encode(payload):
            channel.send(piece)

    async def _send_text(self, payload: bytes) -> None:
        """通过 DataChannel 发送文本消息。"""
        channel = self._channel
        if channel is None or getattr(channel, "readyState", "") != "open":
            raise TransportClosed(
                "webrtc datachannel not open", detail={"session_id": self.session_id}
            )
        channel.send(payload.decode("utf-8"))

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """在基类快照上补充 WebRTC 特有信息。"""
        snapshot = super().describe()
        snapshot["webrtc"] = {
            "ice_state": self._ice_state,
            "connection_state": self._connection_state,
            "has_datachannel": self._channel is not None,
            "has_turn": self.ice_config.has_turn,
            "track_sample_rate": self.track_sample_rate,
            "track_pending_bytes": (
                int(self._out_track.pending_bytes) if self._out_track is not None else 0
            ),
            "track_silence_frames": (
                int(self._out_track.silence_frames) if self._out_track is not None else 0
            ),
            "video_frames_encoded": self._video_frames_encoded,
            "datachannel": self.codec.snapshot(),
        }
        return snapshot

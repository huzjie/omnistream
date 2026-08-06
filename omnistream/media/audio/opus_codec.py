"""Opus 编解码封装（基于 PyAV / libopus）。

**可选依赖降级**：``av`` 未安装时本模块仍可正常 import，
:data:`OPUS_AVAILABLE` 为 ``False``，构造编解码器会抛
:class:`~omnistream.media.errors.OptionalDependencyMissing`。
调用方（egress/ingress）据此自动退回 PCM16 直通。

支持的能力：
* 可变码率（VBR）与恒定码率（CBR）
* 带内 FEC（``fec=True``，配合 ``expected_loss`` 生效）
* DTX（静音期不发包）
* 应用场景优化（``voip`` / ``audio`` / ``lowdelay``）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from omnistream.media import metrics
from omnistream.media.audio import pcm
from omnistream.media.constants import (
    AUDIO_BITRATE_DEFAULT_BPS,
    DEFAULT_FRAME_MS,
    WEBRTC_SAMPLE_RATE,
)
from omnistream.media.errors import CodecError, OptionalDependencyMissing
from omnistream.protocol.enums import AudioEncoding
from omnistream.schemas.audio import AudioChunk

log = structlog.get_logger(__name__)

__all__ = ["OPUS_AVAILABLE", "OpusConfig", "OpusDecoder", "OpusEncoder", "is_available"]

try:  # pragma: no cover - 环境相关
    import av

    OPUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    av = None  # type: ignore[assignment]
    OPUS_AVAILABLE = False

#: libopus 支持的采样率
_OPUS_RATES = (8000, 12000, 16000, 24000, 48000)


def is_available() -> bool:
    """当前环境是否可用 Opus。"""
    return OPUS_AVAILABLE


@dataclass(slots=True)
class OpusConfig:
    """Opus 编码参数。

    Attributes:
        sample_rate: 采样率，必须属于 libopus 支持集。
        channels: 声道数（1 或 2）。
        bitrate: 目标码率（bit/s）。
        frame_ms: 帧长（毫秒），实时语音推荐 20。
        fec: 是否启用带内前向纠错。
        dtx: 是否启用不连续传输（静音省流）。
        vbr: 是否可变码率。
        application: ``voip`` / ``audio`` / ``lowdelay``。
        expected_loss: 预期丢包率百分比（0~100），驱动 FEC 强度。
        complexity: 编码复杂度 0~10。
    """

    sample_rate: int = WEBRTC_SAMPLE_RATE
    channels: int = 1
    bitrate: int = AUDIO_BITRATE_DEFAULT_BPS
    frame_ms: int = DEFAULT_FRAME_MS
    fec: bool = True
    dtx: bool = False
    vbr: bool = True
    application: str = "voip"
    expected_loss: int = 10
    complexity: int = 9

    def options(self) -> dict[str, str]:
        """转成 libopus 的 AVOption 字典。"""
        return {
            "application": self.application,
            "frame_duration": str(self.frame_ms),
            "vbr": "on" if self.vbr else "off",
            "fec": "1" if self.fec else "0",
            "packet_loss": str(max(0, min(100, self.expected_loss))),
            "dtx": "1" if self.dtx else "0",
            "compression_level": str(max(0, min(10, self.complexity))),
        }

    def validate(self) -> None:
        """校验参数组合。

        Raises:
            CodecError: 采样率或声道数非法。
        """
        if self.sample_rate not in _OPUS_RATES:
            raise CodecError(
                f"opus does not support sample rate {self.sample_rate}",
                detail={"supported": list(_OPUS_RATES)},
            )
        if self.channels not in (1, 2):
            raise CodecError(f"opus supports 1 or 2 channels, got {self.channels}")


def _require_av(feature: str) -> None:
    if not OPUS_AVAILABLE:
        raise OptionalDependencyMissing(
            "av", feature, install_hint="pip install av==13.1.0"
        )


def _layout_name(channels: int) -> str:
    return "stereo" if channels == 2 else "mono"


class OpusEncoder:
    """PCM16 → Opus 包序列。

    Example::

        enc = OpusEncoder(OpusConfig(sample_rate=48000, bitrate=32000))
        for packet in enc.encode(pcm16_bytes):
            transport.send(packet)
        for packet in enc.flush():
            transport.send(packet)
    """

    def __init__(self, config: OpusConfig | None = None) -> None:
        self.config = config or OpusConfig()
        self.config.validate()
        _require_av("Opus encoding")
        self._pts = 0
        self._residual = b""
        self._samples_per_frame = int(self.config.sample_rate * self.config.frame_ms / 1000)
        self._frame_bytes = self._samples_per_frame * 2 * self.config.channels
        try:
            ctx = av.CodecContext.create("libopus", "w")  # type: ignore[union-attr]
            ctx.sample_rate = self.config.sample_rate
            ctx.format = "s16"
            ctx.layout = _layout_name(self.config.channels)
            ctx.bit_rate = int(self.config.bitrate)
            ctx.options = self.config.options()
            ctx.open()
            self._ctx: Any = ctx
        except Exception as exc:
            raise CodecError(f"failed to create libopus encoder: {exc}", cause=exc) from exc
        log.debug(
            "opus_encoder_created",
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            bitrate=self.config.bitrate,
            frame_ms=self.config.frame_ms,
        )

    @property
    def frame_bytes(self) -> int:
        """一个 Opus 帧对应的 PCM16 字节数。"""
        return self._frame_bytes

    def set_bitrate(self, bitrate: int) -> None:
        """运行时调整码率（自适应码率控制器调用）。"""
        bitrate = int(max(6000, min(510_000, bitrate)))
        if bitrate == self.config.bitrate:
            return
        self.config.bitrate = bitrate
        try:
            self._ctx.bit_rate = bitrate
        except Exception as exc:  # 某些 libav 版本不支持运行时改码率
            log.warning("opus_bitrate_update_failed", error=str(exc), bitrate=bitrate)

    def encode(self, pcm16: bytes) -> list[bytes]:
        """编码任意长度的 PCM16 字节，返回完整的 Opus 包列表。

        不足一帧的尾部会缓存到下次调用。
        """
        if not pcm16:
            return []
        buffer = self._residual + pcm16
        frames, self._residual = pcm.split_fixed(buffer, self._frame_bytes)
        packets: list[bytes] = []
        for raw in frames:
            packets.extend(self._encode_frame(raw))
        return packets

    def encode_chunk(self, chunk: AudioChunk) -> list[AudioChunk]:
        """编码一个 PCM16 chunk，返回若干 Opus chunk（pts 按帧递推）。"""
        if chunk.encoding != AudioEncoding.PCM16:
            raise CodecError(f"opus encoder expects pcm16, got {chunk.encoding}")
        base_pts = chunk.pts_ms
        out: list[AudioChunk] = []
        for index, packet in enumerate(self.encode(chunk.data)):
            out.append(
                AudioChunk(
                    pts_ms=base_pts + index * self.config.frame_ms,
                    data=packet,
                    sample_rate=self.config.sample_rate,
                    channels=self.config.channels,
                    encoding=AudioEncoding.OPUS,
                    capture_ts_ns=chunk.capture_ts_ns,
                    speaker=chunk.speaker,
                )
            )
        return out

    def _encode_frame(self, raw: bytes) -> list[bytes]:
        try:
            samples = np.frombuffer(raw, dtype="<i2")
            planar = samples.reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(  # type: ignore[union-attr]
                planar, format="s16", layout=_layout_name(self.config.channels)
            )
            frame.sample_rate = self.config.sample_rate
            frame.pts = self._pts
            frame.time_base = __import__("fractions").Fraction(1, self.config.sample_rate)
            self._pts += self._samples_per_frame
            return [bytes(p) for p in self._ctx.encode(frame)]
        except Exception as exc:
            metrics.codec_errors.labels(codec="opus", direction="encode").inc()
            raise CodecError(f"opus encode failed: {exc}", cause=exc) from exc

    def flush(self) -> list[bytes]:
        """冲刷编码器内部缓冲，返回剩余包。"""
        packets: list[bytes] = []
        if self._residual:
            padding = b"\x00" * (self._frame_bytes - len(self._residual))
            packets.extend(self._encode_frame(self._residual + padding))
            self._residual = b""
        try:
            packets.extend(bytes(p) for p in self._ctx.encode(None))
        except Exception:
            pass
        return packets

    def close(self) -> None:
        """释放底层编码器。"""
        try:
            self._ctx.close()
        except Exception:
            pass


class OpusDecoder:
    """Opus 包 → PCM16。"""

    def __init__(self, sample_rate: int = WEBRTC_SAMPLE_RATE, channels: int = 1) -> None:
        if sample_rate not in _OPUS_RATES:
            raise CodecError(
                f"opus does not support sample rate {sample_rate}",
                detail={"supported": list(_OPUS_RATES)},
            )
        _require_av("Opus decoding")
        self.sample_rate = sample_rate
        self.channels = channels
        try:
            ctx = av.CodecContext.create("libopus", "r")  # type: ignore[union-attr]
            ctx.sample_rate = sample_rate
            ctx.format = "s16"
            ctx.layout = _layout_name(channels)
            ctx.open()
            self._ctx: Any = ctx
        except Exception as exc:
            raise CodecError(f"failed to create libopus decoder: {exc}", cause=exc) from exc

    def decode(self, packet: bytes) -> bytes:
        """解码单个 Opus 包为 PCM16 字节。"""
        if not packet:
            return b""
        try:
            av_packet = av.Packet(packet)  # type: ignore[union-attr]
            out: list[bytes] = []
            for frame in self._ctx.decode(av_packet):
                arr = frame.to_ndarray()
                out.append(np.ascontiguousarray(arr.reshape(-1)).astype("<i2").tobytes())
            return b"".join(out)
        except Exception as exc:
            metrics.codec_errors.labels(codec="opus", direction="decode").inc()
            raise CodecError(f"opus decode failed: {exc}", cause=exc) from exc

    def decode_chunk(self, chunk: AudioChunk) -> AudioChunk:
        """解码一个 Opus chunk 为 PCM16 chunk。"""
        if chunk.encoding != AudioEncoding.OPUS:
            return chunk
        return AudioChunk(
            pts_ms=chunk.pts_ms,
            data=self.decode(chunk.data),
            sample_rate=self.sample_rate,
            channels=self.channels,
            encoding=AudioEncoding.PCM16,
            capture_ts_ns=chunk.capture_ts_ns,
            speaker=chunk.speaker,
        )

    def close(self) -> None:
        """释放底层解码器。"""
        try:
            self._ctx.close()
        except Exception:
            pass

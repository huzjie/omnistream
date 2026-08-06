"""视频解码：WebRTC 轨道（PyAV）与单帧图像（Pillow）。

两个层次
--------
* :class:`ImageDecoder` —— 解 JPEG/WebP 单帧，只需 Pillow。WebSocket 链路
  （TS SDK 在浏览器端已经用 Canvas 编好 JPEG）走这条路。
* :class:`VideoStreamDecoder` —— 解 H.264/VP8 码流，需要 PyAV。WebRTC 链路
  拿到的是 ``av.VideoFrame``，本模块负责 YUV→RGB 与 pts 归一。

**关键优化**：如果下游只需要把帧透传给原生支持视频的 Provider，
根本不需要解码 —— 用 :func:`should_decode` 判断，能省掉整条 CPU 开销。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from omnistream.core.clock import now_ns
from omnistream.media.errors import CodecError, OptionalDependencyMissing
from omnistream.media.metrics import codec_errors
from omnistream.media.video.frame import PIL_AVAILABLE, RawVideoFrame, decode_image
from omnistream.protocol.enums import VideoEncoding
from omnistream.schemas.video import VideoFrame

__all__ = [
    "AV_AVAILABLE",
    "DecoderStats",
    "ImageDecoder",
    "VideoStreamDecoder",
    "from_av_frame",
    "hardware_accelerators",
    "is_stream_decoding_available",
    "should_decode",
]

try:  # pragma: no cover - 环境相关
    import av

    AV_AVAILABLE = True
except ImportError:  # pragma: no cover - 环境相关
    av = None  # type: ignore[assignment]
    AV_AVAILABLE = False


def is_stream_decoding_available() -> bool:
    """码流解码（H.264/VP8）是否可用。"""
    return AV_AVAILABLE


def should_decode(*, need_pixels: bool, encoding: VideoEncoding) -> bool:
    """判断是否真的需要解码。

    Args:
        need_pixels: 下游是否需要像素（场景检测、缩放、ROI 都需要）。
        encoding: 帧的当前编码。

    Returns:
        ``True`` 表示必须解码；否则可以字节直通。
    """
    if not need_pixels:
        return False
    if encoding == VideoEncoding.RGB24:
        return True
    return PIL_AVAILABLE


def hardware_accelerators() -> list[str]:
    """探测当前 ffmpeg 构建支持的硬件加速器名称列表。

    未安装 PyAV 时返回空列表。结果仅供日志与运维诊断，
    解码路径本身不依赖它 —— 软解在 768p 抽帧场景下完全够用。
    """
    if not AV_AVAILABLE or av is None:
        return []
    candidates = ("cuda", "vaapi", "videotoolbox", "qsv", "d3d11va", "dxva2", "vdpau")
    available: list[str] = []
    for name in candidates:
        try:
            if hasattr(av, "codec") and name in getattr(av.codec, "hwdevices_available", lambda: [])():
                available.append(name)
        except Exception:  # pragma: no cover - 取决于 PyAV 版本
            continue
    if not available:
        # 老版本 PyAV 没有 hwdevices_available，退化为解码器名称探测
        for name in candidates:
            try:
                av.codec.Codec(f"h264_{name}", "r")
            except Exception:
                continue
            available.append(name)
    return available


def from_av_frame(av_frame: Any, *, pts_ms: int | None = None, source: str = "camera") -> RawVideoFrame:
    """把 ``av.VideoFrame`` 转成 :class:`RawVideoFrame`。

    Args:
        av_frame: PyAV 的视频帧对象。
        pts_ms: 显式 pts；``None`` 时用 ``av_frame.pts * time_base``。
        source: 轨道来源标记。

    Raises:
        CodecError: 像素格式转换失败。
    """
    try:
        pixels = av_frame.to_ndarray(format="rgb24")
    except Exception as exc:
        try:
            codec_errors.labels(codec="rawvideo", direction="decode").inc()
        except Exception:  # 指标绝不能影响主链路
            pass
        raise CodecError("failed to convert av.VideoFrame to rgb24", cause=exc) from exc

    if pts_ms is None:
        raw_pts = getattr(av_frame, "pts", None)
        time_base = getattr(av_frame, "time_base", None)
        if raw_pts is not None and time_base is not None:
            pts_ms = int(round(float(raw_pts) * float(time_base) * 1000.0))
        else:
            pts_ms = 0
    return RawVideoFrame(
        pixels=pixels,
        pts_ms=int(pts_ms),
        capture_ts_ns=now_ns(),
        source=source,
    )


@dataclass(slots=True)
class DecoderStats:
    """解码器滚动统计。"""

    packets: int = 0
    frames: int = 0
    errors: int = 0
    bytes_in: int = 0

    def as_dict(self) -> dict[str, float]:
        """转为可序列化字典。"""
        return {
            "packets": float(self.packets),
            "frames": float(self.frames),
            "errors": float(self.errors),
            "bytes_in": float(self.bytes_in),
        }


class ImageDecoder:
    """单帧图像解码器（JPEG / WebP / RGB24）。"""

    def __init__(self, *, tolerant: bool = True) -> None:
        """初始化解码器。

        Args:
            tolerant: 为 ``True`` 时单帧解码失败只计数不抛异常
                （视频链路丢一帧无所谓，不该拖垮会话）。
        """
        self.tolerant = bool(tolerant)
        self.stats = DecoderStats()

    def decode(self, frame: VideoFrame) -> RawVideoFrame | None:
        """解码一个传输态帧。

        Returns:
            :class:`RawVideoFrame`；``tolerant=True`` 且解码失败时返回 ``None``。

        Raises:
            OptionalDependencyMissing: 需要 Pillow 但未安装（该异常始终向上抛，
                因为它是**配置问题**而非**数据问题**）。
            CodecError: ``tolerant=False`` 且字节损坏。
        """
        self.stats.packets += 1
        self.stats.bytes_in += len(frame.data)
        try:
            pixels = decode_image(
                frame.data, frame.encoding, width=frame.width, height=frame.height
            )
        except OptionalDependencyMissing:
            raise
        except CodecError:
            self.stats.errors += 1
            try:
                codec_errors.labels(codec=str(frame.encoding), direction="decode").inc()
            except Exception:  # 指标绝不能影响主链路
                pass
            if self.tolerant:
                return None
            raise
        self.stats.frames += 1
        return RawVideoFrame(
            pixels=pixels,
            pts_ms=int(frame.pts_ms),
            capture_ts_ns=int(getattr(frame, "capture_ts_ns", now_ns())),
            keyframe_score=float(getattr(frame, "keyframe_score", 0.0)),
        )

    def snapshot(self) -> dict[str, float]:
        """导出统计快照。"""
        return self.stats.as_dict()

    def __repr__(self) -> str:
        """调试用文本表示。"""
        return f"ImageDecoder(frames={self.stats.frames}, errors={self.stats.errors})"


class VideoStreamDecoder:
    """H.264 / VP8 码流解码器（PyAV）。

    Raises:
        OptionalDependencyMissing: 构造时 PyAV 不可用。
    """

    #: 支持的码流编码。
    SUPPORTED: tuple[str, ...] = ("h264", "vp8", "vp9", "av1")

    def __init__(self, codec_name: str = "h264", *, source: str = "camera") -> None:
        """打开底层解码器。"""
        if not AV_AVAILABLE or av is None:
            raise OptionalDependencyMissing(
                "av", f"{codec_name} stream decoding", install_hint="pip install av"
            )
        normalised = codec_name.lower()
        if normalised not in self.SUPPORTED:
            raise CodecError(
                f"unsupported stream codec: {codec_name}",
                detail={"supported": list(self.SUPPORTED)},
            )
        self.codec_name = normalised
        self.source = source
        self.stats = DecoderStats()
        self._closed = False
        try:
            self._context = av.CodecContext.create(normalised, "r")
        except Exception as exc:  # pragma: no cover - 取决于 ffmpeg 构建
            raise OptionalDependencyMissing(
                f"av[{normalised}]",
                f"{normalised} stream decoding",
                install_hint="install ffmpeg build with the required decoder",
            ) from exc

    @property
    def closed(self) -> bool:
        """是否已关闭。"""
        return self._closed

    def decode(self, payload: bytes, *, pts_ms: int | None = None) -> list[RawVideoFrame]:
        """喂入一个码流包，返回解出的 0~N 帧。"""
        if self._closed:
            raise CodecError("VideoStreamDecoder is already closed")
        self.stats.packets += 1
        self.stats.bytes_in += len(payload)
        try:
            packet = av.Packet(payload)
            if pts_ms is not None:
                packet.pts = int(pts_ms)
            frames = self._context.decode(packet)
        except Exception as exc:
            self.stats.errors += 1
            try:
                codec_errors.labels(codec=self.codec_name, direction="decode").inc()
            except Exception:  # 指标绝不能影响主链路
                pass
            raise CodecError(f"{self.codec_name} decoding failed", cause=exc) from exc
        result = [from_av_frame(item, pts_ms=pts_ms, source=self.source) for item in frames]
        self.stats.frames += len(result)
        return result

    def flush(self) -> list[RawVideoFrame]:
        """冲刷解码器内部缓冲。"""
        if self._closed:
            return []
        try:
            frames = self._context.decode(None)
        except Exception:  # pragma: no cover - 冲刷失败不应致命
            return []
        result = [from_av_frame(item, source=self.source) for item in frames]
        self.stats.frames += len(result)
        return result

    def close(self) -> list[RawVideoFrame]:
        """冲刷并关闭（幂等）。"""
        if self._closed:
            return []
        tail = self.flush()
        self._closed = True
        try:
            self._context.close()
        except Exception:  # pragma: no cover - 关闭失败可忽略
            pass
        return tail

    def snapshot(self) -> dict[str, float]:
        """导出统计快照。"""
        return self.stats.as_dict()

    def __repr__(self) -> str:
        """调试用文本表示。"""
        return (
            f"VideoStreamDecoder({self.codec_name}, frames={self.stats.frames}, "
            f"errors={self.stats.errors})"
        )


def decode_to_pixels(frame: VideoFrame) -> np.ndarray:
    """便捷函数：把传输态帧解成像素数组。"""
    return decode_image(frame.data, frame.encoding, width=frame.width, height=frame.height)

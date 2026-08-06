"""视频编码：JPEG/WebP（送 VLM）与 H.264（录制）。

两条路径的取舍完全不同
----------------------
* **送 VLM** —— 单帧独立编码，追求"给定字节预算下的最高可读性"。
  :class:`ImageEncoder` 支持**目标字节数自适应质量**：先按当前质量试编，
  超预算就二分下调，低于预算太多就缓慢上调，最终稳定在预算附近。
* **录制** —— 连续帧、有帧间预测，追求压缩率。:class:`H264Encoder` 走 PyAV，
  未安装时抛 :class:`~omnistream.media.errors.OptionalDependencyMissing`，
  录制功能降级为"只存 JPEG 序列"。
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from omnistream.media.constants import (
    JPEG_QUALITY_DEFAULT,
    VIDEO_BITRATE_DEFAULT_BPS,
    WEBP_QUALITY_DEFAULT,
)
from omnistream.media.errors import CodecError, OptionalDependencyMissing
from omnistream.media.metrics import codec_errors
from omnistream.media.video.frame import RawVideoFrame, encode_image
from omnistream.protocol.enums import VideoEncoding
from omnistream.schemas.video import VideoFrame

__all__ = [
    "AV_AVAILABLE",
    "EncoderStats",
    "H264Encoder",
    "ImageEncoder",
    "encode_jpeg",
    "encode_webp",
    "is_h264_available",
]

try:  # pragma: no cover - 环境相关
    import av

    AV_AVAILABLE = True
except ImportError:  # pragma: no cover - 环境相关
    av = None  # type: ignore[assignment]
    AV_AVAILABLE = False


def is_h264_available() -> bool:
    """H.264 编码是否可用。"""
    return AV_AVAILABLE


def encode_jpeg(pixels: np.ndarray, quality: int = JPEG_QUALITY_DEFAULT) -> bytes:
    """把像素编码为 JPEG。"""
    return encode_image(pixels, VideoEncoding.JPEG, quality=quality)


def encode_webp(pixels: np.ndarray, quality: int = WEBP_QUALITY_DEFAULT) -> bytes:
    """把像素编码为 WebP。"""
    return encode_image(pixels, VideoEncoding.WEBP, quality=quality)


@dataclass(slots=True)
class EncoderStats:
    """编码器滚动统计。"""

    frames: int = 0
    bytes_out: int = 0
    retries: int = 0
    errors: int = 0
    last_quality: int = 0
    last_bytes: int = 0

    @property
    def avg_bytes(self) -> float:
        """平均每帧字节数。"""
        return self.bytes_out / self.frames if self.frames else 0.0

    def as_dict(self) -> dict[str, float]:
        """转为可序列化字典。"""
        return {
            "frames": float(self.frames),
            "bytes_out": float(self.bytes_out),
            "avg_bytes": round(self.avg_bytes, 1),
            "retries": float(self.retries),
            "errors": float(self.errors),
            "last_quality": float(self.last_quality),
            "last_bytes": float(self.last_bytes),
        }


class ImageEncoder:
    """单帧图像编码器，支持目标字节数自适应质量。"""

    #: 质量调整的上下界。
    MIN_QUALITY: int = 25
    MAX_QUALITY: int = 95

    def __init__(
        self,
        *,
        encoding: VideoEncoding = VideoEncoding.JPEG,
        quality: int = JPEG_QUALITY_DEFAULT,
        target_bytes: int = 0,
        max_retries: int = 3,
    ) -> None:
        """初始化编码器。

        Args:
            encoding: 目标编码。
            quality: 初始质量。
            target_bytes: 目标字节数；``0`` 表示不做自适应。
            max_retries: 单帧最多重试几次以逼近目标字节数。
        """
        self.encoding = encoding
        self.quality = int(
            min(self.MAX_QUALITY, max(self.MIN_QUALITY, quality))
        )
        self.target_bytes = max(0, int(target_bytes))
        self.max_retries = max(0, int(max_retries))
        self.stats = EncoderStats(last_quality=self.quality)

    def set_target_bytes(self, target_bytes: int) -> None:
        """由自适应码率控制器下发每帧字节预算。"""
        self.target_bytes = max(0, int(target_bytes))

    def set_quality(self, quality: int) -> None:
        """直接设定质量（关闭自适应时使用）。"""
        self.quality = int(min(self.MAX_QUALITY, max(self.MIN_QUALITY, quality)))

    def encode_pixels(self, pixels: np.ndarray) -> bytes:
        """编码裸像素，返回字节。"""
        try:
            payload = encode_image(pixels, self.encoding, quality=self.quality)
        except OptionalDependencyMissing:
            raise
        except CodecError:
            self.stats.errors += 1
            try:
                codec_errors.labels(codec=str(self.encoding), direction="encode").inc()
            except Exception:  # 指标绝不能影响主链路
                pass
            raise

        if self.target_bytes > 0:
            payload = self._adapt(pixels, payload)

        self.stats.frames += 1
        self.stats.bytes_out += len(payload)
        self.stats.last_bytes = len(payload)
        self.stats.last_quality = self.quality
        return payload

    def _adapt(self, pixels: np.ndarray, payload: bytes) -> bytes:
        """按目标字节数微调质量并重编。"""
        low = self.MIN_QUALITY
        high = self.MAX_QUALITY
        best = payload
        for _ in range(self.max_retries):
            size = len(best)
            if size > self.target_bytes:
                high = self.quality - 1
            elif size < int(self.target_bytes * 0.7):
                low = self.quality + 1
            else:
                break
            if low > high:
                break
            self.quality = (low + high) // 2
            self.stats.retries += 1
            best = encode_image(pixels, self.encoding, quality=self.quality)
        return best

    def encode(self, frame: RawVideoFrame) -> VideoFrame:
        """编码一帧，返回传输态 :class:`VideoFrame`。"""
        payload = self.encode_pixels(frame.pixels)
        return VideoFrame(
            pts_ms=frame.pts_ms,
            width=frame.width,
            height=frame.height,
            data=payload,
            encoding=self.encoding,
            keyframe_score=round(float(frame.keyframe_score), 4),
            capture_ts_ns=frame.capture_ts_ns,
        )

    def snapshot(self) -> dict[str, float]:
        """导出统计快照。"""
        data = self.stats.as_dict()
        data["quality"] = float(self.quality)
        data["target_bytes"] = float(self.target_bytes)
        return data

    def __repr__(self) -> str:
        """调试用文本表示。"""
        return (
            f"ImageEncoder({self.encoding}, q={self.quality}, "
            f"target={self.target_bytes}B, frames={self.stats.frames})"
        )


class H264Encoder:
    """基于 PyAV 的 H.264 编码器（录制用）。

    Raises:
        OptionalDependencyMissing: 构造时 PyAV 不可用。
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: float = 15.0,
        bitrate: int = VIDEO_BITRATE_DEFAULT_BPS,
        gop_size: int = 30,
        preset: str = "veryfast",
    ) -> None:
        """初始化编码器并打开底层 codec context。"""
        if not AV_AVAILABLE or av is None:
            raise OptionalDependencyMissing(
                "av", "H.264 video encoding", install_hint="pip install av"
            )
        # H.264 要求偶数尺寸
        self.width = max(2, int(width) - (int(width) % 2))
        self.height = max(2, int(height) - (int(height) % 2))
        self.fps = max(1.0, float(fps))
        self.bitrate = max(10_000, int(bitrate))
        self.gop_size = max(1, int(gop_size))
        self.frames_encoded = 0
        self.bytes_out = 0
        self._closed = False

        try:
            context = av.CodecContext.create("libx264", "w")
        except Exception as exc:  # pragma: no cover - 取决于 ffmpeg 构建
            raise OptionalDependencyMissing(
                "av[libx264]",
                "H.264 video encoding",
                install_hint="install ffmpeg build with libx264",
            ) from exc
        context.width = self.width
        context.height = self.height
        context.pix_fmt = "yuv420p"
        context.bit_rate = self.bitrate
        context.gop_size = self.gop_size
        context.time_base = Fraction(1, 1000)
        context.options = {"preset": preset, "tune": "zerolatency"}
        self._context = context

    @property
    def closed(self) -> bool:
        """是否已关闭。"""
        return self._closed

    def set_bitrate(self, bitrate: int) -> None:
        """运行时调整目标码率。"""
        self.bitrate = max(10_000, int(bitrate))
        try:
            self._context.bit_rate = self.bitrate
        except Exception:  # pragma: no cover - 部分构建不支持热改
            return None
        return None

    def encode(self, frame: RawVideoFrame) -> list[bytes]:
        """编码一帧，返回 0~N 个 H.264 包（编码器有前瞻延迟）。"""
        if self._closed:
            raise CodecError("H264Encoder is already closed")
        pixels = frame.pixels
        if pixels.shape[1] != self.width or pixels.shape[0] != self.height:
            from omnistream.media.video.downscaler import resize_pixels

            pixels, _backend = resize_pixels(pixels, self.width, self.height)
        av_frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(pixels), format="rgb24")
        av_frame = av_frame.reformat(format="yuv420p")
        av_frame.pts = int(frame.pts_ms)
        av_frame.time_base = Fraction(1, 1000)
        try:
            packets = self._context.encode(av_frame)
        except Exception as exc:
            try:
                codec_errors.labels(codec="h264", direction="encode").inc()
            except Exception:  # 指标绝不能影响主链路
                pass
            raise CodecError("H.264 encoding failed", cause=exc) from exc
        payloads = [bytes(packet) for packet in packets]
        self.frames_encoded += 1
        self.bytes_out += sum(len(item) for item in payloads)
        return payloads

    def flush(self) -> list[bytes]:
        """冲刷编码器内部缓冲。"""
        if self._closed:
            return []
        try:
            packets = self._context.encode(None)
        except Exception:  # pragma: no cover - 冲刷失败不应致命
            return []
        payloads = [bytes(packet) for packet in packets]
        self.bytes_out += sum(len(item) for item in payloads)
        return payloads

    def close(self) -> list[bytes]:
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
        return {
            "frames_encoded": float(self.frames_encoded),
            "bytes_out": float(self.bytes_out),
            "bitrate": float(self.bitrate),
            "width": float(self.width),
            "height": float(self.height),
        }

    def __repr__(self) -> str:
        """调试用文本表示。"""
        return (
            f"H264Encoder({self.width}x{self.height}@{self.fps:.0f}fps, "
            f"{self.bitrate // 1000}kbps, frames={self.frames_encoded})"
        )

"""原始视频帧容器与图像编解码适配。

分层
----
* :class:`~omnistream.schemas.video.VideoFrame`（WS1 契约）—— **传输态**，
  持有已编码字节（JPEG/WebP/RGB24），跨进程、跨网络传递。
* :class:`RawVideoFrame`（本模块）—— **处理态**，持有 ``(H, W, 3) uint8`` RGB
  ndarray，供场景检测、缩放、ROI 裁剪等算法直接操作。

两者的转换是显式的：``RawVideoFrame.from_video_frame()`` /
``RawVideoFrame.to_video_frame()``。管线里**尽量晚解码、尽量早编码**，
在不需要像素的路径上（例如 Provider 原生支持视频输入）JPEG 字节可以全程直通，
从而彻底避免 PIL 依赖。

降级
----
``Pillow`` 未安装时 :data:`PIL_AVAILABLE` 为 ``False``：RGB24 原始互转仍然可用
（纯 numpy），JPEG/WebP 编解码则抛
:class:`~omnistream.media.errors.OptionalDependencyMissing`，由上层选择直通或跳过。
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np

from omnistream.core.clock import now_ns
from omnistream.media.constants import JPEG_QUALITY_DEFAULT, WEBP_QUALITY_DEFAULT
from omnistream.media.errors import CodecError, OptionalDependencyMissing
from omnistream.protocol.enums import VideoEncoding
from omnistream.schemas.video import VideoFrame

__all__ = [
    "PIL_AVAILABLE",
    "RawVideoFrame",
    "decode_image",
    "encode_image",
    "is_available",
    "require_pil",
    "to_gray",
]

try:  # pragma: no cover - 环境相关
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - 环境相关
    Image = None  # type: ignore[assignment]
    PIL_AVAILABLE = False


#: 需要 PIL 才能处理的编码。
_PIL_ENCODINGS: frozenset[VideoEncoding] = frozenset({VideoEncoding.JPEG, VideoEncoding.WEBP})

#: BT.601 亮度权重，用于快速灰度化。
_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def is_available() -> bool:
    """Pillow 是否可用（决定 JPEG/WebP 能力）。"""
    return PIL_AVAILABLE


def require_pil(feature: str) -> None:
    """断言 Pillow 可用，否则抛可预期的降级信号。

    Raises:
        OptionalDependencyMissing: Pillow 未安装。
    """
    if not PIL_AVAILABLE:
        raise OptionalDependencyMissing("Pillow", feature, install_hint="pip install Pillow")


def _normalise_pixels(arr: np.ndarray) -> np.ndarray:
    """把任意合法输入规整为 ``(H, W, 3) uint8`` RGB 数组。"""
    pixels = np.asarray(arr)
    if pixels.ndim == 2:
        pixels = np.repeat(pixels[:, :, None], 3, axis=2)
    if pixels.ndim != 3:
        raise CodecError(
            "video frame must be 2D grayscale or 3D HxWxC array",
            detail={"ndim": int(pixels.ndim)},
        )
    if pixels.shape[2] == 4:
        pixels = pixels[:, :, :3]
    elif pixels.shape[2] == 1:
        pixels = np.repeat(pixels, 3, axis=2)
    elif pixels.shape[2] != 3:
        raise CodecError(
            "video frame must have 1, 3 or 4 channels",
            detail={"channels": int(pixels.shape[2])},
        )
    if pixels.dtype != np.uint8:
        if np.issubdtype(pixels.dtype, np.floating):
            pixels = np.clip(pixels * 255.0 if pixels.max() <= 1.0 else pixels, 0, 255)
        else:
            pixels = np.clip(pixels, 0, 255)
        pixels = pixels.astype(np.uint8)
    return np.ascontiguousarray(pixels)


def to_gray(pixels: np.ndarray) -> np.ndarray:
    """RGB → 灰度 ``float32``（范围 0~255），供直方图/边缘统计使用。"""
    rgb = _normalise_pixels(pixels).astype(np.float32)
    return rgb @ _LUMA_WEIGHTS


def encode_image(
    pixels: np.ndarray,
    encoding: VideoEncoding = VideoEncoding.JPEG,
    *,
    quality: int | None = None,
) -> bytes:
    """把像素数组编码为字节。

    Args:
        pixels: ``(H, W, 3)`` RGB 数组。
        encoding: 目标编码。
        quality: JPEG/WebP 质量（1~100）；``None`` 用默认值。

    Returns:
        编码后的字节。

    Raises:
        OptionalDependencyMissing: 目标编码需要 Pillow 但未安装。
        CodecError: 编码失败。
    """
    rgb = _normalise_pixels(pixels)
    if encoding == VideoEncoding.RGB24:
        return rgb.tobytes()
    require_pil(f"{encoding} encoding")
    if encoding == VideoEncoding.JPEG:
        params = {"format": "JPEG", "quality": int(quality or JPEG_QUALITY_DEFAULT), "optimize": True}
    elif encoding == VideoEncoding.WEBP:
        params = {"format": "WEBP", "quality": int(quality or WEBP_QUALITY_DEFAULT), "method": 4}
    else:
        raise CodecError(f"unsupported video encoding: {encoding}")
    buffer = io.BytesIO()
    try:
        Image.fromarray(rgb, mode="RGB").save(buffer, **params)
    except OSError as exc:  # pragma: no cover - 依赖 Pillow 内部错误
        raise CodecError(f"failed to encode {encoding} frame", cause=exc) from exc
    return buffer.getvalue()


def decode_image(
    data: bytes,
    encoding: VideoEncoding = VideoEncoding.JPEG,
    *,
    width: int = 0,
    height: int = 0,
) -> np.ndarray:
    """把编码字节解成 ``(H, W, 3) uint8`` RGB 数组。

    Args:
        data: 编码字节。
        encoding: 源编码。
        width: RGB24 时必须提供。
        height: RGB24 时必须提供。

    Raises:
        OptionalDependencyMissing: 源编码需要 Pillow 但未安装。
        CodecError: 字节长度与声明尺寸不符，或解码失败。
    """
    if encoding == VideoEncoding.RGB24:
        if width <= 0 or height <= 0:
            raise CodecError("rgb24 decoding requires explicit width/height")
        expected = width * height * 3
        if len(data) < expected:
            raise CodecError(
                "rgb24 payload shorter than declared geometry",
                detail={"expected": expected, "actual": len(data)},
            )
        return np.frombuffer(data[:expected], dtype=np.uint8).reshape(height, width, 3).copy()
    require_pil(f"{encoding} decoding")
    try:
        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except OSError as exc:
        raise CodecError(f"failed to decode {encoding} frame", cause=exc) from exc


@dataclass(slots=True)
class RawVideoFrame:
    """处理态视频帧：持有可直接运算的 RGB 像素。"""

    pixels: np.ndarray = field(repr=False)
    pts_ms: int = 0
    capture_ts_ns: int = field(default_factory=now_ns)
    keyframe_score: float = 0.0
    source: str = "camera"

    def __post_init__(self) -> None:
        """规整像素布局，保证后续算法可以无分支地使用。"""
        self.pixels = _normalise_pixels(self.pixels)

    # ------------------------------------------------------------------
    # 几何属性
    # ------------------------------------------------------------------

    @property
    def height(self) -> int:
        """帧高度（像素）。"""
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        """帧宽度（像素）。"""
        return int(self.pixels.shape[1])

    @property
    def long_side(self) -> int:
        """长边像素数。"""
        return max(self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        """宽高比。"""
        return self.width / self.height if self.height else 0.0

    @property
    def nbytes(self) -> int:
        """原始像素占用字节数。"""
        return int(self.pixels.nbytes)

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------

    @classmethod
    def from_ndarray(
        cls,
        arr: np.ndarray,
        *,
        pts_ms: int = 0,
        capture_ts_ns: int | None = None,
        keyframe_score: float = 0.0,
        source: str = "camera",
    ) -> "RawVideoFrame":
        """由 ndarray 构造。"""
        return cls(
            pixels=arr,
            pts_ms=int(pts_ms),
            capture_ts_ns=capture_ts_ns if capture_ts_ns is not None else now_ns(),
            keyframe_score=float(keyframe_score),
            source=source,
        )

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        encoding: VideoEncoding = VideoEncoding.JPEG,
        pts_ms: int = 0,
        width: int = 0,
        height: int = 0,
        source: str = "camera",
    ) -> "RawVideoFrame":
        """由编码字节构造（会立即解码）。"""
        pixels = decode_image(data, encoding, width=width, height=height)
        return cls(pixels=pixels, pts_ms=int(pts_ms), source=source)

    @classmethod
    def from_video_frame(cls, frame: VideoFrame) -> "RawVideoFrame":
        """由传输态 :class:`VideoFrame` 解码而来。"""
        pixels = decode_image(
            frame.data, frame.encoding, width=frame.width, height=frame.height
        )
        return cls(
            pixels=pixels,
            pts_ms=int(frame.pts_ms),
            capture_ts_ns=int(getattr(frame, "capture_ts_ns", now_ns())),
            keyframe_score=float(getattr(frame, "keyframe_score", 0.0)),
        )

    @classmethod
    def blank(
        cls,
        width: int,
        height: int,
        *,
        pts_ms: int = 0,
        color: tuple[int, int, int] = (0, 0, 0),
    ) -> "RawVideoFrame":
        """生成纯色帧（占位、补洞、测试用）。"""
        canvas = np.zeros((max(1, height), max(1, width), 3), dtype=np.uint8)
        canvas[:, :] = np.array(color, dtype=np.uint8)
        return cls(pixels=canvas, pts_ms=int(pts_ms))

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------

    def to_bytes(
        self,
        encoding: VideoEncoding = VideoEncoding.JPEG,
        *,
        quality: int | None = None,
    ) -> bytes:
        """编码为字节。"""
        return encode_image(self.pixels, encoding, quality=quality)

    def to_video_frame(
        self,
        *,
        encoding: VideoEncoding = VideoEncoding.JPEG,
        quality: int | None = None,
    ) -> VideoFrame:
        """转换为传输态 :class:`VideoFrame`。"""
        return VideoFrame(
            pts_ms=self.pts_ms,
            width=self.width,
            height=self.height,
            data=self.to_bytes(encoding, quality=quality),
            encoding=encoding,
            keyframe_score=round(float(self.keyframe_score), 4),
            capture_ts_ns=self.capture_ts_ns,
        )

    def to_gray(self) -> np.ndarray:
        """灰度视图（``float32``，0~255）。"""
        return to_gray(self.pixels)

    # ------------------------------------------------------------------
    # 变换
    # ------------------------------------------------------------------

    def with_pixels(self, pixels: np.ndarray) -> "RawVideoFrame":
        """替换像素并保留全部元数据。"""
        return RawVideoFrame(
            pixels=pixels,
            pts_ms=self.pts_ms,
            capture_ts_ns=self.capture_ts_ns,
            keyframe_score=self.keyframe_score,
            source=self.source,
        )

    def crop(self, left: int, top: int, right: int, bottom: int) -> "RawVideoFrame":
        """按像素坐标裁剪（越界自动夹取）。"""
        x0 = max(0, min(int(left), self.width - 1))
        y0 = max(0, min(int(top), self.height - 1))
        x1 = max(x0 + 1, min(int(right), self.width))
        y1 = max(y0 + 1, min(int(bottom), self.height))
        return self.with_pixels(self.pixels[y0:y1, x0:x1])

    def copy(self) -> "RawVideoFrame":
        """深拷贝像素。"""
        return self.with_pixels(self.pixels.copy())

    def describe(self) -> dict[str, float | str]:
        """导出元信息字典（日志与控制台展示用）。"""
        return {
            "pts_ms": float(self.pts_ms),
            "width": float(self.width),
            "height": float(self.height),
            "keyframe_score": round(float(self.keyframe_score), 4),
            "nbytes": float(self.nbytes),
            "source": self.source,
        }

    def __repr__(self) -> str:
        """调试用文本表示。"""
        return (
            f"RawVideoFrame({self.width}x{self.height}, pts={self.pts_ms}ms, "
            f"ks={self.keyframe_score:.2f}, src={self.source})"
        )

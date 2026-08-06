"""视频子包：帧容器、编解码、场景检测、自适应抽帧、缩放与屏幕共享。

依赖分层
--------
* **零依赖**（numpy only）：``frame`` 的 RGB24 互转、``scene_change``、``sampler``、
  ``downscaler`` 的 numpy 后端、``frame_buffer``、``roi``、``timeline``、``screenshare``。
* **需要 Pillow**：JPEG/WebP 编解码、``thumbnail`` 的雪碧图输出。
* **需要 PyAV**：``decoder.VideoStreamDecoder``（H.264/VP8 码流）、
  ``encoder.H264Encoder``（录制）。

缺失可选依赖时相应能力抛
:class:`~omnistream.media.errors.OptionalDependencyMissing`，其余功能不受影响，
本包始终可以被 import。
"""
from __future__ import annotations

from omnistream.media.video.decoder import (
    AV_AVAILABLE,
    DecoderStats,
    ImageDecoder,
    VideoStreamDecoder,
    decode_to_pixels,
    from_av_frame,
    hardware_accelerators,
    is_stream_decoding_available,
    should_decode,
)
from omnistream.media.video.downscaler import (
    Downscaler,
    ScaleResult,
    fit_long_side,
    nearest_ladder_size,
    resize_pixels,
)
from omnistream.media.video.encoder import (
    EncoderStats,
    H264Encoder,
    ImageEncoder,
    encode_jpeg,
    encode_webp,
    is_h264_available,
)
from omnistream.media.video.frame import (
    PIL_AVAILABLE,
    RawVideoFrame,
    decode_image,
    encode_image,
    require_pil,
    to_gray,
)
from omnistream.media.video.frame_buffer import FrameBufferStats, VideoFrameBuffer
from omnistream.media.video.roi import (
    RegionOfInterest,
    RoiDetector,
    crop_to_aspect,
    expand_box,
)
from omnistream.media.video.sampler import (
    AdaptiveFrameSampler,
    SamplerDecision,
    SamplerStats,
)
from omnistream.media.video.scene_change import (
    SceneChangeDetector,
    SceneSignal,
    edge_density,
    histogram_distance,
)
from omnistream.media.video.screenshare import (
    ScreenShareProcessor,
    ScreenShareResult,
    looks_like_screenshare,
    text_density_map,
)
from omnistream.media.video.thumbnail import (
    SpriteSheet,
    ThumbnailBuilder,
    ThumbnailTile,
    make_thumbnail,
)
from omnistream.media.video.timeline import VideoTimeline, VideoTimelineStats

__all__ = [
    "AV_AVAILABLE",
    "PIL_AVAILABLE",
    "AdaptiveFrameSampler",
    "DecoderStats",
    "Downscaler",
    "EncoderStats",
    "FrameBufferStats",
    "H264Encoder",
    "ImageDecoder",
    "ImageEncoder",
    "RawVideoFrame",
    "RegionOfInterest",
    "RoiDetector",
    "SamplerDecision",
    "SamplerStats",
    "ScaleResult",
    "SceneChangeDetector",
    "SceneSignal",
    "ScreenShareProcessor",
    "ScreenShareResult",
    "SpriteSheet",
    "ThumbnailBuilder",
    "ThumbnailTile",
    "VideoFrameBuffer",
    "VideoStreamDecoder",
    "VideoTimeline",
    "VideoTimelineStats",
    "crop_to_aspect",
    "decode_image",
    "decode_to_pixels",
    "edge_density",
    "encode_image",
    "encode_jpeg",
    "encode_webp",
    "expand_box",
    "fit_long_side",
    "from_av_frame",
    "hardware_accelerators",
    "histogram_distance",
    "is_h264_available",
    "is_stream_decoding_available",
    "looks_like_screenshare",
    "make_thumbnail",
    "nearest_ladder_size",
    "require_pil",
    "resize_pixels",
    "should_decode",
    "text_density_map",
    "to_gray",
]

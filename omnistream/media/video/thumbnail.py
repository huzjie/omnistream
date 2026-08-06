"""会话缩略图与关键帧雪碧图。

用途
----
控制台的会话时间轴需要在一条横轴上展示"这段对话里画面发生了什么"。
逐帧存 JPEG 太重，改为：

* **缩略图** —— 单帧长边 240px，用于时间轴上的悬停预览。
* **雪碧图（sprite sheet）** —— 把 N 张缩略图拼成一张大图 + 一份坐标清单，
  前端只需一次请求就能拿到整条时间轴的预览，避免 N 次小文件请求。

拼接用纯 numpy 完成（就是数组切片赋值），只有最后一步编码需要 Pillow。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from omnistream.media.constants import (
    JPEG_QUALITY_THUMBNAIL,
    SPRITE_COLUMNS,
    THUMBNAIL_LONG_SIDE,
)
from omnistream.media.video.downscaler import fit_long_side, resize_pixels
from omnistream.media.video.frame import RawVideoFrame, encode_image
from omnistream.protocol.enums import VideoEncoding

__all__ = ["SpriteSheet", "ThumbnailBuilder", "ThumbnailTile", "make_thumbnail"]


def make_thumbnail(
    frame: RawVideoFrame,
    *,
    long_side: int = THUMBNAIL_LONG_SIDE,
) -> np.ndarray:
    """把一帧缩成缩略图像素（不编码）。"""
    new_w, new_h = fit_long_side(frame.width, frame.height, long_side)
    pixels, _backend = resize_pixels(frame.pixels, new_w, new_h)
    return pixels


@dataclass(slots=True, frozen=True)
class ThumbnailTile:
    """雪碧图中的一格。"""

    index: int
    pts_ms: int
    left: int
    top: int
    width: int
    height: int
    keyframe_score: float

    def as_dict(self) -> dict[str, float]:
        """转为可序列化字典（前端按此定位裁剪区域）。"""
        return {
            "index": float(self.index),
            "pts_ms": float(self.pts_ms),
            "left": float(self.left),
            "top": float(self.top),
            "width": float(self.width),
            "height": float(self.height),
            "keyframe_score": round(self.keyframe_score, 4),
        }


@dataclass(slots=True)
class SpriteSheet:
    """雪碧图结果。"""

    data: bytes = field(repr=False, default=b"")
    width: int = 0
    height: int = 0
    columns: int = 0
    rows: int = 0
    tile_width: int = 0
    tile_height: int = 0
    encoding: VideoEncoding = VideoEncoding.JPEG
    tiles: list[ThumbnailTile] = field(default_factory=list)

    @property
    def count(self) -> int:
        """格子数量。"""
        return len(self.tiles)

    def as_dict(self) -> dict[str, object]:
        """转为可序列化字典（不含图像字节）。"""
        return {
            "width": self.width,
            "height": self.height,
            "columns": self.columns,
            "rows": self.rows,
            "tile_width": self.tile_width,
            "tile_height": self.tile_height,
            "encoding": str(self.encoding),
            "bytes": len(self.data),
            "tiles": [tile.as_dict() for tile in self.tiles],
        }


class ThumbnailBuilder:
    """收集关键帧并生成缩略图 / 雪碧图。

    典型用法::

        builder = ThumbnailBuilder()
        for frame in accepted_frames:
            builder.add(frame)
        sheet = builder.build_sprite()
    """

    def __init__(
        self,
        *,
        long_side: int = THUMBNAIL_LONG_SIDE,
        columns: int = SPRITE_COLUMNS,
        max_tiles: int = 40,
        quality: int = JPEG_QUALITY_THUMBNAIL,
        encoding: VideoEncoding = VideoEncoding.JPEG,
    ) -> None:
        """初始化构造器。

        Args:
            long_side: 单格缩略图长边。
            columns: 雪碧图列数。
            max_tiles: 最多保留多少格；超出时按 ``keyframe_score`` 淘汰最不重要的。
            quality: 编码质量。
            encoding: 输出编码。
        """
        self.long_side = max(32, int(long_side))
        self.columns = max(1, int(columns))
        self.max_tiles = max(1, int(max_tiles))
        self.quality = int(quality)
        self.encoding = encoding
        self._tiles: list[tuple[int, float, np.ndarray]] = []

    @property
    def count(self) -> int:
        """当前收集的格子数。"""
        return len(self._tiles)

    def add(self, frame: RawVideoFrame) -> None:
        """加入一帧（自动缩放，超额时淘汰分数最低者）。"""
        pixels = make_thumbnail(frame, long_side=self.long_side)
        self._tiles.append((int(frame.pts_ms), float(frame.keyframe_score), pixels))
        if len(self._tiles) > self.max_tiles:
            # 保留首尾（时间覆盖）+ 分数最高者
            head = self._tiles[0]
            tail = self._tiles[-1]
            middle = sorted(self._tiles[1:-1], key=lambda item: item[1], reverse=True)
            keep = middle[: self.max_tiles - 2]
            self._tiles = sorted([head, *keep, tail], key=lambda item: item[0])

    def encode_single(self, frame: RawVideoFrame) -> bytes:
        """直接把一帧编码为缩略图字节。"""
        return encode_image(
            make_thumbnail(frame, long_side=self.long_side),
            self.encoding,
            quality=self.quality,
        )

    def build_sprite(self) -> SpriteSheet:
        """把已收集的格子拼成雪碧图。

        所有格子统一缩放到"最常见尺寸"，保证网格对齐；空位补黑。
        """
        if not self._tiles:
            return SpriteSheet(encoding=self.encoding)

        ordered = sorted(self._tiles, key=lambda item: item[0])
        # 以第一格的尺寸为基准（同一路视频宽高比一致）
        tile_h, tile_w = ordered[0][2].shape[0], ordered[0][2].shape[1]
        columns = min(self.columns, len(ordered))
        rows = (len(ordered) + columns - 1) // columns
        canvas = np.zeros((rows * tile_h, columns * tile_w, 3), dtype=np.uint8)

        tiles: list[ThumbnailTile] = []
        for index, (pts_ms, score, pixels) in enumerate(ordered):
            if pixels.shape[0] != tile_h or pixels.shape[1] != tile_w:
                pixels, _backend = resize_pixels(pixels, tile_w, tile_h)
            row, col = divmod(index, columns)
            top, left = row * tile_h, col * tile_w
            canvas[top : top + tile_h, left : left + tile_w] = pixels
            tiles.append(
                ThumbnailTile(
                    index=index,
                    pts_ms=pts_ms,
                    left=left,
                    top=top,
                    width=tile_w,
                    height=tile_h,
                    keyframe_score=score,
                )
            )

        return SpriteSheet(
            data=encode_image(canvas, self.encoding, quality=self.quality),
            width=int(canvas.shape[1]),
            height=int(canvas.shape[0]),
            columns=columns,
            rows=rows,
            tile_width=tile_w,
            tile_height=tile_h,
            encoding=self.encoding,
            tiles=tiles,
        )

    def clear(self) -> int:
        """清空已收集的格子，返回清除数量。"""
        count = len(self._tiles)
        self._tiles.clear()
        return count

    def __repr__(self) -> str:
        """调试用文本表示。"""
        return (
            f"ThumbnailBuilder(tiles={len(self._tiles)}/{self.max_tiles}, "
            f"long_side={self.long_side})"
        )

"""等比缩放到 VLM 友好分辨率。

设计
----
* **优先用 Pillow 的 LANCZOS**（质量最好）；Pillow 缺失时回退到**纯 numpy 的
  area/box 平均降采样**，质量足够 VLM 使用且完全无依赖。
* 只缩不放：源分辨率已经小于目标长边时原样返回，避免无意义的插值噪声。
* 保留原始几何元数据（``origin_width`` / ``origin_height`` / ``scale``），
  下游做 ROI 坐标反算时需要。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omnistream.media.constants import RESOLUTION_LADDER, VLM_LONG_SIDE
from omnistream.media.video.frame import PIL_AVAILABLE, RawVideoFrame

if PIL_AVAILABLE:  # pragma: no cover - 环境相关
    from PIL import Image
else:  # pragma: no cover - 环境相关
    Image = None  # type: ignore[assignment]

__all__ = [
    "Downscaler",
    "ScaleResult",
    "fit_long_side",
    "nearest_ladder_size",
    "resize_pixels",
]


@dataclass(slots=True, frozen=True)
class ScaleResult:
    """缩放结果的几何元数据。"""

    origin_width: int
    origin_height: int
    width: int
    height: int
    scale: float
    backend: str

    @property
    def changed(self) -> bool:
        """是否真的发生了尺寸变化。"""
        return (self.width, self.height) != (self.origin_width, self.origin_height)

    def as_dict(self) -> dict[str, float | str]:
        """转为可序列化字典。"""
        return {
            "origin_width": float(self.origin_width),
            "origin_height": float(self.origin_height),
            "width": float(self.width),
            "height": float(self.height),
            "scale": round(self.scale, 4),
            "backend": self.backend,
        }


def fit_long_side(width: int, height: int, long_side: int, *, even: bool = True) -> tuple[int, int]:
    """按长边等比计算目标尺寸。

    Args:
        width: 源宽度。
        height: 源高度。
        long_side: 目标长边像素。
        even: 是否把结果对齐到偶数（H.264/JPEG 子采样友好）。

    Returns:
        ``(目标宽, 目标高)``；源已足够小则原样返回。
    """
    if width <= 0 or height <= 0:
        return max(1, width), max(1, height)
    current = max(width, height)
    if current <= long_side:
        return width, height
    ratio = long_side / float(current)
    new_w = max(1, int(round(width * ratio)))
    new_h = max(1, int(round(height * ratio)))
    if even:
        new_w = max(2, new_w - (new_w % 2))
        new_h = max(2, new_h - (new_h % 2))
    return new_w, new_h


def nearest_ladder_size(width: int, height: int) -> tuple[int, int]:
    """把任意尺寸吸附到 :data:`RESOLUTION_LADDER` 中最接近且不超过的档位。"""
    long_side = max(width, height)
    chosen = RESOLUTION_LADDER[0]
    for candidate in RESOLUTION_LADDER:
        if max(candidate) <= long_side:
            chosen = candidate
        else:
            break
    return fit_long_side(width, height, max(chosen))


def _resize_numpy(pixels: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """纯 numpy 缩放。

    降采样走 box 平均（抗锯齿），放大走最近邻（VLM 场景几乎不会用到放大）。
    """
    height, width = pixels.shape[0], pixels.shape[1]
    if new_w == width and new_h == height:
        return pixels

    if new_w <= width and new_h <= height:
        # box 平均：对每个目标像素，取源图中对应区间的均值
        row_edges = np.linspace(0, height, new_h + 1).astype(np.int64)
        col_edges = np.linspace(0, width, new_w + 1).astype(np.int64)
        source = pixels.astype(np.float32)
        # 先按行聚合，再按列聚合，避免 O(new_h * new_w) 的 python 循环
        row_acc = np.add.reduceat(source, row_edges[:-1], axis=0)
        row_counts = np.diff(row_edges).astype(np.float32).reshape(-1, 1, 1)
        row_mean = row_acc / np.maximum(row_counts, 1.0)
        col_acc = np.add.reduceat(row_mean, col_edges[:-1], axis=1)
        col_counts = np.diff(col_edges).astype(np.float32).reshape(1, -1, 1)
        result = col_acc / np.maximum(col_counts, 1.0)
        return np.clip(result, 0, 255).astype(np.uint8)

    row_index = np.clip((np.arange(new_h) * height // max(1, new_h)), 0, height - 1)
    col_index = np.clip((np.arange(new_w) * width // max(1, new_w)), 0, width - 1)
    return pixels[row_index][:, col_index]


def resize_pixels(pixels: np.ndarray, new_w: int, new_h: int) -> tuple[np.ndarray, str]:
    """缩放像素数组，返回 ``(结果, 使用的后端名)``。"""
    target_w = max(1, int(new_w))
    target_h = max(1, int(new_h))
    if pixels.shape[1] == target_w and pixels.shape[0] == target_h:
        return pixels, "passthrough"
    if PIL_AVAILABLE and Image is not None:
        image = Image.fromarray(pixels, mode="RGB")
        resized = image.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
        return np.asarray(resized, dtype=np.uint8), "pillow"
    return _resize_numpy(pixels, target_w, target_h), "numpy"


class Downscaler:
    """把视频帧统一缩放到 VLM 友好分辨率的有状态封装。

    保存最近一次的缩放元数据，方便 ROI 模块把裁剪坐标映射回原始画面。
    """

    def __init__(
        self,
        *,
        long_side: int = VLM_LONG_SIDE,
        even_dimensions: bool = True,
        snap_to_ladder: bool = False,
    ) -> None:
        """初始化缩放器。

        Args:
            long_side: 目标长边。
            even_dimensions: 结果尺寸是否对齐偶数。
            snap_to_ladder: 是否把结果吸附到标准分辨率档位。
        """
        self.long_side = max(16, int(long_side))
        self.even_dimensions = bool(even_dimensions)
        self.snap_to_ladder = bool(snap_to_ladder)
        self.last_result: ScaleResult | None = None
        self.frames_scaled: int = 0
        self.frames_passthrough: int = 0

    def set_long_side(self, long_side: int) -> None:
        """运行时调整目标长边（自适应策略降档时调用）。"""
        self.long_side = max(16, int(long_side))

    def target_size(self, width: int, height: int) -> tuple[int, int]:
        """计算给定源尺寸的目标尺寸。"""
        new_w, new_h = fit_long_side(width, height, self.long_side, even=self.even_dimensions)
        if self.snap_to_ladder:
            new_w, new_h = nearest_ladder_size(new_w, new_h)
        return new_w, new_h

    def scale_pixels(self, pixels: np.ndarray) -> tuple[np.ndarray, ScaleResult]:
        """缩放裸像素并返回元数据。"""
        origin_h, origin_w = int(pixels.shape[0]), int(pixels.shape[1])
        new_w, new_h = self.target_size(origin_w, origin_h)
        result_pixels, backend = resize_pixels(pixels, new_w, new_h)
        scale = new_w / float(origin_w) if origin_w else 1.0
        meta = ScaleResult(
            origin_width=origin_w,
            origin_height=origin_h,
            width=int(result_pixels.shape[1]),
            height=int(result_pixels.shape[0]),
            scale=scale,
            backend=backend,
        )
        self.last_result = meta
        if meta.changed:
            self.frames_scaled += 1
        else:
            self.frames_passthrough += 1
        return result_pixels, meta

    def process(self, frame: RawVideoFrame) -> RawVideoFrame:
        """缩放一帧，元数据保持不变。"""
        pixels, _meta = self.scale_pixels(frame.pixels)
        if pixels is frame.pixels:
            return frame
        return frame.with_pixels(pixels)

    def snapshot(self) -> dict[str, float | str]:
        """导出统计快照。"""
        data: dict[str, float | str] = {
            "long_side": float(self.long_side),
            "frames_scaled": float(self.frames_scaled),
            "frames_passthrough": float(self.frames_passthrough),
            "backend": "pillow" if PIL_AVAILABLE else "numpy",
        }
        if self.last_result is not None:
            data.update(self.last_result.as_dict())
        return data

    def __repr__(self) -> str:
        """调试用文本表示。"""
        backend = "pillow" if PIL_AVAILABLE else "numpy"
        return f"Downscaler(long_side={self.long_side}, backend={backend})"

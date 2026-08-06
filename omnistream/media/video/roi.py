"""感兴趣区域（ROI）检测与裁剪。

目标不是做人脸识别 —— 那是 WS4 感知层的事。这里解决的是**带宽与 token 的分配**：
在一张 1080p 的画面里，真正承载信息的往往只有中间一小块（说话的人、屏幕上的
代码区域）。把这块裁出来再送 VLM，同样的 token 预算能换到高得多的有效分辨率。

检测手段（全部 numpy，无依赖）：

* **运动显著性** —— 与上一帧的逐块绝对差，捕捉"人在动"。
* **边缘显著性** —— 梯度幅值密度，捕捉"这里有内容"（文字区域尤其明显）。
* **中心先验** —— 二维高斯权重，摄像头场景下主体几乎总在中间偏上。

三者加权得到显著性热力图，再用积分图找出**得分密度最高的矩形窗口**。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omnistream.media.video.frame import RawVideoFrame, to_gray

__all__ = ["RegionOfInterest", "RoiDetector", "crop_to_aspect", "expand_box"]

#: 显著性分析的网格粒度。
_GRID: int = 16

_EPS: float = 1e-9


@dataclass(slots=True, frozen=True)
class RegionOfInterest:
    """一个归一化的感兴趣区域。"""

    left: int
    top: int
    right: int
    bottom: int
    score: float
    source: str = "saliency"

    @property
    def width(self) -> int:
        """区域宽度。"""
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        """区域高度。"""
        return max(0, self.bottom - self.top)

    @property
    def area(self) -> int:
        """区域面积。"""
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        """区域中心点坐标。"""
        return (self.left + self.width // 2, self.top + self.height // 2)

    def scaled(self, factor: float) -> "RegionOfInterest":
        """按比例映射坐标（缩放前后互转用）。"""
        return RegionOfInterest(
            left=int(round(self.left * factor)),
            top=int(round(self.top * factor)),
            right=int(round(self.right * factor)),
            bottom=int(round(self.bottom * factor)),
            score=self.score,
            source=self.source,
        )

    def as_dict(self) -> dict[str, float | str]:
        """转为可序列化字典。"""
        return {
            "left": float(self.left),
            "top": float(self.top),
            "right": float(self.right),
            "bottom": float(self.bottom),
            "width": float(self.width),
            "height": float(self.height),
            "score": round(self.score, 4),
            "source": self.source,
        }


def expand_box(
    box: RegionOfInterest,
    *,
    margin: float,
    width: int,
    height: int,
) -> RegionOfInterest:
    """按比例向外扩张区域并夹取到画面边界。"""
    pad_x = int(round(box.width * max(0.0, margin)))
    pad_y = int(round(box.height * max(0.0, margin)))
    return RegionOfInterest(
        left=max(0, box.left - pad_x),
        top=max(0, box.top - pad_y),
        right=min(width, box.right + pad_x),
        bottom=min(height, box.bottom + pad_y),
        score=box.score,
        source=box.source,
    )


def crop_to_aspect(
    box: RegionOfInterest,
    *,
    aspect: float,
    width: int,
    height: int,
) -> RegionOfInterest:
    """把区域调整到目标宽高比（尽量保持中心不变）。"""
    if aspect <= 0 or box.area == 0:
        return box
    center_x, center_y = box.center
    current = box.width / float(max(1, box.height))
    if current < aspect:
        new_w = min(width, int(round(box.height * aspect)))
        new_h = box.height
    else:
        new_w = box.width
        new_h = min(height, int(round(box.width / aspect)))
    left = max(0, min(width - new_w, center_x - new_w // 2))
    top = max(0, min(height - new_h, center_y - new_h // 2))
    return RegionOfInterest(
        left=left,
        top=top,
        right=left + new_w,
        bottom=top + new_h,
        score=box.score,
        source=box.source,
    )


def _saliency_map(gray: np.ndarray, prev_gray: np.ndarray | None, grid: int) -> np.ndarray:
    """计算 ``grid x grid`` 的显著性热力图（归一化到 0~1）。"""
    height, width = gray.shape
    rows = max(1, height // grid)
    cols = max(1, width // grid)

    def blockify(source: np.ndarray) -> np.ndarray:
        usable = source[: rows * grid, : cols * grid]
        if usable.size == 0:
            return np.zeros((grid, grid), dtype=np.float32)
        return usable.reshape(grid, rows, grid, cols).mean(axis=(1, 3)).astype(np.float32)

    # 边缘显著性
    dx = np.zeros_like(gray, dtype=np.float32)
    dy = np.zeros_like(gray, dtype=np.float32)
    dx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    dy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    edges = blockify(np.sqrt(dx * dx + dy * dy))
    edges = edges / (float(edges.max()) + _EPS)

    # 运动显著性
    if prev_gray is not None and prev_gray.shape == gray.shape:
        motion = blockify(np.abs(gray - prev_gray))
        motion = motion / (float(motion.max()) + _EPS)
    else:
        motion = np.zeros((grid, grid), dtype=np.float32)

    # 中心先验（垂直方向重心略微上移，符合人像构图）
    axis_x = np.linspace(-1.0, 1.0, grid, dtype=np.float32)
    axis_y = np.linspace(-1.0, 1.0, grid, dtype=np.float32) + 0.15
    prior = np.exp(-(axis_y[:, None] ** 2 + axis_x[None, :] ** 2) / 0.6).astype(np.float32)

    combined = 0.45 * motion + 0.35 * edges + 0.20 * prior
    return combined / (float(combined.max()) + _EPS)


def _best_window(heat: np.ndarray, win_h: int, win_w: int) -> tuple[int, int, float]:
    """用积分图找出得分密度最高的窗口，返回 ``(row, col, score)``。"""
    grid_h, grid_w = heat.shape
    win_h = max(1, min(win_h, grid_h))
    win_w = max(1, min(win_w, grid_w))
    integral = np.zeros((grid_h + 1, grid_w + 1), dtype=np.float32)
    integral[1:, 1:] = np.cumsum(np.cumsum(heat, axis=0), axis=1)

    best_score = -1.0
    best_row = 0
    best_col = 0
    for row in range(grid_h - win_h + 1):
        for col in range(grid_w - win_w + 1):
            total = (
                integral[row + win_h, col + win_w]
                - integral[row, col + win_w]
                - integral[row + win_h, col]
                + integral[row, col]
            )
            if total > best_score:
                best_score = float(total)
                best_row, best_col = row, col
    density = best_score / float(win_h * win_w)
    return best_row, best_col, density


class RoiDetector:
    """基于显著性热力图的 ROI 检测器。"""

    def __init__(
        self,
        *,
        grid: int = _GRID,
        coverage: float = 0.6,
        margin: float = 0.08,
        min_score: float = 0.35,
        smoothing: float = 0.5,
    ) -> None:
        """初始化检测器。

        Args:
            grid: 显著性网格粒度。
            coverage: ROI 窗口占画面的边长比例（0.6 表示裁出约 36% 面积）。
            margin: 找到窗口后向外扩张的比例。
            min_score: 低于该显著性密度则认为"没有明显主体"，返回全画面。
            smoothing: ROI 框的时间平滑系数，避免裁剪框跳动。
        """
        self.grid = max(4, int(grid))
        self.coverage = min(1.0, max(0.2, float(coverage)))
        self.margin = max(0.0, float(margin))
        self.min_score = float(min_score)
        self.smoothing = min(0.95, max(0.0, float(smoothing)))
        self._prev_gray: np.ndarray | None = None
        self._prev_box: tuple[float, float, float, float] | None = None
        self.detections: int = 0
        self.fallbacks: int = 0

    def detect(self, frame: RawVideoFrame) -> RegionOfInterest:
        """检测一帧的 ROI。

        画面没有明显主体时返回覆盖全图的区域（``source="full"``），
        调用方据此决定是否裁剪。
        """
        gray = frame.to_gray()
        heat = _saliency_map(gray, self._prev_gray, self.grid)
        self._prev_gray = gray

        win = max(1, int(round(self.grid * self.coverage)))
        row, col, density = _best_window(heat, win, win)

        if density < self.min_score:
            self.fallbacks += 1
            self._prev_box = None
            return RegionOfInterest(
                left=0,
                top=0,
                right=frame.width,
                bottom=frame.height,
                score=density,
                source="full",
            )

        cell_h = frame.height / float(self.grid)
        cell_w = frame.width / float(self.grid)
        raw_box = (
            col * cell_w,
            row * cell_h,
            (col + win) * cell_w,
            (row + win) * cell_h,
        )
        if self._prev_box is not None and self.smoothing > 0.0:
            alpha = self.smoothing
            raw_box = tuple(  # type: ignore[assignment]
                alpha * old + (1.0 - alpha) * new
                for old, new in zip(self._prev_box, raw_box, strict=True)
            )
        self._prev_box = raw_box  # type: ignore[assignment]
        self.detections += 1

        box = RegionOfInterest(
            left=int(round(raw_box[0])),
            top=int(round(raw_box[1])),
            right=int(round(raw_box[2])),
            bottom=int(round(raw_box[3])),
            score=density,
            source="saliency",
        )
        return expand_box(box, margin=self.margin, width=frame.width, height=frame.height)

    def crop(
        self,
        frame: RawVideoFrame,
        *,
        keep_aspect: bool = True,
    ) -> tuple[RawVideoFrame, RegionOfInterest]:
        """检测并裁剪，返回 ``(裁剪后的帧, ROI)``。

        ROI 判定为"全画面"时原样返回，不做无意义的拷贝。
        """
        box = self.detect(frame)
        if box.source == "full":
            return frame, box
        if keep_aspect and frame.height:
            box = crop_to_aspect(
                box,
                aspect=frame.width / float(frame.height),
                width=frame.width,
                height=frame.height,
            )
        return frame.crop(box.left, box.top, box.right, box.bottom), box

    def reset(self) -> None:
        """清空历史状态。"""
        self._prev_gray = None
        self._prev_box = None
        self.detections = 0
        self.fallbacks = 0

    def snapshot(self) -> dict[str, float]:
        """导出统计快照。"""
        return {
            "detections": float(self.detections),
            "fallbacks": float(self.fallbacks),
            "coverage": round(self.coverage, 3),
            "grid": float(self.grid),
        }

    def __repr__(self) -> str:
        """调试用文本表示。"""
        return (
            f"RoiDetector(grid={self.grid}, coverage={self.coverage:.2f}, "
            f"hits={self.detections}, fallbacks={self.fallbacks})"
        )

"""场景变化检测：直方图差 + 边缘密度差 → ``keyframe_score``。

算法
----
对每一帧计算三个互补的信号，加权合成 ``0.0 ~ 1.0`` 的变化分数：

1. **灰度直方图卡方距离**（权重 0.45）—— 对整体光照/构图变化最敏感，
   对轻微抖动不敏感。
2. **边缘密度差**（权重 0.35）—— Sobel 幅值的网格化密度图，捕捉"东西被拿走/
   放上来"这种结构性变化，不受色温漂移影响。
3. **网格均值绝对差**（权重 0.20）—— 8x8 缩略图的 MAE，兜底捕捉局部大块位移。

三者都做归一化，因此对分辨率不敏感。全部用 numpy 实现，**无任何可选依赖**。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omnistream.media.constants import (
    SCENE_CHANGE_BOOST_THRESHOLD,
    SCENE_CHANGE_IDLE_THRESHOLD,
)
from omnistream.media.video.frame import RawVideoFrame, to_gray

__all__ = ["SceneChangeDetector", "SceneSignal", "histogram_distance", "edge_density"]

#: 直方图 bin 数。
_HIST_BINS: int = 48

#: 网格化尺寸（边缘密度与均值差都用这个粒度）。
_GRID: int = 8

_EPS: float = 1e-9


@dataclass(slots=True, frozen=True)
class SceneSignal:
    """单帧的场景变化分析结果。"""

    score: float
    hist_distance: float
    edge_distance: float
    grid_distance: float
    brightness: float
    is_scene_change: bool
    is_idle: bool

    def as_dict(self) -> dict[str, float]:
        """转为可序列化字典。"""
        return {
            "score": round(self.score, 4),
            "hist_distance": round(self.hist_distance, 4),
            "edge_distance": round(self.edge_distance, 4),
            "grid_distance": round(self.grid_distance, 4),
            "brightness": round(self.brightness, 2),
            "is_scene_change": 1.0 if self.is_scene_change else 0.0,
            "is_idle": 1.0 if self.is_idle else 0.0,
        }


def _downgrid(gray: np.ndarray, grid: int = _GRID) -> np.ndarray:
    """把灰度图降采样到 ``grid x grid`` 的均值网格。"""
    height, width = gray.shape
    rows = max(1, height // grid)
    cols = max(1, width // grid)
    usable = gray[: rows * grid, : cols * grid]
    if usable.size == 0:
        return np.zeros((grid, grid), dtype=np.float32)
    return usable.reshape(grid, rows, grid, cols).mean(axis=(1, 3)).astype(np.float32)


def _histogram(gray: np.ndarray, bins: int = _HIST_BINS) -> np.ndarray:
    """归一化灰度直方图。"""
    hist, _ = np.histogram(gray, bins=bins, range=(0.0, 255.0))
    total = float(hist.sum())
    if total <= 0.0:
        return np.zeros(bins, dtype=np.float32)
    return (hist / total).astype(np.float32)


def histogram_distance(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """两个归一化直方图的卡方距离，结果归一到 ``0 ~ 1``。"""
    numerator = np.square(hist_a - hist_b)
    denominator = hist_a + hist_b + _EPS
    chi = float(0.5 * np.sum(numerator / denominator))
    return float(min(1.0, max(0.0, chi)))


def edge_density(gray: np.ndarray, grid: int = _GRID) -> np.ndarray:
    """Sobel 边缘幅值的网格密度图，归一到 ``0 ~ 1``。"""
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return np.zeros((grid, grid), dtype=np.float32)
    # 中心差分近似 Sobel，避免引入 scipy
    dx = np.zeros_like(gray, dtype=np.float32)
    dy = np.zeros_like(gray, dtype=np.float32)
    dx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    dy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    magnitude = np.sqrt(dx * dx + dy * dy)
    # 510 = 2 * 255，中心差分的理论上限
    normalised = np.clip(magnitude / 510.0, 0.0, 1.0)
    return _downgrid(normalised * 255.0, grid) / 255.0


class SceneChangeDetector:
    """有状态的场景变化检测器（逐帧喂入，内部保留上一帧特征）。

    典型用法::

        detector = SceneChangeDetector()
        signal = detector.observe(frame)
        if signal.is_scene_change:
            sampler.boost()
    """

    def __init__(
        self,
        *,
        boost_threshold: float = SCENE_CHANGE_BOOST_THRESHOLD,
        idle_threshold: float = SCENE_CHANGE_IDLE_THRESHOLD,
        hist_weight: float = 0.45,
        edge_weight: float = 0.35,
        grid_weight: float = 0.20,
        smoothing: float = 0.25,
    ) -> None:
        """初始化检测器。

        Args:
            boost_threshold: 高于该分数判定为场景切换。
            idle_threshold: 低于该分数判定为静止画面。
            hist_weight: 直方图信号权重。
            edge_weight: 边缘密度信号权重。
            grid_weight: 网格均值信号权重。
            smoothing: 分数的一阶平滑系数（0 表示不平滑）。
        """
        total = hist_weight + edge_weight + grid_weight
        if total <= 0:
            raise ValueError("scene change weights must sum to a positive value")
        self.boost_threshold = float(boost_threshold)
        self.idle_threshold = float(idle_threshold)
        self.hist_weight = float(hist_weight) / total
        self.edge_weight = float(edge_weight) / total
        self.grid_weight = float(grid_weight) / total
        self.smoothing = min(0.95, max(0.0, float(smoothing)))

        self._prev_hist: np.ndarray | None = None
        self._prev_edges: np.ndarray | None = None
        self._prev_grid: np.ndarray | None = None
        self._smoothed: float = 0.0
        self.frames_seen: int = 0
        self.scene_changes: int = 0
        self.last_signal: SceneSignal | None = None

    def observe_gray(self, gray: np.ndarray) -> SceneSignal:
        """直接对灰度图做检测（内部路径，也可被复用于缩略图）。"""
        hist = _histogram(gray)
        edges = edge_density(gray)
        grid = _downgrid(gray) / 255.0
        brightness = float(np.mean(gray))

        if self._prev_hist is None or self._prev_edges is None or self._prev_grid is None:
            # 首帧：定义为一次场景切换，保证一定会被采样
            self._prev_hist, self._prev_edges, self._prev_grid = hist, edges, grid
            self._smoothed = 1.0
            self.frames_seen += 1
            self.scene_changes += 1
            signal = SceneSignal(
                score=1.0,
                hist_distance=1.0,
                edge_distance=1.0,
                grid_distance=1.0,
                brightness=brightness,
                is_scene_change=True,
                is_idle=False,
            )
            self.last_signal = signal
            return signal

        hist_d = histogram_distance(self._prev_hist, hist)
        edge_d = float(np.mean(np.abs(self._prev_edges - edges)))
        grid_d = float(np.mean(np.abs(self._prev_grid - grid)))
        # 边缘与网格差的动态范围远小于 1，做经验放大后再夹取
        edge_d = min(1.0, edge_d * 6.0)
        grid_d = min(1.0, grid_d * 4.0)

        raw = self.hist_weight * hist_d + self.edge_weight * edge_d + self.grid_weight * grid_d
        if self.smoothing > 0.0:
            self._smoothed = self.smoothing * self._smoothed + (1.0 - self.smoothing) * raw
        else:
            self._smoothed = raw
        score = float(min(1.0, max(0.0, self._smoothed)))

        self._prev_hist, self._prev_edges, self._prev_grid = hist, edges, grid
        self.frames_seen += 1

        is_change = score >= self.boost_threshold
        if is_change:
            self.scene_changes += 1
        signal = SceneSignal(
            score=score,
            hist_distance=hist_d,
            edge_distance=edge_d,
            grid_distance=grid_d,
            brightness=brightness,
            is_scene_change=is_change,
            is_idle=score <= self.idle_threshold,
        )
        self.last_signal = signal
        return signal

    def observe(self, frame: RawVideoFrame) -> SceneSignal:
        """对一帧做检测，并把分数写回 ``frame.keyframe_score``。"""
        signal = self.observe_gray(frame.to_gray())
        frame.keyframe_score = signal.score
        return signal

    def observe_pixels(self, pixels: np.ndarray) -> SceneSignal:
        """对裸像素数组做检测。"""
        return self.observe_gray(to_gray(pixels))

    def reset(self) -> None:
        """清空历史特征（换轨道 / 重协商后调用）。"""
        self._prev_hist = None
        self._prev_edges = None
        self._prev_grid = None
        self._smoothed = 0.0
        self.frames_seen = 0
        self.scene_changes = 0
        self.last_signal = None

    def snapshot(self) -> dict[str, float]:
        """导出统计快照。"""
        data: dict[str, float] = {
            "frames_seen": float(self.frames_seen),
            "scene_changes": float(self.scene_changes),
            "smoothed_score": round(self._smoothed, 4),
        }
        if self.last_signal is not None:
            data.update(self.last_signal.as_dict())
        return data

    def __repr__(self) -> str:
        """调试用文本表示。"""
        return (
            f"SceneChangeDetector(frames={self.frames_seen}, "
            f"changes={self.scene_changes}, score={self._smoothed:.3f})"
        )

"""屏幕共享轨道的特化处理。

摄像头与屏幕共享是两种完全不同的信号
------------------------------------

|            | 摄像头            | 屏幕共享                  |
|------------|-------------------|---------------------------|
| 变化频率   | 持续微动          | 长时间静止 + 突然整屏切换 |
| 关键信息   | 人、物体、动作    | **文字**、图表、代码      |
| 分辨率诉求 | 768px 长边足够    | 文字要可读，需 1280px     |
| 帧率诉求   | 0.5fps            | 0.25fps 足够              |

用同一套参数处理两者必然二选一地糟糕：按摄像头的 768px 缩放，屏幕上的代码就糊了；
按屏幕的 0.25fps 采样，摄像头里的动作就丢了。

:class:`ScreenShareProcessor` 因此做三件事：

1. **保分辨率**：长边 :data:`SCREENSHARE_LONG_SIDE`，且优先裁剪而非缩放。
2. **降帧率**：基础 :data:`SCREENSHARE_FPS_DEFAULT`，但对整屏切换极其敏感。
3. **文字区域优先**：用边缘密度找出文字最密集的区域，超分辨率时优先保留它。

另外提供 :func:`looks_like_screenshare` 做自动判别 —— 屏幕内容的
**颜色饱和度低、直方图尖锐、边缘方向以水平/垂直为主**，与自然图像区分度很高。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omnistream.media.constants import (
    SCREENSHARE_FPS_DEFAULT,
    SCREENSHARE_LONG_SIDE,
    VLM_LONG_SIDE,
)
from omnistream.media.video.downscaler import Downscaler
from omnistream.media.video.frame import RawVideoFrame
from omnistream.media.video.roi import RegionOfInterest, crop_to_aspect
from omnistream.media.video.sampler import AdaptiveFrameSampler, SamplerDecision
from omnistream.media.video.scene_change import SceneChangeDetector, SceneSignal

__all__ = [
    "ScreenShareProcessor",
    "ScreenShareResult",
    "looks_like_screenshare",
    "text_density_map",
]

_EPS: float = 1e-9


def looks_like_screenshare(frame: RawVideoFrame) -> float:
    """判断一帧像屏幕内容的程度，返回 ``0.0 ~ 1.0`` 的置信度。

    三个判据：

    * **低饱和度占比** —— 屏幕上大片纯色背景（白/黑/灰）。
    * **直方图集中度** —— 少数几个灰度级占据绝大部分像素。
    * **轴对齐边缘占比** —— 水平/垂直边缘远多于斜向边缘（窗口、表格、文本行）。
    """
    pixels = frame.pixels.astype(np.float32)
    # 抽样以控制开销：屏幕分辨率通常很大
    step = max(1, min(pixels.shape[0], pixels.shape[1]) // 240)
    sample = pixels[::step, ::step, :]
    if sample.size == 0:
        return 0.0

    channel_max = sample.max(axis=2)
    channel_min = sample.min(axis=2)
    saturation = (channel_max - channel_min) / (channel_max + _EPS)
    low_sat_ratio = float(np.mean(saturation < 0.12))

    gray = sample.mean(axis=2)
    hist, _ = np.histogram(gray, bins=32, range=(0.0, 255.0))
    total = float(hist.sum()) + _EPS
    top_bins = np.sort(hist)[-4:].sum() / total
    concentration = float(top_bins)

    dx = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
    dy = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
    diag = (
        np.abs(gray[1:, 1:] - gray[:-1, :-1]).mean()
        if gray.shape[0] > 1 and gray.shape[1] > 1
        else 0.0
    )
    axis_ratio = float((dx + dy) / (2.0 * (diag + _EPS)))
    axis_score = float(min(1.0, max(0.0, (axis_ratio - 0.8) / 0.6)))

    score = 0.40 * low_sat_ratio + 0.35 * concentration + 0.25 * axis_score
    return float(min(1.0, max(0.0, score)))


def text_density_map(frame: RawVideoFrame, grid: int = 12) -> np.ndarray:
    """网格化文字密度图，值越大表示该格文字/细节越密集。

    文字的特征是**高频且轴对齐**：用水平方向的二阶差分强度近似，
    比通用边缘检测更能把文字与图片边框区分开。
    """
    gray = frame.to_gray()
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return np.zeros((grid, grid), dtype=np.float32)
    second = np.zeros_like(gray, dtype=np.float32)
    second[:, 1:-1] = np.abs(gray[:, 2:] - 2.0 * gray[:, 1:-1] + gray[:, :-2])
    rows = max(1, gray.shape[0] // grid)
    cols = max(1, gray.shape[1] // grid)
    usable = second[: rows * grid, : cols * grid]
    density = usable.reshape(grid, rows, grid, cols).mean(axis=(1, 3)).astype(np.float32)
    return density / (float(density.max()) + _EPS)


@dataclass(slots=True)
class ScreenShareResult:
    """屏幕共享帧的处理结果。"""

    frame: RawVideoFrame | None
    accepted: bool
    reason: str
    scene: SceneSignal | None
    decision: SamplerDecision | None
    text_region: RegionOfInterest | None
    screenshare_confidence: float

    def as_dict(self) -> dict[str, object]:
        """转为可序列化字典。"""
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "screenshare_confidence": round(self.screenshare_confidence, 4),
            "scene": self.scene.as_dict() if self.scene else None,
            "decision": self.decision.as_dict() if self.decision else None,
            "text_region": self.text_region.as_dict() if self.text_region else None,
        }


class ScreenShareProcessor:
    """屏幕共享轨道处理器：低帧率 + 高分辨率 + 文字区域优先。"""

    def __init__(
        self,
        *,
        long_side: int = SCREENSHARE_LONG_SIDE,
        fallback_long_side: int = VLM_LONG_SIDE,
        base_fps: float = SCREENSHARE_FPS_DEFAULT,
        boost_fps: float = 1.5,
        change_threshold: float = 0.30,
        text_focus: bool = True,
        text_coverage: float = 0.7,
    ) -> None:
        """初始化处理器。

        Args:
            long_side: 屏幕内容的目标长边（保证文字可读）。
            fallback_long_side: 被判定为非屏幕内容时的目标长边。
            base_fps: 基础采样帧率。
            boost_fps: 内容切换时的提采样帧率。
            change_threshold: 场景切换判定阈值（比摄像头更低，文字变化幅度小）。
            text_focus: 是否启用文字区域优先裁剪。
            text_coverage: 文字区域窗口占画面的边长比例。
        """
        self.long_side = max(320, int(long_side))
        self.fallback_long_side = max(160, int(fallback_long_side))
        self.text_focus = bool(text_focus)
        self.text_coverage = min(1.0, max(0.3, float(text_coverage)))

        self._detector = SceneChangeDetector(
            boost_threshold=float(change_threshold),
            idle_threshold=0.02,
            # 屏幕内容的直方图变化最能反映"翻页/切窗口"
            hist_weight=0.55,
            edge_weight=0.30,
            grid_weight=0.15,
            smoothing=0.1,
        )
        self._sampler = AdaptiveFrameSampler(
            base_fps=float(base_fps),
            min_fps=0.05,
            max_fps=max(float(boost_fps), float(base_fps)),
            boost_fps=float(boost_fps),
            boost_threshold=float(change_threshold),
            idle_threshold=0.02,
            adapt_rate=0.15,
        )
        self._downscaler = Downscaler(long_side=self.long_side)
        self.frames_seen: int = 0
        self.frames_accepted: int = 0
        self.last_confidence: float = 0.0

    # ------------------------------------------------------------------
    # 外部控制
    # ------------------------------------------------------------------

    @property
    def sampler(self) -> AdaptiveFrameSampler:
        """内部抽帧器（供自适应策略直接调参）。"""
        return self._sampler

    def set_load(self, load: float) -> None:
        """设置下游负载。"""
        self._sampler.set_load(load)

    def set_long_side(self, long_side: int) -> None:
        """调整目标长边（拥塞降档时调用）。"""
        self.long_side = max(160, int(long_side))
        self._downscaler.set_long_side(self.long_side)

    # ------------------------------------------------------------------
    # 处理
    # ------------------------------------------------------------------

    def find_text_region(self, frame: RawVideoFrame) -> RegionOfInterest | None:
        """定位文字最密集的矩形区域。

        Returns:
            :class:`RegionOfInterest`；文字分布均匀（整屏都是文字）时返回 ``None``，
            表示不该裁剪。
        """
        density = text_density_map(frame)
        grid = density.shape[0]
        win = max(1, int(round(grid * self.text_coverage)))
        if win >= grid:
            return None

        integral = np.zeros((grid + 1, grid + 1), dtype=np.float32)
        integral[1:, 1:] = np.cumsum(np.cumsum(density, axis=0), axis=1)
        best = -1.0
        best_rc = (0, 0)
        for row in range(grid - win + 1):
            for col in range(grid - win + 1):
                total = (
                    integral[row + win, col + win]
                    - integral[row, col + win]
                    - integral[row + win, col]
                    + integral[row, col]
                )
                if total > best:
                    best = float(total)
                    best_rc = (row, col)

        window_mean = best / float(win * win)
        overall_mean = float(density.mean())
        # 局部密度没有明显高于全局时，说明文字铺满全屏，裁剪反而丢信息
        if overall_mean <= _EPS or window_mean < overall_mean * 1.35:
            return None

        cell_h = frame.height / float(grid)
        cell_w = frame.width / float(grid)
        row, col = best_rc
        box = RegionOfInterest(
            left=int(round(col * cell_w)),
            top=int(round(row * cell_h)),
            right=int(round((col + win) * cell_w)),
            bottom=int(round((row + win) * cell_h)),
            score=window_mean,
            source="text_density",
        )
        if frame.height:
            box = crop_to_aspect(
                box,
                aspect=frame.width / float(frame.height),
                width=frame.width,
                height=frame.height,
            )
        return box

    def process(self, frame: RawVideoFrame) -> ScreenShareResult:
        """处理一帧屏幕共享画面。

        流程：判别 → 场景检测 → 采样决策 → 文字区域裁剪 → 缩放。
        被采样器拒绝的帧会提前返回，不做任何昂贵运算。
        """
        self.frames_seen += 1
        frame.source = "screen"
        confidence = looks_like_screenshare(frame)
        self.last_confidence = confidence

        scene = self._detector.observe(frame)
        decision = self._sampler.decide(frame, score=scene.score)
        if not decision.accepted:
            return ScreenShareResult(
                frame=None,
                accepted=False,
                reason=decision.reason,
                scene=scene,
                decision=decision,
                text_region=None,
                screenshare_confidence=confidence,
            )

        # 置信度低说明其实是摄像头画面被误路由过来了，退回摄像头参数
        target_long_side = self.long_side if confidence >= 0.45 else self.fallback_long_side
        self._downscaler.set_long_side(target_long_side)

        region: RegionOfInterest | None = None
        working = frame
        if self.text_focus and confidence >= 0.45 and frame.long_side > target_long_side:
            region = self.find_text_region(frame)
            if region is not None and region.area > 0:
                working = frame.crop(region.left, region.top, region.right, region.bottom)

        result_frame = self._downscaler.process(working)
        result_frame.keyframe_score = scene.score
        result_frame.source = "screen"
        self.frames_accepted += 1
        return ScreenShareResult(
            frame=result_frame,
            accepted=True,
            reason=decision.reason,
            scene=scene,
            decision=decision,
            text_region=region,
            screenshare_confidence=confidence,
        )

    def reset(self) -> None:
        """重置全部内部状态。"""
        self._detector.reset()
        self._sampler.reset()
        self.frames_seen = 0
        self.frames_accepted = 0
        self.last_confidence = 0.0

    def snapshot(self) -> dict[str, object]:
        """导出统计快照。"""
        return {
            "frames_seen": float(self.frames_seen),
            "frames_accepted": float(self.frames_accepted),
            "long_side": float(self.long_side),
            "confidence": round(self.last_confidence, 4),
            "sampler": self._sampler.snapshot(),
            "scene": self._detector.snapshot(),
        }

    def __repr__(self) -> str:
        """调试用文本表示。"""
        return (
            f"ScreenShareProcessor(long_side={self.long_side}, "
            f"accepted={self.frames_accepted}/{self.frames_seen}, "
            f"confidence={self.last_confidence:.2f})"
        )

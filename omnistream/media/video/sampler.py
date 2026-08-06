"""自适应视频抽帧器。

问题
----
摄像头是 30fps，VLM 每秒只能吃 0.5 帧。固定间隔抽帧会犯两个错误：
画面静止时白白浪费预算；画面剧变时又漏掉关键瞬间。

策略
----
:class:`AdaptiveFrameSampler` 用**基础帧率 + 事件驱动提采样**的双层结构：

* **基础帧率**随画面活跃度缓慢升降（静止 → 逼近 :data:`VIDEO_FPS_MIN`，
  活跃 → 逼近配置上限）。
* **场景切换**（``keyframe_score`` 越过阈值）立刻把有效帧率提到 ``boost_fps``，
  持续 :data:`SCENE_BOOST_WINDOW_MS` 后线性衰减回基础帧率。
* **下游背压**（VLM 队列深度 / 预算耗尽）通过 :meth:`set_load` 直接压制有效帧率，
  优先级高于前两者 —— 宁可少看几帧，也不能把 VLM 打爆。

判定完全基于 pts（会话内相对时基），不依赖墙钟，因此离线回放与实时链路结果一致。
"""
from __future__ import annotations

from dataclasses import dataclass

from omnistream.media.constants import (
    SCENE_BOOST_WINDOW_MS,
    SCENE_CHANGE_BOOST_THRESHOLD,
    SCENE_CHANGE_IDLE_THRESHOLD,
    VIDEO_FPS_DEFAULT,
    VIDEO_FPS_MAX,
    VIDEO_FPS_MIN,
)
from omnistream.media.metrics import set_gauge, video_frames_accepted, video_sampler_fps
from omnistream.media.video.frame import RawVideoFrame

__all__ = ["AdaptiveFrameSampler", "SamplerDecision", "SamplerStats"]


@dataclass(slots=True, frozen=True)
class SamplerDecision:
    """单帧的采样决策。"""

    accepted: bool
    reason: str
    effective_fps: float
    interval_ms: float
    since_last_ms: float
    boosted: bool

    def as_dict(self) -> dict[str, float | str | bool]:
        """转为可序列化字典。"""
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "effective_fps": round(self.effective_fps, 3),
            "interval_ms": round(self.interval_ms, 1),
            "since_last_ms": round(self.since_last_ms, 1),
            "boosted": self.boosted,
        }


@dataclass(slots=True)
class SamplerStats:
    """采样器滚动统计。"""

    seen: int = 0
    accepted: int = 0
    rejected_rate: int = 0
    rejected_load: int = 0
    boosts: int = 0
    boosted_accepts: int = 0

    def as_dict(self) -> dict[str, float]:
        """转为可序列化字典。"""
        ratio = self.accepted / self.seen if self.seen else 0.0
        return {
            "seen": float(self.seen),
            "accepted": float(self.accepted),
            "rejected_rate": float(self.rejected_rate),
            "rejected_load": float(self.rejected_load),
            "boosts": float(self.boosts),
            "boosted_accepts": float(self.boosted_accepts),
            "accept_ratio": round(ratio, 4),
        }


class AdaptiveFrameSampler:
    """按场景变化度与下游负载动态调整采样率的抽帧器。

    典型用法::

        sampler = AdaptiveFrameSampler()
        signal = detector.observe(frame)
        decision = sampler.decide(frame, score=signal.score)
        if decision.accepted:
            buffer.push(frame)
    """

    def __init__(
        self,
        *,
        base_fps: float = VIDEO_FPS_DEFAULT,
        min_fps: float = VIDEO_FPS_MIN,
        max_fps: float = VIDEO_FPS_MAX,
        boost_fps: float = 2.0,
        boost_window_ms: int = SCENE_BOOST_WINDOW_MS,
        boost_threshold: float = SCENE_CHANGE_BOOST_THRESHOLD,
        idle_threshold: float = SCENE_CHANGE_IDLE_THRESHOLD,
        adapt_rate: float = 0.08,
    ) -> None:
        """初始化抽帧器。

        Args:
            base_fps: 初始基础帧率。
            min_fps: 帧率下限。
            max_fps: 帧率上限。
            boost_fps: 场景切换时的提采样帧率。
            boost_window_ms: 提采样持续时长。
            boost_threshold: 触发提采样的场景分数。
            idle_threshold: 判定为静止画面的场景分数。
            adapt_rate: 基础帧率的自适应步长（每帧调整比例）。
        """
        self.min_fps = max(0.01, float(min_fps))
        self.max_fps = max(self.min_fps, float(max_fps))
        self.base_fps = min(self.max_fps, max(self.min_fps, float(base_fps)))
        self.configured_max_fps = self.max_fps
        self.boost_fps = min(self.max_fps, max(self.min_fps, float(boost_fps)))
        self.boost_window_ms = max(0, int(boost_window_ms))
        self.boost_threshold = float(boost_threshold)
        self.idle_threshold = float(idle_threshold)
        self.adapt_rate = min(0.5, max(0.001, float(adapt_rate)))

        self._load: float = 0.0
        self._last_accept_pts: int | None = None
        self._boost_until_pts: int | None = None
        self.stats = SamplerStats()
        self.last_decision: SamplerDecision | None = None

    # ------------------------------------------------------------------
    # 外部控制
    # ------------------------------------------------------------------

    @property
    def load(self) -> float:
        """下游负载 ``0.0 ~ 1.0``，越大越压制帧率。"""
        return self._load

    def set_load(self, load: float) -> None:
        """设置下游负载（VLM 队列深度 / 预算使用率）。"""
        self._load = min(1.0, max(0.0, float(load)))

    def set_max_fps(self, max_fps: float) -> None:
        """由自适应策略下调/恢复帧率上限。"""
        self.configured_max_fps = max(self.min_fps, float(max_fps))
        self.max_fps = self.configured_max_fps
        self.base_fps = min(self.base_fps, self.max_fps)
        self.boost_fps = min(self.boost_fps, self.max_fps)

    def boost(self, pts_ms: int) -> None:
        """从 ``pts_ms`` 起手动开启一次提采样窗口。"""
        self._boost_until_pts = int(pts_ms) + self.boost_window_ms
        self.stats.boosts += 1

    def is_boosting(self, pts_ms: int) -> bool:
        """当前 pts 是否处于提采样窗口内。"""
        return self._boost_until_pts is not None and pts_ms < self._boost_until_pts

    # ------------------------------------------------------------------
    # 帧率计算
    # ------------------------------------------------------------------

    def _adapt_base(self, score: float) -> None:
        """按场景分数缓慢调整基础帧率。"""
        if score >= self.boost_threshold:
            target = self.max_fps
        elif score <= self.idle_threshold:
            target = self.min_fps
        else:
            # 在两个阈值之间做线性插值
            span = max(1e-6, self.boost_threshold - self.idle_threshold)
            ratio = (score - self.idle_threshold) / span
            target = self.min_fps + ratio * (self.max_fps - self.min_fps)
        self.base_fps += (target - self.base_fps) * self.adapt_rate
        self.base_fps = min(self.max_fps, max(self.min_fps, self.base_fps))

    def effective_fps(self, pts_ms: int) -> float:
        """当前有效帧率（已计入提采样与负载压制）。"""
        fps = self.base_fps
        if self.is_boosting(pts_ms) and self._boost_until_pts is not None:
            remaining = max(0, self._boost_until_pts - pts_ms)
            ratio = remaining / float(self.boost_window_ms) if self.boost_window_ms else 0.0
            # 提采样帧率线性衰减回基础帧率
            fps = self.base_fps + (self.boost_fps - self.base_fps) * ratio
        # 负载压制：满载时压到下限
        fps *= 1.0 - 0.9 * self._load
        return min(self.max_fps, max(self.min_fps, fps))

    @property
    def interval_ms(self) -> float:
        """当前基础帧率对应的最小间隔。"""
        return 1000.0 / self.base_fps if self.base_fps > 0 else 1e9

    # ------------------------------------------------------------------
    # 决策
    # ------------------------------------------------------------------

    def decide(self, frame: RawVideoFrame, *, score: float | None = None) -> SamplerDecision:
        """判断一帧是否应当被送往下游。

        Args:
            frame: 待判定的帧。
            score: 场景变化分数；``None`` 时用 ``frame.keyframe_score``。

        Returns:
            :class:`SamplerDecision`。
        """
        self.stats.seen += 1
        value = float(frame.keyframe_score if score is None else score)
        pts = int(frame.pts_ms)

        self._adapt_base(value)
        if value >= self.boost_threshold and not self.is_boosting(pts):
            self.boost(pts)

        boosted = self.is_boosting(pts)
        fps = self.effective_fps(pts)
        interval = 1000.0 / fps if fps > 0 else 1e9

        if self._last_accept_pts is None:
            since = interval
        else:
            since = float(pts - self._last_accept_pts)

        set_gauge(video_sampler_fps, fps)

        # 负载彻底打满时，只放行场景切换帧
        if self._load >= 0.98 and value < self.boost_threshold:
            self.stats.rejected_load += 1
            decision = SamplerDecision(
                accepted=False,
                reason="downstream_saturated",
                effective_fps=fps,
                interval_ms=interval,
                since_last_ms=since,
                boosted=boosted,
            )
            self.last_decision = decision
            return decision

        if since + 1e-6 < interval:
            self.stats.rejected_rate += 1
            decision = SamplerDecision(
                accepted=False,
                reason="rate_limited",
                effective_fps=fps,
                interval_ms=interval,
                since_last_ms=since,
                boosted=boosted,
            )
            self.last_decision = decision
            return decision

        self._last_accept_pts = pts
        self.stats.accepted += 1
        reason = "scene_change" if boosted else "interval"
        if boosted:
            self.stats.boosted_accepts += 1
        try:
            video_frames_accepted.labels(reason=reason).inc()
        except Exception:  # 指标绝不能影响主链路
            pass
        decision = SamplerDecision(
            accepted=True,
            reason=reason,
            effective_fps=fps,
            interval_ms=interval,
            since_last_ms=since,
            boosted=boosted,
        )
        self.last_decision = decision
        return decision

    def accepts(self, frame: RawVideoFrame, *, score: float | None = None) -> bool:
        """:meth:`decide` 的布尔快捷方式。"""
        return self.decide(frame, score=score).accepted

    def reset(self) -> None:
        """重置内部状态与统计。"""
        self._last_accept_pts = None
        self._boost_until_pts = None
        self._load = 0.0
        self.base_fps = min(self.max_fps, max(self.min_fps, VIDEO_FPS_DEFAULT))
        self.stats = SamplerStats()
        self.last_decision = None

    def snapshot(self) -> dict[str, float]:
        """导出统计快照。"""
        data = self.stats.as_dict()
        data["base_fps"] = round(self.base_fps, 3)
        data["max_fps"] = round(self.max_fps, 3)
        data["load"] = round(self._load, 3)
        return data

    def __repr__(self) -> str:
        """调试用文本表示。"""
        return (
            f"AdaptiveFrameSampler(base={self.base_fps:.2f}fps, "
            f"load={self._load:.2f}, accepted={self.stats.accepted}/{self.stats.seen})"
        )

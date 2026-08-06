"""Mock Provider 的延迟模型。

真实厂商的延迟不是常数，而是"基线 + 长尾"。本模型用对数正态分布刻画
TTFT / token 间隔 / TTFA，让延迟看板与 SLO 告警在 Mock 模式下依然有意义。

预置档位::

    LatencyModel.fast()      # 本地/低延迟专线
    LatencyModel.typical()   # 云端实时 API 的常见表现（默认）
    LatencyModel.slow()      # 弱网/高负载
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["LatencyModel", "LatencyProfile"]


@dataclass(slots=True)
class LatencyProfile:
    """一组延迟参数，单位统一为毫秒。"""

    name: str = "typical"
    connect_ms: float = 180.0
    ttft_ms: float = 320.0
    ttft_sigma: float = 0.35
    token_ms: float = 28.0
    token_sigma: float = 0.25
    ttfa_ms: float = 420.0
    ttfa_sigma: float = 0.30
    tool_ms: float = 240.0
    cancel_ms: float = 35.0
    #: 音频推流相对实时的倍率：1.0 表示严格实时，<1 表示比实时更快
    realtime_factor: float = 0.92
    #: 长尾概率与倍率：模拟偶发的网络抖动
    spike_probability: float = 0.03
    spike_multiplier: float = 3.5
    jitter_ms: float = 12.0


_PROFILES: dict[str, LatencyProfile] = {
    "fast": LatencyProfile(name="fast", connect_ms=40.0, ttft_ms=110.0, ttft_sigma=0.22,
                           token_ms=12.0, token_sigma=0.18, ttfa_ms=160.0,
                           ttfa_sigma=0.20, tool_ms=90.0, cancel_ms=15.0,
                           realtime_factor=0.85, spike_probability=0.01,
                           spike_multiplier=2.0, jitter_ms=5.0),
    "typical": LatencyProfile(),
    "slow": LatencyProfile(name="slow", connect_ms=520.0, ttft_ms=780.0, ttft_sigma=0.5,
                           token_ms=55.0, token_sigma=0.4, ttfa_ms=980.0,
                           ttfa_sigma=0.45, tool_ms=650.0, cancel_ms=80.0,
                           realtime_factor=1.0, spike_probability=0.12,
                           spike_multiplier=4.5, jitter_ms=40.0),
}


@dataclass(slots=True)
class LatencyModel:
    """按 profile 采样延迟。同一 seed 下序列完全可复现。"""

    profile: LatencyProfile = field(default_factory=LatencyProfile)
    seed: int = 42
    _rng: np.random.Generator = field(init=False, repr=False)
    _samples: dict[str, list[float]] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._samples = {"ttft": [], "ttfa": [], "token": [], "tool": []}

    # ---------------- 构造快捷方式 ----------------

    @classmethod
    def of(cls, name: str = "typical", *, seed: int = 42) -> "LatencyModel":
        return cls(profile=_PROFILES.get(name, _PROFILES["typical"]), seed=seed)

    @classmethod
    def fast(cls, *, seed: int = 42) -> "LatencyModel":
        return cls.of("fast", seed=seed)

    @classmethod
    def typical(cls, *, seed: int = 42) -> "LatencyModel":
        return cls.of("typical", seed=seed)

    @classmethod
    def slow(cls, *, seed: int = 42) -> "LatencyModel":
        return cls.of("slow", seed=seed)

    # ---------------- 采样 ----------------

    def _lognormal(self, mean_ms: float, sigma: float) -> float:
        """对数正态采样，均值对齐到 ``mean_ms``。"""
        if mean_ms <= 0.0:
            return 0.0
        mu = math.log(max(1e-3, mean_ms)) - 0.5 * sigma * sigma
        value = float(self._rng.lognormal(mean=mu, sigma=sigma))
        if float(self._rng.random()) < self.profile.spike_probability:
            value *= self.profile.spike_multiplier
        value += float(self._rng.normal(0.0, self.profile.jitter_ms))
        return max(0.0, value)

    def _record(self, key: str, value_ms: float) -> float:
        bucket = self._samples.setdefault(key, [])
        bucket.append(value_ms)
        if len(bucket) > 1000:
            del bucket[:500]
        return value_ms

    # ---------------- 毫秒接口 ----------------

    def connect_ms(self) -> float:
        return self._lognormal(self.profile.connect_ms, 0.25)

    def ttft_ms(self) -> float:
        """首个文本 token 延迟。"""
        return self._record("ttft", self._lognormal(self.profile.ttft_ms,
                                                    self.profile.ttft_sigma))

    def ttfa_ms(self) -> float:
        """首个音频字节延迟（关键 SLO）。"""
        return self._record("ttfa", self._lognormal(self.profile.ttfa_ms,
                                                    self.profile.ttfa_sigma))

    def token_ms(self) -> float:
        """相邻文本 token 间隔。"""
        return self._record("token", self._lognormal(self.profile.token_ms,
                                                     self.profile.token_sigma))

    def tool_ms(self) -> float:
        return self._record("tool", self._lognormal(self.profile.tool_ms, 0.4))

    def cancel_ms(self) -> float:
        """取消生效耗时。必须显著小于 100ms 预算。"""
        return min(self.profile.cancel_ms * 1.5,
                   self._lognormal(self.profile.cancel_ms, 0.2))

    # ---------------- 秒接口（asyncio.sleep 直接用） ----------------

    def connect_s(self) -> float:
        return self.connect_ms() / 1000.0

    def ttft_s(self) -> float:
        return self.ttft_ms() / 1000.0

    def ttfa_s(self) -> float:
        return self.ttfa_ms() / 1000.0

    def token_s(self) -> float:
        return self.token_ms() / 1000.0

    def tool_s(self) -> float:
        return self.tool_ms() / 1000.0

    def cancel_s(self) -> float:
        return self.cancel_ms() / 1000.0

    # ---------------- 观测 ----------------

    @property
    def realtime_factor(self) -> float:
        """音频推流速度相对实时的倍率，供生成循环控制节奏。"""
        return self.profile.realtime_factor

    def percentile(self, key: str, q: float = 95.0) -> float:
        """返回已采样序列的分位数（毫秒），无样本时返回 0。"""
        bucket = self._samples.get(key) or []
        if not bucket:
            return 0.0
        return float(np.percentile(np.asarray(bucket, dtype=np.float64), q))

    def stats(self) -> dict[str, float]:
        """导出常用统计，供 Mock 模式下的延迟看板展示。"""
        out: dict[str, float] = {"profile": 0.0}
        out.pop("profile")
        for key in ("ttft", "ttfa", "token", "tool"):
            out[f"{key}_p50_ms"] = round(self.percentile(key, 50.0), 2)
            out[f"{key}_p95_ms"] = round(self.percentile(key, 95.0), 2)
        return out

    def reset(self) -> None:
        """重置随机源与统计，让下一次运行完全复现。"""
        self._rng = np.random.default_rng(self.seed)
        for bucket in self._samples.values():
            bucket.clear()

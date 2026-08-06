"""ASR 引擎路由：按语言 / 延迟预算 / 成本 / 健康度选型，并在失败时降级。

路由不是"读配置直接 new 一个"，而是一条**可解释的决策链**：

1. **显式指定** —— 配置里写死引擎名，直接用（除非不可用）；
2. **硬约束过滤** —— 语言不支持 / 需要网络但处于离线模式 / 依赖缺失，直接淘汰；
3. **打分排序** —— 延迟、成本、准确度三项加权；
4. **熔断规避** —— 最近连续失败超过阈值的引擎降权，冷却期后自动恢复；
5. **降级链** —— 选中的引擎构造失败时，沿 ``fallback_chain`` 逐个尝试，
   最终一定能落到 ``mock``（零依赖、永远可用）。

这条链保证的是：**任何环境下 ASR 都不会整体不可用**，最差退化为 mock。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Final, Iterable

import structlog

from omnistream.perception.asr.base import ASRCapabilities, ASRUnavailable, BaseStreamingASR
from omnistream.perception.asr.mock_asr import MockStreamingASR

log = structlog.get_logger(__name__)

__all__ = [
    "ASRRoute",
    "ASRRouter",
    "EngineHealth",
    "EngineProfile",
    "ENGINE_PROFILES",
    "FALLBACK_CHAIN",
]

#: 引擎构造失败多少次后进入冷却
DEFAULT_FAILURE_THRESHOLD: Final[int] = 3

#: 冷却时长（秒）
DEFAULT_COOLDOWN_S: Final[float] = 60.0

#: 兜底降级链：从左到右尝试，mock 永远可用
FALLBACK_CHAIN: Final[tuple[str, ...]] = (
    "sensevoice", "faster_whisper", "cloud", "mock",
)


@dataclass(slots=True)
class EngineProfile:
    """一个引擎的静态画像，用于打分。"""

    name: str
    #: 典型首个 partial 延迟（毫秒），越小越好
    latency_ms: int = 400
    #: 相对成本（0=免费本地，1=最贵云端）
    cost: float = 0.0
    #: 相对准确度（0~1）
    accuracy: float = 0.7
    #: 支持的语言（``*`` 表示全支持）
    languages: tuple[str, ...] = ("*",)
    needs_network: bool = False
    needs_gpu: bool = False

    def supports(self, language: str) -> bool:
        """是否支持该语言。"""
        if "*" in self.languages:
            return True
        code = (language or "").split("-")[0].lower()
        return code in self.languages


#: 内置引擎画像
ENGINE_PROFILES: Final[dict[str, EngineProfile]] = {
    "mock": EngineProfile(name="mock", latency_ms=120, cost=0.0, accuracy=0.25,
                          languages=("*",)),
    "faster_whisper": EngineProfile(name="faster_whisper", latency_ms=600, cost=0.0,
                                    accuracy=0.86,
                                    languages=("zh", "en", "ja", "ko", "ru", "de",
                                               "fr", "es")),
    "sensevoice": EngineProfile(name="sensevoice", latency_ms=400, cost=0.0,
                                accuracy=0.90,
                                languages=("zh", "yue", "en", "ja", "ko")),
    "funasr": EngineProfile(name="funasr", latency_ms=400, cost=0.0, accuracy=0.90,
                            languages=("zh", "yue", "en", "ja", "ko")),
    "cloud": EngineProfile(name="cloud", latency_ms=200, cost=0.8, accuracy=0.93,
                           languages=("*",), needs_network=True),
    "azure": EngineProfile(name="azure", latency_ms=220, cost=0.8, accuracy=0.93,
                           languages=("*",), needs_network=True),
    "deepgram": EngineProfile(name="deepgram", latency_ms=180, cost=0.85, accuracy=0.92,
                              languages=("*",), needs_network=True),
    "volc": EngineProfile(name="volc", latency_ms=200, cost=0.7, accuracy=0.94,
                          languages=("zh", "en"), needs_network=True),
}

#: 把配置里的引擎名映射到实际实现名
_ALIASES: Final[dict[str, str]] = {
    "funasr": "sensevoice",
    "whisper": "faster_whisper",
    "azure": "cloud",
    "deepgram": "cloud",
    "volc": "cloud",
    "": "mock",
}


@dataclass(slots=True)
class EngineHealth:
    """引擎运行时健康状态。"""

    name: str
    failures: int = 0
    successes: int = 0
    last_failure_at: float = 0.0
    last_error: str = ""

    @property
    def cooling(self) -> bool:
        """是否处于冷却期。"""
        if self.failures < DEFAULT_FAILURE_THRESHOLD:
            return False
        return (time.monotonic() - self.last_failure_at) < DEFAULT_COOLDOWN_S

    def record_success(self) -> None:
        self.successes += 1
        self.failures = 0
        self.last_error = ""

    def record_failure(self, error: str) -> None:
        self.failures += 1
        self.last_failure_at = time.monotonic()
        self.last_error = error[:200]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "failures": self.failures,
            "successes": self.successes,
            "cooling": self.cooling,
            "last_error": self.last_error,
        }


@dataclass(slots=True)
class ASRRoute:
    """一次路由决策的结果与理由。"""

    engine: str
    reason: str
    score: float = 0.0
    fallback_from: str = ""
    candidates: tuple[str, ...] = ()
    profile: EngineProfile | None = field(default=None, repr=False)

    @property
    def degraded(self) -> bool:
        """是否发生了降级。"""
        return bool(self.fallback_from) and self.fallback_from != self.engine

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "reason": self.reason,
            "score": round(self.score, 3),
            "fallback_from": self.fallback_from,
            "degraded": self.degraded,
            "candidates": list(self.candidates),
        }


class ASRRouter:
    """ASR 引擎选择器与构造器。

    Example::

        router = ASRRouter(preferred="sensevoice", offline=False)
        asr, route = router.create(language="zh-CN", session_id="s1")
        if route.degraded:
            log.warning("asr_degraded", **route.as_dict())
    """

    def __init__(
        self,
        *,
        preferred: str = "mock",
        offline: bool = False,
        latency_budget_ms: int = 800,
        cost_weight: float = 0.3,
        accuracy_weight: float = 0.5,
        latency_weight: float = 0.2,
        fallback_chain: Iterable[str] = FALLBACK_CHAIN,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            preferred: 配置指定的引擎名（``auto`` 表示自动选型）。
            offline: 离线模式，淘汰所有需要网络的引擎。
            latency_budget_ms: 首个 partial 的延迟预算。
            cost_weight/accuracy_weight/latency_weight: 打分权重。
            fallback_chain: 降级链。
            engine_kwargs: 传给引擎构造函数的额外参数（模型路径、密钥等）。
        """
        self.preferred = _ALIASES.get(preferred, preferred) or "mock"
        self.raw_preferred = preferred
        self.offline = bool(offline)
        self.latency_budget_ms = int(latency_budget_ms)
        self.cost_weight = float(cost_weight)
        self.accuracy_weight = float(accuracy_weight)
        self.latency_weight = float(latency_weight)
        self.fallback_chain = tuple(fallback_chain)
        self.engine_kwargs = dict(engine_kwargs or {})
        self._health: dict[str, EngineHealth] = {}
        self.decisions = 0
        self.degradations = 0

    # ------------------------------------------------------------------
    # 可用性
    # ------------------------------------------------------------------

    def health_of(self, name: str) -> EngineHealth:
        """取（或创建）引擎健康记录。"""
        if name not in self._health:
            self._health[name] = EngineHealth(name=name)
        return self._health[name]

    def _engine_class(self, name: str) -> type[BaseStreamingASR] | None:
        """按名加载引擎类（延迟 import，避免为未使用的引擎付出 import 代价）。"""
        resolved = _ALIASES.get(name, name)
        if resolved == "mock":
            return MockStreamingASR
        if resolved == "faster_whisper":
            from omnistream.perception.asr.faster_whisper_asr import FasterWhisperASR

            return FasterWhisperASR
        if resolved == "sensevoice":
            from omnistream.perception.asr.sensevoice_asr import SenseVoiceASR

            return SenseVoiceASR
        if resolved == "cloud":
            from omnistream.perception.asr.cloud_stream_asr import CloudStreamASR

            return CloudStreamASR
        return None

    def is_usable(self, name: str, language: str) -> tuple[bool, str]:
        """判断引擎在当前条件下是否可用。

        Returns:
            ``(可用, 不可用原因)``。
        """
        profile = ENGINE_PROFILES.get(_ALIASES.get(name, name))
        if profile is None:
            return False, "unknown_engine"
        if not profile.supports(language):
            return False, f"language_unsupported:{language}"
        if self.offline and profile.needs_network:
            return False, "offline_mode"
        if self.health_of(name).cooling:
            return False, "cooling_down"
        cls = self._engine_class(name)
        if cls is None:
            return False, "no_implementation"
        try:
            if not cls.is_available():
                return False, "dependency_missing"
        except Exception as exc:  # noqa: BLE001
            return False, f"availability_check_failed:{exc}"
        return True, ""

    def score(self, name: str) -> float:
        """给引擎打分，越高越优先。"""
        profile = ENGINE_PROFILES.get(_ALIASES.get(name, name))
        if profile is None:
            return 0.0
        # 延迟得分：在预算内满分，超预算线性衰减
        if profile.latency_ms <= self.latency_budget_ms:
            latency_score = 1.0 - profile.latency_ms / max(1, self.latency_budget_ms) * 0.5
        else:
            latency_score = max(0.0, 0.5 - (profile.latency_ms - self.latency_budget_ms)
                                / max(1, self.latency_budget_ms) * 0.5)
        cost_score = 1.0 - min(1.0, profile.cost)
        total = (self.accuracy_weight * profile.accuracy
                 + self.latency_weight * latency_score
                 + self.cost_weight * cost_score)
        health = self.health_of(name)
        if health.failures:
            total *= max(0.2, 1.0 - 0.25 * health.failures)
        return total

    # ------------------------------------------------------------------
    # 决策
    # ------------------------------------------------------------------

    def decide(self, *, language: str = "zh-CN") -> ASRRoute:
        """选出目标引擎（不构造实例）。

        Args:
            language: 目标语言。

        Returns:
            路由决策。
        """
        self.decisions += 1
        candidates: list[str] = []
        for name in dict.fromkeys((self.preferred, *self.fallback_chain)):
            usable, _ = self.is_usable(name, language)
            if usable:
                candidates.append(name)

        # 显式指定且可用 -> 直接采纳
        if self.preferred not in ("auto", "") and self.preferred in candidates:
            return ASRRoute(engine=self.preferred, reason="explicit",
                            score=self.score(self.preferred),
                            candidates=tuple(candidates),
                            profile=ENGINE_PROFILES.get(self.preferred))

        if not candidates:
            return ASRRoute(engine="mock", reason="no_candidate",
                            score=self.score("mock"), fallback_from=self.preferred,
                            candidates=(), profile=ENGINE_PROFILES.get("mock"))

        best = max(candidates, key=self.score)
        reason = "auto_selected" if self.preferred in ("auto", "") else "preferred_unusable"
        fallback_from = "" if reason == "auto_selected" else self.preferred
        return ASRRoute(engine=best, reason=reason, score=self.score(best),
                        fallback_from=fallback_from, candidates=tuple(candidates),
                        profile=ENGINE_PROFILES.get(best))

    def create(
        self,
        *,
        language: str = "zh-CN",
        session_id: str = "",
        turn_id: str = "",
        hotwords: Iterable[str] | None = None,
        **overrides: Any,
    ) -> tuple[BaseStreamingASR, ASRRoute]:
        """构造引擎实例；失败时沿降级链继续尝试。

        Args:
            language: 目标语言。
            session_id: 会话 id。
            turn_id: 轮次 id。
            hotwords: 热词表。
            **overrides: 覆盖 ``engine_kwargs``。

        Returns:
            ``(引擎实例, 路由决策)``；最差返回 :class:`MockStreamingASR`。
        """
        route = self.decide(language=language)
        chain = [route.engine] + [n for n in self.fallback_chain if n != route.engine]
        kwargs = {**self.engine_kwargs, **overrides}
        original = route.engine

        for name in chain:
            usable, why = self.is_usable(name, language)
            if not usable and name != "mock":
                log.debug("asr.route_skip", engine=name, reason=why)
                continue
            cls = self._engine_class(name) or MockStreamingASR
            accepted = _filter_kwargs(cls, kwargs)
            try:
                instance = cls(
                    language=language,
                    session_id=session_id,
                    turn_id=turn_id,
                    hotwords=list(hotwords or []),
                    **accepted,
                )
            except ASRUnavailable as exc:
                self.health_of(name).record_failure(str(exc))
                log.warning("asr.engine_unavailable", engine=name, error=str(exc))
                continue
            except Exception as exc:  # noqa: BLE001
                self.health_of(name).record_failure(str(exc))
                log.warning("asr.engine_init_failed", engine=name, error=str(exc))
                continue

            self.health_of(name).record_success()
            if name != original:
                self.degradations += 1
                route = ASRRoute(engine=name, reason="fallback",
                                 score=self.score(name), fallback_from=original,
                                 candidates=route.candidates,
                                 profile=ENGINE_PROFILES.get(name))
                log.warning("asr.degraded", **route.as_dict())
            else:
                log.info("asr.route", **route.as_dict())
            return instance, route

        # 理论不可达：mock 无依赖必然构造成功；保险起见兜底
        fallback = MockStreamingASR(language=language, session_id=session_id,
                                    turn_id=turn_id, hotwords=list(hotwords or []))
        self.degradations += 1
        return fallback, ASRRoute(engine="mock", reason="last_resort",
                                  score=self.score("mock"), fallback_from=original,
                                  candidates=route.candidates,
                                  profile=ENGINE_PROFILES.get("mock"))

    def report_failure(self, engine: str, error: str) -> None:
        """运行期失败上报（由编排层在解码异常时调用）。"""
        self.health_of(engine).record_failure(error)
        log.warning("asr.runtime_failure", engine=engine, error=error[:200])

    def report_success(self, engine: str) -> None:
        """运行期成功上报，清零失败计数。"""
        self.health_of(engine).record_success()

    def capabilities_of(self, engine: str) -> ASRCapabilities:
        """取引擎能力声明。"""
        cls = self._engine_class(engine)
        return cls.capabilities if cls is not None else ASRCapabilities()

    def stats(self) -> dict[str, Any]:
        """路由统计。"""
        return {
            "preferred": self.raw_preferred,
            "offline": self.offline,
            "decisions": self.decisions,
            "degradations": self.degradations,
            "health": {k: v.as_dict() for k, v in self._health.items()},
        }

    def __repr__(self) -> str:
        return f"<ASRRouter preferred={self.preferred} offline={self.offline}>"


def _filter_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """只保留目标构造函数认识的关键字参数。"""
    import inspect

    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):  # pragma: no cover
        return {}
    names = {
        p.name for p in sig.parameters.values()
        if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
    }
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in names}

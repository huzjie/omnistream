"""TTS 引擎路由：按 TTFA 预算 / 音质 / 成本 / 健康度选型，并在失败时降级。

与 ASR 路由同构的一条**可解释决策链**：

1. **显式指定** —— 配置写死引擎名，直接用（除非不可用）；
2. **硬约束过滤** —— 语言不支持 / 需要网络但离线 / 依赖缺失 / 熔断冷却中；
3. **打分排序** —— TTFA、音质、成本三项加权，超出 TTFA 预算的重罚；
4. **熔断规避** —— 连续构造失败的引擎进入冷却，冷却期满自动恢复；
5. **降级链** —— 沿 ``fallback_chain`` 逐个尝试，终点是永远可用的 ``mock``。

与 ASR 的关键差异
-----------------
TTS 的第一权重不是准确率而是 **TTFA（首字发声延迟）**：
用户宁可听到音质一般但立刻开口的回答，也不愿等 800ms 的完美音色。
所以默认权重是 ``ttfa 0.5 / quality 0.35 / cost 0.15``，
且一旦某引擎的典型 TTFA 超出预算，分数直接砍半——这不是软偏好，是硬倾向。
"""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Final, Iterable

import structlog

from omnistream.perception.tts.base import (
    BaseStreamingTTS,
    TTSCapabilities,
    TTSUnavailable,
)
from omnistream.perception.tts.cache import CachedTTS, TTSCache
from omnistream.perception.tts.mock_tts import MockStreamingTTS

log = structlog.get_logger(__name__)

__all__ = [
    "ENGINE_PROFILES",
    "FALLBACK_CHAIN",
    "TTSEngineHealth",
    "TTSEngineProfile",
    "TTSRoute",
    "TTSRouter",
]

#: 引擎构造失败多少次后进入冷却
DEFAULT_FAILURE_THRESHOLD: Final[int] = 3

#: 冷却时长（秒）
DEFAULT_COOLDOWN_S: Final[float] = 60.0

#: 默认 TTFA 预算（毫秒）。超出预算的引擎会被显著降权。
DEFAULT_TTFA_BUDGET_MS: Final[int] = 350

#: 兜底降级链：本地优先，mock 永远可用
FALLBACK_CHAIN: Final[tuple[str, ...]] = ("piper", "cloud", "mock")


@dataclass(slots=True)
class TTSEngineProfile:
    """一个引擎的静态画像，用于打分。"""

    name: str
    #: 典型首包音频延迟（毫秒），越小越好
    ttfa_ms: int = 300
    #: 相对成本（0=免费本地，1=最贵云端）
    cost: float = 0.0
    #: 相对音质/自然度（0~1）
    quality: float = 0.7
    #: 支持的语言（``*`` 表示全支持）
    languages: tuple[str, ...] = ("*",)
    needs_network: bool = False
    needs_gpu: bool = False
    #: 是否支持情绪控制
    emotion: bool = False

    def supports(self, language: str) -> bool:
        """是否支持该语言。"""
        if "*" in self.languages:
            return True
        code = (language or "").split("-")[0].lower()
        return code in self.languages


#: 内置引擎画像
ENGINE_PROFILES: Final[dict[str, TTSEngineProfile]] = {
    "mock": TTSEngineProfile(name="mock", ttfa_ms=25, cost=0.0, quality=0.20,
                             languages=("*",), emotion=True),
    "piper": TTSEngineProfile(name="piper", ttfa_ms=180, cost=0.0, quality=0.78,
                              languages=("zh", "en", "de", "fr", "es", "it",
                                         "ru", "pt", "nl")),
    "cloud": TTSEngineProfile(name="cloud", ttfa_ms=300, cost=0.75, quality=0.95,
                              languages=("*",), needs_network=True, emotion=True),
    "azure": TTSEngineProfile(name="azure", ttfa_ms=300, cost=0.7, quality=0.94,
                              languages=("*",), needs_network=True, emotion=True),
    "openai": TTSEngineProfile(name="openai", ttfa_ms=420, cost=0.8, quality=0.93,
                               languages=("*",), needs_network=True),
    "elevenlabs": TTSEngineProfile(name="elevenlabs", ttfa_ms=380, cost=1.0,
                                   quality=0.97, languages=("*",),
                                   needs_network=True, emotion=True),
    "volc": TTSEngineProfile(name="volc", ttfa_ms=260, cost=0.6, quality=0.94,
                             languages=("zh", "en", "ja"), needs_network=True,
                             emotion=True),
}

#: 配置里的引擎名 -> 实际实现名
_ALIASES: Final[dict[str, str]] = {
    "azure": "cloud",
    "openai": "cloud",
    "elevenlabs": "cloud",
    "volc": "cloud",
    "generic": "cloud",
    "generic_ws": "cloud",
    "vits": "piper",
    "": "mock",
}

#: 实现名 -> 云端方言名（构造 CloudStreamTTS 时透传）
_DIALECT_OF: Final[dict[str, str]] = {
    "azure": "azure",
    "openai": "openai",
    "elevenlabs": "elevenlabs",
    "volc": "volc",
    "cloud": "generic",
    "generic": "generic",
    "generic_ws": "generic_ws",
}


@dataclass(slots=True)
class TTSEngineHealth:
    """引擎运行时健康状态。"""

    name: str
    failures: int = 0
    successes: int = 0
    last_failure_at: float = 0.0
    last_error: str = ""

    @property
    def cooling(self) -> bool:
        """是否处于熔断冷却期。"""
        if self.failures < DEFAULT_FAILURE_THRESHOLD:
            return False
        return (time.monotonic() - self.last_failure_at) < DEFAULT_COOLDOWN_S

    def record_success(self) -> None:
        """记一次成功，清零失败计数。"""
        self.successes += 1
        self.failures = 0
        self.last_error = ""

    def record_failure(self, error: str) -> None:
        """记一次失败。"""
        self.failures += 1
        self.last_failure_at = time.monotonic()
        self.last_error = error[:200]

    def as_dict(self) -> dict[str, Any]:
        """序列化。"""
        return {
            "engine": self.name,
            "failures": self.failures,
            "successes": self.successes,
            "cooling": self.cooling,
            "last_error": self.last_error,
        }


@dataclass(slots=True)
class TTSRoute:
    """一次路由决策的结果与理由。"""

    engine: str
    reason: str
    score: float = 0.0
    fallback_from: str = ""
    candidates: tuple[str, ...] = ()
    dialect: str = ""
    cached: bool = False
    profile: TTSEngineProfile | None = field(default=None, repr=False)

    @property
    def degraded(self) -> bool:
        """是否发生了降级。"""
        return bool(self.fallback_from) and self.fallback_from != self.engine

    def as_dict(self) -> dict[str, Any]:
        """序列化（写进 metrics / 事件流）。"""
        return {
            "engine": self.engine,
            "reason": self.reason,
            "score": round(self.score, 3),
            "fallback_from": self.fallback_from,
            "degraded": self.degraded,
            "dialect": self.dialect,
            "cached": self.cached,
            "candidates": list(self.candidates),
        }


class TTSRouter:
    """TTS 引擎选择器与构造器。

    Example::

        router = TTSRouter(preferred="auto", offline=True)
        tts, route = router.create(language="zh-CN", voice="xiaoyan")
        if route.degraded:
            log.warning("tts_degraded", **route.as_dict())
    """

    def __init__(
        self,
        *,
        preferred: str = "mock",
        offline: bool = False,
        ttfa_budget_ms: int = DEFAULT_TTFA_BUDGET_MS,
        ttfa_weight: float = 0.5,
        quality_weight: float = 0.35,
        cost_weight: float = 0.15,
        fallback_chain: Iterable[str] = FALLBACK_CHAIN,
        engine_kwargs: dict[str, Any] | None = None,
        cache_enabled: bool = True,
        cache_size: int = 512,
        shared_cache: TTSCache | None = None,
    ) -> None:
        """初始化路由器。

        Args:
            preferred: 配置指定的引擎名（``auto`` 表示自动选型）。
            offline: 离线模式，淘汰所有需要网络的引擎。
            ttfa_budget_ms: 首字发声延迟预算。
            ttfa_weight: TTFA 权重。
            quality_weight: 音质权重。
            cost_weight: 成本权重。
            fallback_chain: 降级链。
            engine_kwargs: 传给引擎构造函数的额外参数。
            cache_enabled: 是否给选中的引擎套缓存层。
            cache_size: 新建缓存的条目上限。
            shared_cache: 跨会话共享的缓存实例。
        """
        raw = (preferred or "mock").strip().lower()
        self.raw_preferred = raw
        self.preferred = _ALIASES.get(raw, raw) or "mock"
        self.offline = bool(offline)
        self.ttfa_budget_ms = int(ttfa_budget_ms)
        self.ttfa_weight = float(ttfa_weight)
        self.quality_weight = float(quality_weight)
        self.cost_weight = float(cost_weight)
        self.fallback_chain = tuple(fallback_chain)
        self.engine_kwargs = dict(engine_kwargs or {})
        self.cache_enabled = bool(cache_enabled)
        self.cache_size = max(0, int(cache_size))
        self.shared_cache = shared_cache
        self._health: dict[str, TTSEngineHealth] = {}
        self.decisions = 0
        self.degradations = 0

    # ------------------------------------------------------------------
    # 可用性
    # ------------------------------------------------------------------

    def health_of(self, name: str) -> TTSEngineHealth:
        """取（或创建）引擎健康记录。"""
        if name not in self._health:
            self._health[name] = TTSEngineHealth(name=name)
        return self._health[name]

    def _engine_class(self, name: str) -> type[BaseStreamingTTS] | None:
        """按名延迟加载引擎类，避免为未使用的引擎付出 import 代价。"""
        resolved = _ALIASES.get(name, name)
        if resolved == "mock":
            return MockStreamingTTS
        if resolved == "piper":
            from omnistream.perception.tts.piper_tts import PiperTTS

            return PiperTTS
        if resolved == "cloud":
            from omnistream.perception.tts.cloud_tts import CloudStreamTTS

            return CloudStreamTTS
        return None

    def is_usable(self, name: str, language: str) -> tuple[bool, str]:
        """判断引擎在当前条件下是否可用。

        Args:
            name: 引擎名（可以是别名）。
            language: 目标语言。

        Returns:
            ``(可用, 不可用原因)``。
        """
        resolved = _ALIASES.get(name, name)
        profile = ENGINE_PROFILES.get(name) or ENGINE_PROFILES.get(resolved)
        if profile is None:
            return False, f"未知引擎 {name}"
        if not profile.supports(language):
            return False, f"{name} 不支持语言 {language}"
        if self.offline and profile.needs_network:
            return False, f"{name} 需要网络但当前是离线模式"
        cls = self._engine_class(resolved)
        if cls is None:
            return False, f"{name} 没有对应实现"
        if not cls.is_available():
            return False, f"{name} 依赖缺失"
        if resolved == "cloud":
            ok, reason = self._cloud_ready(name)
            if not ok:
                return False, reason
        if self.health_of(resolved).cooling:
            return False, f"{name} 处于熔断冷却期"
        return True, ""

    def _cloud_ready(self, name: str) -> tuple[bool, str]:
        """云端方言的前置条件检查：传输依赖 + API Key。

        提前判掉"注定 401"的引擎，避免它在打分阶段挤掉本地引擎、
        然后在用户已经说完话之后才失败。

        Args:
            name: 引擎名或方言名。

        Returns:
            ``(是否就绪, 原因)``。
        """
        import os

        from omnistream.perception.tts.cloud_tts import cloud_tts_available, get_tts_dialect

        dialect_name = _DIALECT_OF.get(name, "generic")
        dialect = get_tts_dialect(dialect_name)
        if not cloud_tts_available(dialect.transport):
            return False, f"{name} 缺少 {dialect.transport} 传输依赖"
        if not dialect.requires_key:
            return True, ""
        if self.engine_kwargs.get("api_key"):
            return True, ""
        if any(os.environ.get(env, "").strip() for env in dialect.api_key_envs):
            return True, ""
        return False, f"{name} 未配置 API Key（{'/'.join(dialect.api_key_envs)}）"

    def score(self, name: str) -> float:
        """给引擎打分（0~1，越高越好）。

        Args:
            name: 引擎名。

        Returns:
            加权得分；超出 TTFA 预算的引擎分数砍半。
        """
        profile = ENGINE_PROFILES.get(name) or ENGINE_PROFILES.get(
            _ALIASES.get(name, name)
        )
        if profile is None:
            return 0.0
        budget = max(1, self.ttfa_budget_ms)
        # TTFA 项：预算内接近 1，越慢越小
        ttfa_score = max(0.0, min(1.0, 1.0 - profile.ttfa_ms / (budget * 2.0)))
        cost_score = 1.0 - max(0.0, min(1.0, profile.cost))
        total = (
            self.ttfa_weight * ttfa_score
            + self.quality_weight * profile.quality
            + self.cost_weight * cost_score
        )
        if profile.ttfa_ms > budget:
            total *= 0.5
        health = self.health_of(_ALIASES.get(name, name))
        if health.failures:
            total *= max(0.2, 1.0 - 0.25 * health.failures)
        return round(total, 4)

    # ------------------------------------------------------------------
    # 决策
    # ------------------------------------------------------------------

    def decide(self, *, language: str = "zh-CN") -> TTSRoute:
        """只做选型不构造，返回可解释的决策结果。

        Args:
            language: 目标语言。

        Returns:
            路由决策。
        """
        self.decisions += 1
        usable: list[str] = []
        rejected: dict[str, str] = {}
        for name in ENGINE_PROFILES:
            ok, reason = self.is_usable(name, language)
            if ok:
                usable.append(name)
            else:
                rejected[name] = reason

        # 1) 显式指定
        if self.raw_preferred not in {"auto", "", "any"}:
            target = self.raw_preferred
            ok, reason = self.is_usable(target, language)
            if ok:
                resolved = _ALIASES.get(target, target)
                return TTSRoute(
                    engine=resolved,
                    reason=f"配置显式指定 {target}",
                    score=self.score(target),
                    candidates=tuple(usable),
                    dialect=_DIALECT_OF.get(target, ""),
                    profile=ENGINE_PROFILES.get(target),
                )
            log.warning("tts.route.preferred_unusable",
                        preferred=target, reason=reason)

        # 2) 打分排序
        if usable:
            ranked = sorted(usable, key=self.score, reverse=True)
            best = ranked[0]
            resolved = _ALIASES.get(best, best)
            return TTSRoute(
                engine=resolved,
                reason=f"打分最高（ttfa 预算 {self.ttfa_budget_ms}ms）",
                score=self.score(best),
                candidates=tuple(ranked),
                dialect=_DIALECT_OF.get(best, ""),
                profile=ENGINE_PROFILES.get(best),
            )

        # 3) 全军覆没
        return TTSRoute(
            engine="mock",
            reason="没有可用引擎，回落到 mock：" + "; ".join(
                f"{k}={v}" for k, v in list(rejected.items())[:4]
            ),
            score=self.score("mock"),
            candidates=("mock",),
            profile=ENGINE_PROFILES["mock"],
        )

    def create(
        self,
        *,
        language: str = "zh-CN",
        voice: str = "default",
        sample_rate: int = 24000,
        speed: float = 1.0,
        pitch: float = 0.0,
        volume_db: float = 0.0,
        session_id: str = "",
        **overrides: Any,
    ) -> tuple[BaseStreamingTTS, TTSRoute]:
        """选型并构造引擎，失败时沿降级链继续尝试。

        Args:
            language: 目标语言。
            voice: 逻辑音色名。
            sample_rate: 输出采样率。
            speed: 语速倍率。
            pitch: 音高偏移（半音）。
            volume_db: 音量增益。
            session_id: 会话 ID。
            **overrides: 覆盖 ``engine_kwargs`` 的构造参数。

        Returns:
            ``(引擎实例, 路由结果)``。引擎实例一定非空。
        """
        route = self.decide(language=language)
        chain: list[str] = [route.engine]
        for name in self.fallback_chain:
            resolved = _ALIASES.get(name, name)
            if resolved not in chain:
                chain.append(resolved)
        if "mock" not in chain:
            chain.append("mock")

        base_kwargs: dict[str, Any] = {
            "voice": voice,
            "language": language,
            "sample_rate": sample_rate,
            "speed": speed,
            "pitch": pitch,
            "volume_db": volume_db,
            "session_id": session_id,
        }
        base_kwargs.update(self.engine_kwargs)
        base_kwargs.update(overrides)

        first = chain[0]
        last_error = ""
        for name in chain:
            cls = self._engine_class(name)
            if cls is None:
                continue
            ok, reason = self.is_usable(name, language)
            if not ok and name != "mock":
                last_error = reason
                continue
            kwargs = dict(base_kwargs)
            # 首选引擎沿用 decide() 已经算好的方言，降级目标按自身名字取
            dialect = route.dialect if name == route.engine else _DIALECT_OF.get(name, "")
            if name == "cloud" and dialect:
                kwargs.setdefault("dialect", dialect)
            try:
                engine = cls(**_filter_kwargs(cls, kwargs))
            except TTSUnavailable as exc:
                last_error = str(exc)
                self.health_of(name).record_failure(last_error)
                log.warning("tts.route.engine_unavailable",
                            engine=name, error=last_error)
                continue
            except Exception as exc:  # noqa: BLE001 - 任何构造异常都触发降级
                last_error = f"{type(exc).__name__}: {exc}"
                self.health_of(name).record_failure(last_error)
                log.warning("tts.route.engine_failed", engine=name, error=last_error)
                continue

            self.health_of(name).record_success()
            if name != first:
                self.degradations += 1
                route = TTSRoute(
                    engine=name,
                    reason=f"降级：{first} 不可用（{last_error or '构造失败'}）",
                    score=self.score(name),
                    fallback_from=first,
                    candidates=tuple(chain),
                    dialect=kwargs.get("dialect", ""),
                    profile=ENGINE_PROFILES.get(name),
                )
            else:
                route.dialect = kwargs.get("dialect", route.dialect)

            engine = self._maybe_wrap_cache(engine)
            route.cached = isinstance(engine, CachedTTS)
            log.info("tts.route.selected", **route.as_dict())
            return engine, route

        # 理论上不可达：mock 零依赖恒可构造，这里只是最后一道保险
        engine = self._maybe_wrap_cache(MockStreamingTTS(**_filter_kwargs(
            MockStreamingTTS, base_kwargs)))
        route = TTSRoute(
            engine="mock",
            reason=f"全链降级到 mock：{last_error}",
            score=self.score("mock"),
            fallback_from=first,
            candidates=tuple(chain),
            cached=isinstance(engine, CachedTTS),
            profile=ENGINE_PROFILES["mock"],
        )
        self.degradations += 1
        log.warning("tts.route.all_failed", **route.as_dict())
        return engine, route

    def _maybe_wrap_cache(self, engine: BaseStreamingTTS) -> BaseStreamingTTS:
        """按配置给引擎套上缓存层。"""
        if not self.cache_enabled or self.cache_size <= 0:
            return engine
        return CachedTTS(engine, cache=self.shared_cache,
                         max_entries=self.cache_size)

    # ------------------------------------------------------------------
    # 运行时反馈
    # ------------------------------------------------------------------

    def report_failure(self, engine: str, error: str) -> None:
        """运行期失败回报（合成中途报错时由编排器调用）。"""
        self.health_of(_ALIASES.get(engine, engine)).record_failure(error)

    def report_success(self, engine: str) -> None:
        """运行期成功回报。"""
        self.health_of(_ALIASES.get(engine, engine)).record_success()

    def capabilities_of(self, engine: str) -> TTSCapabilities:
        """取引擎能力声明。"""
        cls = self._engine_class(engine)
        return cls.capabilities if cls is not None else TTSCapabilities()

    def stats(self) -> dict[str, Any]:
        """统计快照。"""
        return {
            "preferred": self.raw_preferred,
            "offline": self.offline,
            "ttfa_budget_ms": self.ttfa_budget_ms,
            "decisions": self.decisions,
            "degradations": self.degradations,
            "cache_enabled": self.cache_enabled,
            "health": [h.as_dict() for h in self._health.values()],
        }

    def __repr__(self) -> str:
        return (f"<TTSRouter preferred={self.raw_preferred} "
                f"offline={self.offline} degradations={self.degradations}>")


def _filter_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """只保留目标类 ``__init__`` 真正接受的关键字参数。

    这样上层可以无脑把所有配置塞进来，各引擎各取所需，
    不会因为多传一个 ``model_path`` 就把 mock 引擎搞崩。
    """
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    allowed = {name for name in sig.parameters if name != "self"}
    return {k: v for k, v in kwargs.items() if k in allowed}

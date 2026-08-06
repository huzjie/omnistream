"""流式语音识别子包。

对外只暴露两个入口：

* :func:`create_asr` —— 按引擎名（或 ``auto``）构造识别器，自带降级；
* :class:`StreamingASRSession` —— 把拉模式引擎包成推模式，供编排器使用。

Example::

    from omnistream.perception.asr import create_asr, StreamingASRSession

    asr, route = create_asr("auto", language="zh-CN", session_id="s1")
    session = StreamingASRSession(asr)
    await session.start()
    await session.push(chunk)
    async for delta in session.deltas():
        print(delta.text)
"""
from __future__ import annotations

from typing import Any, Iterable

from omnistream.perception.asr.base import (
    ASR_SAMPLE_RATE,
    ASRCapabilities,
    ASRUnavailable,
    BaseStreamingASR,
    StreamingASRSession,
    build_delta,
    build_word_timing,
)
from omnistream.perception.asr.hotwords import Hotword, HotwordBias
from omnistream.perception.asr.language_id import (
    DEFAULT_IDENTIFIER,
    LanguageGuess,
    LanguageIdentifier,
    detect_language,
)
from omnistream.perception.asr.mock_asr import MockStreamingASR
from omnistream.perception.asr.punctuation import PauseMark, PunctuationRestorer
from omnistream.perception.asr.router import (
    ENGINE_PROFILES,
    FALLBACK_CHAIN,
    ASRRoute,
    ASRRouter,
)
from omnistream.perception.asr.stabilizer import (
    LocalAgreementStabilizer,
    SegmentStabilizer,
    StabilizedText,
)
from omnistream.perception.asr.word_timing import (
    WordSpan,
    interpolate_words,
    locate_word_at,
    text_before_ms,
)

__all__ = [
    "ASR_SAMPLE_RATE",
    "ASRCapabilities",
    "ASRRoute",
    "ASRRouter",
    "ASRUnavailable",
    "BaseStreamingASR",
    "DEFAULT_IDENTIFIER",
    "ENGINE_PROFILES",
    "FALLBACK_CHAIN",
    "Hotword",
    "HotwordBias",
    "LanguageGuess",
    "LanguageIdentifier",
    "LocalAgreementStabilizer",
    "MockStreamingASR",
    "PauseMark",
    "PunctuationRestorer",
    "SegmentStabilizer",
    "StabilizedText",
    "StreamingASRSession",
    "WordSpan",
    "build_delta",
    "build_word_timing",
    "create_asr",
    "create_asr_router",
    "detect_language",
    "interpolate_words",
    "locate_word_at",
    "text_before_ms",
]


def create_asr_router(
    settings: Any = None,
    *,
    engine: str = "",
    offline: bool = False,
    latency_budget_ms: int = 800,
    **engine_kwargs: Any,
) -> ASRRouter:
    """从配置构造路由器。

    Args:
        settings: :class:`~omnistream.config.perception.PerceptionSettings`，
            给出时从中读取引擎名与模型参数。
        engine: 显式覆盖引擎名。
        offline: 离线模式（禁用云端引擎）。
        latency_budget_ms: 首个 partial 延迟预算。
        **engine_kwargs: 追加/覆盖引擎构造参数。

    Returns:
        配置好的 :class:`ASRRouter`。
    """
    resolved = engine
    kwargs: dict[str, Any] = {}
    if settings is not None:
        resolved = resolved or str(getattr(settings, "asr_engine", "mock"))
        kwargs.update({
            "model_size": getattr(settings, "asr_model", "small"),
            "model": getattr(settings, "asr_model", "small"),
            "model_path": getattr(settings, "asr_model_path", None),
            "device": getattr(settings, "asr_device", "auto"),
            "compute_type": getattr(settings, "asr_compute_type", "int8"),
            "beam_size": getattr(settings, "asr_beam_size", 1),
            "stride_ms": getattr(settings, "asr_stride_ms", 160) or 160,
            "vad_filter": getattr(settings, "asr_vad_filter", True),
            "enable_word_timestamps": getattr(settings, "asr_word_timestamps", False),
        })
    kwargs.update(engine_kwargs)
    return ASRRouter(
        preferred=resolved or "mock",
        offline=offline,
        latency_budget_ms=latency_budget_ms,
        engine_kwargs=kwargs,
    )


def create_asr(
    engine: str = "auto",
    *,
    language: str = "zh-CN",
    session_id: str = "",
    turn_id: str = "",
    hotwords: Iterable[str] | None = None,
    settings: Any = None,
    offline: bool = False,
    **engine_kwargs: Any,
) -> tuple[BaseStreamingASR, ASRRoute]:
    """构造流式识别器（带完整降级链）。

    Args:
        engine: ``auto`` / ``mock`` / ``faster_whisper`` / ``sensevoice`` / ``cloud`` 等。
        language: 目标语言（BCP-47 或裸语种码）。
        session_id: 会话 id（也用作 mock 引擎的确定性种子）。
        turn_id: 轮次 id。
        hotwords: 热词表。
        settings: 感知层配置。
        offline: 离线模式。
        **engine_kwargs: 引擎构造参数。

    Returns:
        ``(引擎实例, 路由决策)``。永远不会抛异常，最差返回 mock 引擎。
    """
    router = create_asr_router(settings, engine=engine, offline=offline, **engine_kwargs)
    return router.create(language=language, session_id=session_id,
                         turn_id=turn_id, hotwords=hotwords)

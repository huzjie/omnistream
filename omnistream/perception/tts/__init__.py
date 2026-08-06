"""流式语音合成子系统。

对外只暴露两个入口：

* :func:`create_tts` —— 读配置，返回一个**一定能用**的引擎实例；
* :class:`TTSSession` —— 把拉模式引擎适配成"随时推文本、随时取音频"的推模式。

其余（方言、缓存、音色表、切句器）都是可选的进阶用法。

Example::

    from omnistream.perception.tts import TTSSession, create_tts

    tts, route = create_tts(language="zh-CN", voice="xiaoyan")
    session = TTSSession(tts)
    await session.start()
    await session.push_text("你好，")
    await session.push_text("今天天气不错。")
    async for chunk in session.audio():
        await playback.enqueue(chunk)
    await session.finish()
"""
from __future__ import annotations

from typing import Any

import structlog

from omnistream.perception.tts.base import (
    DEFAULT_TTS_SAMPLE_RATE,
    TTS_FRAME_MS,
    BaseStreamingTTS,
    SynthesisRequest,
    TTSCapabilities,
    TTSSession,
    TTSUnavailable,
    frame_pcm,
    pcm_to_chunks,
    resample_f32,
)
from omnistream.perception.tts.cache import (
    COMMON_PHRASES,
    CachedTTS,
    CacheEntry,
    CacheStats,
    TTSCache,
    warmup_cache,
)
from omnistream.perception.tts.cloud_tts import (
    DIALECTS,
    CloudStreamTTS,
    CloudTTSConfig,
    CloudTTSDialect,
    cloud_tts_available,
    get_tts_dialect,
)
from omnistream.perception.tts.mock_tts import MockStreamingTTS, estimate_duration_ms
from omnistream.perception.tts.piper_tts import PiperTTS, find_piper_model, piper_available
from omnistream.perception.tts.router import (
    ENGINE_PROFILES,
    FALLBACK_CHAIN,
    TTSEngineHealth,
    TTSEngineProfile,
    TTSRoute,
    TTSRouter,
)
from omnistream.perception.tts.sentencizer import (
    StreamingSentencizer,
    iter_sentences,
    split_sentences,
)
from omnistream.perception.tts.ssml import (
    AZURE_DIALECT,
    EMOTION_STYLES,
    STANDARD_DIALECT,
    Prosody,
    SSMLDialect,
    build_ssml,
    escape_text,
    extract_breaks,
    is_ssml,
    strip_ssml,
)
from omnistream.perception.tts.voice_registry import (
    DEFAULT_VOICE,
    VOICE_CATALOG,
    Voice,
    VoiceRegistry,
    default_registry,
)

log = structlog.get_logger(__name__)

__all__ = [
    "AZURE_DIALECT",
    "COMMON_PHRASES",
    "CacheEntry",
    "CacheStats",
    "CachedTTS",
    "CloudStreamTTS",
    "CloudTTSConfig",
    "CloudTTSDialect",
    "DEFAULT_TTS_SAMPLE_RATE",
    "DEFAULT_VOICE",
    "DIALECTS",
    "EMOTION_STYLES",
    "ENGINE_PROFILES",
    "FALLBACK_CHAIN",
    "BaseStreamingTTS",
    "MockStreamingTTS",
    "PiperTTS",
    "Prosody",
    "SSMLDialect",
    "STANDARD_DIALECT",
    "StreamingSentencizer",
    "SynthesisRequest",
    "TTSCache",
    "TTSCapabilities",
    "TTSEngineHealth",
    "TTSEngineProfile",
    "TTSRoute",
    "TTSRouter",
    "TTSSession",
    "TTSUnavailable",
    "TTS_FRAME_MS",
    "VOICE_CATALOG",
    "Voice",
    "VoiceRegistry",
    "build_ssml",
    "cloud_tts_available",
    "create_tts",
    "create_tts_router",
    "default_registry",
    "escape_text",
    "estimate_duration_ms",
    "extract_breaks",
    "find_piper_model",
    "frame_pcm",
    "get_tts_dialect",
    "is_ssml",
    "iter_sentences",
    "pcm_to_chunks",
    "piper_available",
    "resample_f32",
    "split_sentences",
    "strip_ssml",
    "warmup_cache",
]


def create_tts_router(
    settings: Any = None,
    *,
    offline: bool = False,
    shared_cache: TTSCache | None = None,
    **overrides: Any,
) -> TTSRouter:
    """按感知层配置构造一个 TTS 路由器。

    Args:
        settings: :class:`~omnistream.config.perception.PerceptionSettings`；
            为 ``None`` 时从全局配置读取，再失败则用默认值。
        offline: 强制离线模式（淘汰所有云端引擎）。
        shared_cache: 跨会话共享的缓存实例。
        **overrides: 覆盖路由器构造参数。

    Returns:
        路由器实例。
    """
    settings = _resolve_settings(settings)
    engine_kwargs: dict[str, Any] = {}
    if settings is not None:
        model_path = getattr(settings, "tts_model_path", None)
        if model_path:
            engine_kwargs["model_path"] = model_path
        model = getattr(settings, "tts_model", "")
        if model and model != "default":
            engine_kwargs["model"] = model
        first_chunk = getattr(settings, "tts_first_chunk_chars", 0)
        if first_chunk:
            engine_kwargs["first_chunk_chars"] = int(first_chunk)

    kwargs: dict[str, Any] = {
        "preferred": getattr(settings, "tts_engine", "mock") if settings else "mock",
        "offline": offline,
        "engine_kwargs": engine_kwargs,
        "cache_enabled": bool(getattr(settings, "tts_cache_enabled", True))
        if settings else True,
        "cache_size": int(getattr(settings, "tts_cache_size", 512)) if settings else 512,
        "shared_cache": shared_cache,
    }
    kwargs.update(overrides)
    return TTSRouter(**kwargs)


def create_tts(
    engine: str = "",
    *,
    settings: Any = None,
    language: str = "",
    voice: str = "",
    sample_rate: int = 0,
    speed: float = 0.0,
    pitch: float | None = None,
    volume_db: float | None = None,
    session_id: str = "",
    offline: bool = False,
    shared_cache: TTSCache | None = None,
    **overrides: Any,
) -> tuple[BaseStreamingTTS, TTSRoute]:
    """构造一个可用的 TTS 引擎。

    这是业务层唯一需要调用的入口：它读配置、做选型、处理依赖缺失、
    在必要时降级，**保证返回一个能出声的引擎**（最差是 mock）。

    Args:
        engine: 引擎名（``auto`` 自动选型，空串则读配置）。
        settings: 感知层配置对象，``None`` 时自动从全局配置读取。
        language: 语言标签，空串则读配置。
        voice: 逻辑音色名，空串则读配置。
        sample_rate: 输出采样率，0 则读配置。
        speed: 语速倍率，0 则读配置。
        pitch: 音高偏移（半音），``None`` 则读配置。
        volume_db: 音量增益，``None`` 则读配置。
        session_id: 会话 ID，用于日志关联。
        offline: 强制离线模式。
        shared_cache: 跨会话共享的音频缓存。
        **overrides: 透传给引擎构造函数的额外参数。

    Returns:
        ``(引擎实例, 路由结果)``。
    """
    settings = _resolve_settings(settings)
    router = create_tts_router(settings, offline=offline, shared_cache=shared_cache)
    if engine:
        router.raw_preferred = engine.strip().lower()
        router.preferred = router.raw_preferred

    resolved_language = language or (
        getattr(settings, "tts_language", "zh-CN") if settings else "zh-CN")
    resolved_voice = voice or (
        getattr(settings, "tts_voice", DEFAULT_VOICE) if settings else DEFAULT_VOICE)
    resolved_rate = sample_rate or (
        int(getattr(settings, "tts_sample_rate", DEFAULT_TTS_SAMPLE_RATE))
        if settings else DEFAULT_TTS_SAMPLE_RATE)
    resolved_speed = speed or (
        float(getattr(settings, "tts_speed", 1.0)) if settings else 1.0)
    resolved_pitch = pitch if pitch is not None else (
        float(getattr(settings, "tts_pitch", 0.0)) if settings else 0.0)
    resolved_volume = volume_db if volume_db is not None else (
        float(getattr(settings, "tts_volume_db", 0.0)) if settings else 0.0)

    return router.create(
        language=resolved_language,
        voice=resolved_voice,
        sample_rate=resolved_rate,
        speed=resolved_speed,
        pitch=resolved_pitch,
        volume_db=resolved_volume,
        session_id=session_id,
        **overrides,
    )


def _resolve_settings(settings: Any) -> Any:
    """拿到感知层配置；拿不到就返回 ``None``，由调用方用默认值。"""
    if settings is not None:
        return settings
    try:
        from omnistream.config.settings import get_settings

        root = get_settings()
        return getattr(root, "perception", None)
    except Exception as exc:  # noqa: BLE001 - 配置不可用时不阻断感知层
        log.debug("tts.settings_unavailable", error=str(exc))
        return None

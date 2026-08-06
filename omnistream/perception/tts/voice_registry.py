"""音色注册表：统一管理跨引擎的音色标识、能力与默认韵律。

痛点：Azure 叫 ``zh-CN-XiaoxiaoNeural``，piper 叫 ``zh_CN-huayan-medium``，
火山叫 ``BV700_streaming``。业务层不应该关心这些。

方案：平台内部只认**逻辑音色名**（``xiaoyan`` / ``narrator`` / ``news_male``），
由本注册表把逻辑名映射到各引擎的物理名，并携带默认韵律与元数据。
未注册的音色**原样透传**给引擎，保证不阻塞高级用户直接用物理名。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Iterable, Iterator

import structlog

from omnistream.perception.tts.ssml import Prosody

log = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_VOICE",
    "VOICE_CATALOG",
    "Voice",
    "VoiceRegistry",
    "default_registry",
]

#: 平台默认逻辑音色
DEFAULT_VOICE: Final[str] = "default"


@dataclass(slots=True)
class Voice:
    """一个逻辑音色的完整描述。"""

    #: 逻辑名，业务层唯一标识
    id: str
    #: 展示名
    display_name: str = ""
    language: str = "zh-CN"
    #: ``female`` / ``male`` / ``neutral``
    gender: str = "female"
    #: 风格标签，用于控制台筛选
    tags: tuple[str, ...] = ()
    #: 各引擎的物理音色名：``{"piper": "zh_CN-huayan-medium", ...}``
    engine_names: dict[str, str] = field(default_factory=dict)
    #: 该音色的默认韵律（有些音色天生偏快/偏低）
    prosody: Prosody = field(default_factory=Prosody)
    #: 支持的情绪
    emotions: tuple[str, ...] = ()
    #: 推荐采样率
    sample_rate: int = 24000
    description: str = ""

    def name_for(self, engine: str) -> str:
        """取该音色在指定引擎下的物理名，未映射则返回逻辑名。"""
        return self.engine_names.get(engine, self.id)

    def supports(self, engine: str) -> bool:
        """该引擎是否有此音色的映射。"""
        return engine in self.engine_names

    def as_dict(self) -> dict[str, Any]:
        """序列化（控制台音色列表接口用）。"""
        return {
            "id": self.id,
            "display_name": self.display_name or self.id,
            "language": self.language,
            "gender": self.gender,
            "tags": list(self.tags),
            "engines": sorted(self.engine_names),
            "emotions": list(self.emotions),
            "sample_rate": self.sample_rate,
            "description": self.description,
        }


#: 内置音色目录。物理名取各引擎官方公开的常用音色，未安装时由引擎自行降级。
VOICE_CATALOG: Final[tuple[Voice, ...]] = (
    Voice(
        id="default",
        display_name="默认女声",
        language="zh-CN",
        gender="female",
        tags=("general", "warm"),
        engine_names={
            "mock": "default",
            "piper": "zh_CN-huayan-medium",
            "azure": "zh-CN-XiaoxiaoNeural",
            "volc": "BV700_streaming",
            "edge": "zh-CN-XiaoxiaoNeural",
        },
        emotions=("neutral", "cheerful", "friendly"),
        description="通用中文女声，语速适中，适合客服与助理场景",
    ),
    Voice(
        id="xiaoyan",
        display_name="晓燕·亲和",
        language="zh-CN",
        gender="female",
        tags=("assistant", "soft"),
        engine_names={
            "mock": "xiaoyan",
            "piper": "zh_CN-huayan-medium",
            "azure": "zh-CN-XiaoyiNeural",
            "volc": "BV701_streaming",
        },
        prosody=Prosody(rate=1.02, pitch_st=0.5),
        emotions=("neutral", "cheerful", "empathetic"),
        description="偏亲和的女声，适合陪伴与教育场景",
    ),
    Voice(
        id="news_male",
        display_name="新闻男声",
        language="zh-CN",
        gender="male",
        tags=("news", "formal"),
        engine_names={
            "mock": "news_male",
            "azure": "zh-CN-YunxiNeural",
            "volc": "BV002_streaming",
        },
        prosody=Prosody(rate=0.98, pitch_st=-2.0),
        emotions=("neutral", "serious"),
        description="播报风格男声，吐字清晰，适合资讯播报",
    ),
    Voice(
        id="narrator",
        display_name="低沉旁白",
        language="zh-CN",
        gender="male",
        tags=("narration", "deep"),
        engine_names={
            "mock": "narrator",
            "azure": "zh-CN-YunyeNeural",
        },
        prosody=Prosody(rate=0.92, pitch_st=-4.0, volume_db=1.0),
        emotions=("neutral", "calm"),
        description="低沉缓慢的旁白音色，适合长文朗读",
    ),
    Voice(
        id="en_female",
        display_name="English Female",
        language="en-US",
        gender="female",
        tags=("general", "english"),
        engine_names={
            "mock": "en_female",
            "piper": "en_US-amy-medium",
            "azure": "en-US-JennyNeural",
        },
        emotions=("neutral", "cheerful", "friendly"),
        description="通用英文女声",
    ),
    Voice(
        id="en_male",
        display_name="English Male",
        language="en-US",
        gender="male",
        tags=("general", "english"),
        engine_names={
            "mock": "en_male",
            "piper": "en_US-ryan-high",
            "azure": "en-US-GuyNeural",
        },
        prosody=Prosody(pitch_st=-3.0),
        description="通用英文男声",
    ),
)


class VoiceRegistry:
    """音色注册表。

    Example::

        reg = VoiceRegistry()
        physical = reg.resolve("xiaoyan", engine="azure")   # zh-CN-XiaoyiNeural
        prosody = reg.prosody_of("narrator")
    """

    def __init__(self, voices: Iterable[Voice] | None = None) -> None:
        self._voices: dict[str, Voice] = {}
        for voice in voices if voices is not None else VOICE_CATALOG:
            self.register(voice)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._voices)

    def __contains__(self, voice_id: str) -> bool:
        return voice_id in self._voices

    def __iter__(self) -> Iterator[Voice]:
        return iter(self._voices.values())

    def register(self, voice: Voice, *, override: bool = True) -> bool:
        """注册（或覆盖）一个音色。

        Args:
            voice: 音色描述。
            override: 已存在时是否覆盖。

        Returns:
            是否写入成功。
        """
        if not voice.id:
            return False
        if voice.id in self._voices and not override:
            return False
        self._voices[voice.id] = voice
        return True

    def get(self, voice_id: str) -> Voice | None:
        """按逻辑名取音色。"""
        return self._voices.get(voice_id)

    def resolve(self, voice_id: str, *, engine: str) -> str:
        """把逻辑音色名翻译成引擎物理名。

        未注册的音色原样返回（允许高级用户直接写物理名）。

        Args:
            voice_id: 逻辑名或物理名。
            engine: 目标引擎名。

        Returns:
            物理音色名。
        """
        voice = self._voices.get(voice_id)
        if voice is None:
            return voice_id or DEFAULT_VOICE
        physical = voice.name_for(engine)
        if not voice.supports(engine):
            log.debug("tts.voice_unmapped", voice=voice_id, engine=engine,
                      fallback=physical)
        return physical

    def prosody_of(self, voice_id: str) -> Prosody:
        """取音色的默认韵律。"""
        voice = self._voices.get(voice_id)
        return voice.prosody if voice is not None else Prosody()

    def for_language(self, language: str) -> list[Voice]:
        """按语言筛选音色。"""
        code = (language or "").split("-")[0].lower()
        return [v for v in self._voices.values()
                if v.language.split("-")[0].lower() == code]

    def for_engine(self, engine: str) -> list[Voice]:
        """列出某引擎有映射的音色。"""
        return [v for v in self._voices.values() if v.supports(engine)]

    def pick_default(self, *, language: str = "zh-CN", engine: str = "") -> Voice:
        """为指定语言挑一个默认音色。

        Args:
            language: 目标语言。
            engine: 目标引擎；给出时优先挑该引擎支持的音色。

        Returns:
            音色描述；无候选时返回目录第一个。
        """
        candidates = self.for_language(language)
        if engine:
            mapped = [v for v in candidates if v.supports(engine)]
            if mapped:
                return mapped[0]
        if candidates:
            return candidates[0]
        fallback = self._voices.get(DEFAULT_VOICE)
        return fallback if fallback is not None else next(iter(self._voices.values()))

    def catalog(self) -> list[dict[str, Any]]:
        """全量音色目录（控制台接口用）。"""
        return [v.as_dict() for v in self._voices.values()]

    def __repr__(self) -> str:
        return f"<VoiceRegistry n={len(self._voices)}>"


#: 进程级默认注册表
default_registry: Final[VoiceRegistry] = VoiceRegistry()

"""SenseVoice / FunASR 适配器（可选依赖，缺失自动降级）。

SenseVoice 是达摩院开源的多语种语音理解模型，相比 whisper 的优势：

* **中文识别率更高**，且原生支持中英日韩粤五语；
* **推理更快**（非自回归），CPU 上 RTF 约 0.05，适合边缘部署；
* 附带 **情绪 / 音频事件标签**（``<|HAPPY|>`` ``<|Laughter|>``），
  本模块把这些富文本标签解析出来放进 ``details``，供情感 TTS 与质检使用。

同样是非流式模型，因此复用与 whisper 相同的 **滑窗 + 稳定前缀** 策略；
差别在于：SenseVoice 原生支持 ``hotword`` 参数（真·解码期偏置），
所以这里不走后处理纠偏，而是把热词直接传给引擎。
"""
from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator
from typing import Any, Final, Iterable

import numpy as np
import structlog

from omnistream.perception.asr.base import (
    ASR_SAMPLE_RATE,
    ASRCapabilities,
    ASRUnavailable,
    BaseStreamingASR,
    build_delta,
)
from omnistream.perception.asr.hotwords import HotwordBias
from omnistream.perception.asr.punctuation import PunctuationRestorer
from omnistream.perception.asr.stabilizer import LocalAgreementStabilizer
from omnistream.perception.asr.word_timing import interpolate_words
from omnistream.perception.errors import ASRError
from omnistream.schemas.audio import AudioChunk
from omnistream.schemas.transcript import TranscriptDelta

log = structlog.get_logger(__name__)

__all__ = [
    "SenseVoiceASR",
    "RichTags",
    "parse_rich_transcript",
    "sensevoice_available",
]

try:  # pragma: no cover - 依赖是否存在取决于部署环境
    from funasr import AutoModel as _AutoModel  # type: ignore

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    _AutoModel = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc

#: 默认模型 id（首次运行会自动从 ModelScope 拉取）
DEFAULT_MODEL: Final[str] = "iic/SenseVoiceSmall"

#: 解码窗口上限（秒）
DEFAULT_WINDOW_S: Final[float] = 10.0

#: 静音超过该值判定一句话结束
DEFAULT_UTTERANCE_SILENCE_MS: Final[float] = 620.0

_SILENCE_DBFS: Final[float] = -45.0

#: 富文本标签形如 ``<|zh|><|HAPPY|><|Speech|><|withitn|>``
_TAG_RE: Final[re.Pattern[str]] = re.compile(r"<\|([^|>]+)\|>")

#: 情绪标签集合
_EMOTIONS: Final[frozenset[str]] = frozenset({
    "HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED",
})

#: 音频事件标签集合
_EVENTS: Final[frozenset[str]] = frozenset({
    "Speech", "BGM", "Applause", "Laughter", "Cry", "Sneeze", "Breath", "Cough",
})

#: 语种标签集合
_LANGS: Final[frozenset[str]] = frozenset({"zh", "en", "yue", "ja", "ko", "nospeech"})


class RichTags:
    """从 SenseVoice 富文本中解析出的结构化标签。"""

    __slots__ = ("language", "emotion", "events", "raw_tags")

    def __init__(self, language: str = "", emotion: str = "",
                 events: tuple[str, ...] = (), raw_tags: tuple[str, ...] = ()) -> None:
        self.language = language
        self.emotion = emotion
        self.events = events
        self.raw_tags = raw_tags

    def as_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "emotion": self.emotion,
            "events": list(self.events),
            "raw_tags": list(self.raw_tags),
        }

    def __repr__(self) -> str:
        return f"<RichTags lang={self.language} emotion={self.emotion} events={self.events}>"


def parse_rich_transcript(raw: str) -> tuple[str, RichTags]:
    """把 ``<|zh|><|HAPPY|>你好`` 拆成 ``("你好", RichTags(...))``。

    Args:
        raw: SenseVoice 原始输出。

    Returns:
        ``(纯文本, 标签)``。
    """
    if not raw:
        return "", RichTags()
    tags = tuple(_TAG_RE.findall(raw))
    text = _TAG_RE.sub("", raw).strip()
    language = next((t for t in tags if t in _LANGS), "")
    emotion = next((t for t in tags if t in _EMOTIONS), "")
    events = tuple(t for t in tags if t in _EVENTS)
    return text, RichTags(language=language, emotion=emotion,
                          events=events, raw_tags=tags)


def sensevoice_available() -> bool:
    """依赖是否可用。"""
    return _AutoModel is not None


def _dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return 20.0 * float(np.log10(rms)) if rms > 1e-9 else -120.0


class SenseVoiceASR(BaseStreamingASR):
    """基于 FunASR / SenseVoice 的伪流式识别器。

    Raises:
        ASRUnavailable: 构造时 ``funasr`` 未安装。
    """

    name = "sensevoice"
    capabilities = ASRCapabilities(
        streaming=True,
        word_timestamps=False,   # SenseVoice 不给词级时间戳，用插值补
        language_detection=True,
        hotwords=True,
        punctuation=True,
        needs_gpu=False,
        needs_network=False,
        typical_partial_ms=400,
        languages=("zh", "yue", "en", "ja", "ko"),
    )

    def __init__(
        self,
        *,
        language: str = "zh-CN",
        sample_rate: int = ASR_SAMPLE_RATE,
        turn_id: str = "",
        session_id: str = "",
        hotwords: Iterable[str] | None = None,
        enable_punctuation: bool = True,
        enable_word_timestamps: bool = False,
        model: str = DEFAULT_MODEL,
        model_path: str | None = None,
        device: str = "auto",
        stride_ms: int = 400,
        window_s: float = DEFAULT_WINDOW_S,
        use_itn: bool = True,
        engine: Any = None,
    ) -> None:
        """
        Args:
            model: 模型 id 或本地目录。
            model_path: 本地权重目录，优先于 ``model``。
            device: ``auto|cpu|cuda:0``。
            stride_ms: 解码步进。
            window_s: 窗口上限。
            use_itn: 逆文本归一化（把"二零二五"变成"2025"）。
            engine: 预构造的 ``AutoModel``（便于注入与复用）。

        Raises:
            ASRUnavailable: 依赖缺失。
        """
        super().__init__(
            language=language,
            sample_rate=sample_rate,
            turn_id=turn_id,
            session_id=session_id,
            hotwords=hotwords,
            enable_punctuation=enable_punctuation,
            enable_word_timestamps=enable_word_timestamps,
        )
        if engine is None and _AutoModel is None:
            raise ASRUnavailable(f"funasr 不可用: {_IMPORT_ERROR}")

        self.model = model
        self.model_path = model_path
        self.device = device
        self.stride_ms = max(120, int(stride_ms))
        self.window_s = float(window_s)
        self.use_itn = bool(use_itn)

        self._engine = engine
        self._lock = asyncio.Lock()
        self._bias = HotwordBias(self.hotwords)
        self._punct = PunctuationRestorer(language=language)
        self._stabilizer = LocalAgreementStabilizer(agreement=2)

        self._buffer = np.zeros(0, dtype=np.float32)
        self._elapsed_ms = 0.0
        self._segment_start_ms = 0.0
        self._pending_ms = 0.0
        self._silence_ms = 0.0
        self._delta_index = 0
        self.last_tags = RichTags()
        self.decode_calls = 0
        self.decode_ms_total = 0.0

    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        return sensevoice_available()

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:  # pragma: no cover
            import torch  # type: ignore

            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    def _build_engine(self) -> Any:
        if _AutoModel is None:  # pragma: no cover
            raise ASRUnavailable(f"funasr 不可用: {_IMPORT_ERROR}")
        target = self.model_path or self.model
        device = self._resolve_device()
        log.info("asr.sensevoice_loading", model=target, device=device)
        return _AutoModel(model=target, device=device, disable_update=True)

    async def _ensure_engine(self) -> Any:
        if self._engine is None:
            async with self._lock:
                if self._engine is None:
                    self._engine = await asyncio.to_thread(self._build_engine)
        return self._engine

    def _lang_code(self) -> str:
        code = (self.language or "").split("-")[0].lower()
        return code if code in _LANGS else "auto"

    # ------------------------------------------------------------------

    def _decode_sync(self, audio: np.ndarray) -> tuple[str, RichTags]:
        engine = self._engine
        if engine is None:  # pragma: no cover
            raise ASRError("sensevoice 引擎未加载")
        kwargs: dict[str, Any] = {
            "input": audio,
            "cache": {},
            "language": self._lang_code(),
            "use_itn": self.use_itn,
            "batch_size_s": 60,
        }
        hotword_list = self._bias.as_hotword_list()
        if hotword_list:
            kwargs["hotword"] = " ".join(hotword_list)
        result = engine.generate(**kwargs)
        raw = _extract_text(result)
        return parse_rich_transcript(raw)

    async def _decode(self) -> tuple[str, RichTags]:
        await self._ensure_engine()
        audio = self._buffer.copy()
        if audio.size < self.sample_rate // 20:
            return "", RichTags()
        started = time.perf_counter()
        async with self._lock:
            try:
                text, tags = await asyncio.to_thread(self._decode_sync, audio)
            except Exception as exc:  # noqa: BLE001
                raise ASRError(f"sensevoice 解码失败: {exc}", cause=exc) from exc
        self.decode_calls += 1
        self.decode_ms_total += (time.perf_counter() - started) * 1000.0
        if tags.raw_tags:
            self.last_tags = tags
        return text, tags

    def _append(self, chunk: AudioChunk) -> None:
        try:
            samples = np.asarray(chunk.to_ndarray(), dtype=np.float32).reshape(-1)
        except ValueError as exc:
            raise ASRError(f"ASR 只接受 PCM 输入，收到 {chunk.encoding}", cause=exc) from exc
        if samples.size == 0:
            return
        if chunk.sample_rate != self.sample_rate:
            target = max(1, int(round(samples.size * self.sample_rate / chunk.sample_rate)))
            samples = np.interp(
                np.linspace(0.0, samples.size - 1, num=target, dtype=np.float64),
                np.arange(samples.size, dtype=np.float64), samples).astype(np.float32)
        self._buffer = np.concatenate([self._buffer, samples])
        span_ms = samples.size * 1000.0 / self.sample_rate
        self._elapsed_ms += span_ms
        self._pending_ms += span_ms
        if _dbfs(samples) < _SILENCE_DBFS:
            self._silence_ms += span_ms
        else:
            self._silence_ms = 0.0
        max_samples = int(self.window_s * self.sample_rate)
        if self._buffer.size > max_samples:
            self._buffer = self._buffer[self._buffer.size - max_samples:]

    def _make_delta(self, text: str, *, is_final: bool,
                    stability: float) -> TranscriptDelta:
        rendered = text
        if self.enable_punctuation and text:
            rendered = self._punct.restore(text, is_final=is_final)
        words = []
        if self.enable_word_timestamps and rendered:
            words = [s.to_model() for s in interpolate_words(
                rendered, int(self._segment_start_ms), int(self._elapsed_ms))]
        language = self.language
        if self.last_tags.language and self.last_tags.language != "nospeech":
            language = self.last_tags.language
        delta = build_delta(
            turn_id=self.turn_id,
            text=rendered,
            is_final=is_final,
            stability=stability,
            start_ms=int(self._segment_start_ms),
            end_ms=int(self._elapsed_ms),
            speaker="user",
            words=words,
            language=language,
            engine=self.name,
        )
        try:
            delta.index = self._delta_index
        except (AttributeError, ValueError):
            pass
        self._delta_index += 1
        return delta

    def _reset_segment(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._segment_start_ms = self._elapsed_ms
        self._pending_ms = 0.0
        self._delta_index = 0
        self._stabilizer.reset()

    # ------------------------------------------------------------------

    async def stream(self, chunks: AsyncIterator[AudioChunk]) -> AsyncIterator[TranscriptDelta]:
        """滑窗解码 + 稳定前缀提交。"""
        last_text = ""
        emitted_any = False
        async for chunk in chunks:
            self._append(chunk)
            end_of_utterance = self._silence_ms >= DEFAULT_UTTERANCE_SILENCE_MS
            if not end_of_utterance and self._pending_ms < self.stride_ms:
                continue
            self._pending_ms = 0.0

            text, _tags = await self._decode()
            if text:
                last_text = text

            if end_of_utterance:
                if last_text:
                    yield self._make_delta(last_text, is_final=True, stability=1.0)
                    emitted_any = True
                last_text = ""
                self._reset_segment()
                self._silence_ms = 0.0
                continue

            if not text:
                continue
            stabilized = self._stabilizer.update(text)
            yield self._make_delta(stabilized.text, is_final=False,
                                   stability=stabilized.stability)
            emitted_any = True

        if last_text or not emitted_any:
            if not last_text and self._buffer.size:
                last_text, _ = await self._decode()
            yield self._make_delta(last_text, is_final=True, stability=1.0)
        self._reset_segment()

    async def reset(self) -> None:
        """清空解码状态。"""
        self._reset_segment()
        self._elapsed_ms = 0.0
        self._segment_start_ms = 0.0
        self._silence_ms = 0.0
        self.last_tags = RichTags()
        self._punct.reset()

    async def finalize(self) -> TranscriptDelta | None:
        """强制解码剩余缓冲。"""
        if self._buffer.size == 0:
            return None
        text, _ = await self._decode()
        if not text:
            self._reset_segment()
            return None
        delta = self._make_delta(text, is_final=True, stability=1.0)
        self._reset_segment()
        return delta

    def set_hotwords(self, words: Iterable[str]) -> None:
        """更新热词表（SenseVoice 原生支持解码期偏置）。"""
        super().set_hotwords(words)
        self._bias = HotwordBias(self.hotwords)

    async def aclose(self) -> None:
        """释放引擎引用。"""
        self._engine = None
        self._buffer = np.zeros(0, dtype=np.float32)

    def stats(self) -> dict[str, Any]:
        data = super().stats()
        avg = self.decode_ms_total / self.decode_calls if self.decode_calls else 0.0
        data.update({
            "decode_calls": self.decode_calls,
            "decode_avg_ms": round(avg, 1),
            "engine_loaded": self._engine is not None,
            "tags": self.last_tags.as_dict(),
        })
        return data


def _extract_text(result: Any) -> str:
    """从 funasr 返回结构里取出富文本。"""
    if isinstance(result, str):
        return result
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return str(first.get("text", ""))
        return str(first)
    if isinstance(result, dict):
        return str(result.get("text", ""))
    return ""

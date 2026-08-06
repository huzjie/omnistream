"""faster-whisper 流式适配器（可选依赖，缺失自动降级）。

Whisper 本身是**离线整段模型**，不是流式模型。要把它变成能出 partial 的
实时引擎，业界标准做法是 **滑动窗口 + 稳定前缀提交**（LocalAgreement，
即 whisper_streaming 的核心思路），本模块完整实现了它：

1. 维护一个最长 ``window_s`` 秒的音频缓冲；
2. 每积累 ``stride_ms`` 就把整个窗口丢给 whisper 解码一次（线程池，不阻塞事件循环）；
3. 把本次假设与上一次假设求**最长公共前缀**，公共部分视为稳定 → 提交；
4. 当已提交文本对应的音频超过 ``trim_after_s``，把窗口左端裁掉，
   避免窗口无限增长导致解码越来越慢。

并发保护：单实例内用 ``asyncio.Lock`` 串行化解码，
多会话之间由 :mod:`omnistream.perception.asr.router` 的信号量限流。
"""
from __future__ import annotations

import asyncio
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
from omnistream.perception.asr.word_timing import WordSpan, interpolate_words
from omnistream.perception.errors import ASRError
from omnistream.schemas.audio import AudioChunk
from omnistream.schemas.transcript import TranscriptDelta

log = structlog.get_logger(__name__)

__all__ = ["FasterWhisperASR", "faster_whisper_available"]

try:  # pragma: no cover - 依赖是否存在取决于部署环境
    from faster_whisper import WhisperModel as _WhisperModel  # type: ignore

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    _WhisperModel = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc

#: 解码窗口上限（秒）。越大精度越高、延迟越大。
DEFAULT_WINDOW_S: Final[float] = 12.0

#: 已提交音频超过该时长就裁窗
DEFAULT_TRIM_AFTER_S: Final[float] = 8.0

#: 静音超过该值判定一句话结束
DEFAULT_UTTERANCE_SILENCE_MS: Final[float] = 640.0

#: 静音判定门限
_SILENCE_DBFS: Final[float] = -45.0


def faster_whisper_available() -> bool:
    """依赖是否可用。"""
    return _WhisperModel is not None


def _dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return 20.0 * float(np.log10(rms)) if rms > 1e-9 else -120.0


class FasterWhisperASR(BaseStreamingASR):
    """基于 faster-whisper 的伪流式识别器。

    Raises:
        ASRUnavailable: 构造时 ``faster_whisper`` 未安装。
    """

    name = "faster_whisper"
    capabilities = ASRCapabilities(
        streaming=True,
        word_timestamps=True,
        language_detection=True,
        hotwords=True,          # 通过 initial_prompt 软偏置
        punctuation=True,
        needs_gpu=False,        # CPU int8 也能跑 small
        needs_network=False,
        typical_partial_ms=600,
        languages=("zh", "en", "ja", "ko", "ru", "de", "fr", "es"),
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
        model_size: str = "small",
        model_path: str | None = None,
        device: str = "auto",
        compute_type: str = "int8",
        beam_size: int = 1,
        stride_ms: int = 480,
        window_s: float = DEFAULT_WINDOW_S,
        trim_after_s: float = DEFAULT_TRIM_AFTER_S,
        vad_filter: bool = True,
        model: Any = None,
    ) -> None:
        """
        Args:
            model_size: ``tiny|base|small|medium|large-v3``。
            model_path: 本地权重目录；给出时优先于 ``model_size``。
            device: ``auto|cpu|cuda``。
            compute_type: ``int8|int8_float16|float16|float32``。
            stride_ms: 两次解码之间的音频步进，决定 partial 频率。
            window_s: 解码窗口上限。
            trim_after_s: 已提交音频超过该值就裁窗。
            vad_filter: 是否启用 whisper 内置 VAD 过滤。
            model: 预先构造好的 ``WhisperModel``（便于复用与注入）。

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
        if model is None and _WhisperModel is None:
            raise ASRUnavailable(f"faster_whisper 不可用: {_IMPORT_ERROR}")

        self.model_size = model_size
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.beam_size = max(1, int(beam_size))
        self.stride_ms = max(120, int(stride_ms))
        self.window_s = float(window_s)
        self.trim_after_s = float(trim_after_s)
        self.vad_filter = bool(vad_filter)

        self._model = model
        self._model_lock = asyncio.Lock()
        self._bias = HotwordBias(self.hotwords)
        self._punct = PunctuationRestorer(language=language)
        self._stabilizer = LocalAgreementStabilizer(agreement=2)

        # 解码状态
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_origin_ms = 0.0      # 缓冲左端对应的绝对时间
        self._elapsed_ms = 0.0            # 已吸收音频的总时长
        self._pending_ms = 0.0            # 距上次解码累积的音频
        self._silence_ms = 0.0
        self._committed = ""
        self._segment_start_ms = 0.0
        self._delta_index = 0
        self.decode_calls = 0
        self.decode_ms_total = 0.0

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        return faster_whisper_available()

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:  # pragma: no cover - 取决于运行环境
            import torch  # type: ignore

            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    def _build_model(self) -> Any:
        """在线程池中构造模型（首次调用可能耗时数秒）。"""
        if _WhisperModel is None:  # pragma: no cover - 已在 __init__ 拦截
            raise ASRUnavailable(f"faster_whisper 不可用: {_IMPORT_ERROR}")
        target = self.model_path or self.model_size
        device = self._resolve_device()
        compute = self.compute_type
        if device == "cpu" and compute in ("float16", "int8_float16"):
            compute = "int8"  # CPU 上 fp16 不可用，静默降级
        log.info("asr.whisper_loading", model=target, device=device, compute=compute)
        return _WhisperModel(target, device=device, compute_type=compute)

    async def _ensure_model(self) -> Any:
        if self._model is None:
            async with self._model_lock:
                if self._model is None:
                    self._model = await asyncio.to_thread(self._build_model)
        return self._model

    # ------------------------------------------------------------------
    # 解码
    # ------------------------------------------------------------------

    def _whisper_language(self) -> str | None:
        code = (self.language or "").split("-")[0].lower()
        return code or None

    def _decode_sync(self, audio: np.ndarray) -> tuple[str, list[WordSpan]]:
        """同步解码一段音频，返回 ``(文本, 词级时间戳)``。"""
        model = self._model
        if model is None:  # pragma: no cover - 调用方已保证
            raise ASRError("whisper 模型未加载")
        prompt = self._bias.as_initial_prompt() or None
        segments, _info = model.transcribe(
            audio,
            language=self._whisper_language(),
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            word_timestamps=self.enable_word_timestamps,
            initial_prompt=prompt,
            condition_on_previous_text=False,
        )
        texts: list[str] = []
        spans: list[WordSpan] = []
        base = self._buffer_origin_ms
        for seg in segments:
            seg_text = str(getattr(seg, "text", "") or "")
            if seg_text:
                texts.append(seg_text)
            words = getattr(seg, "words", None) or []
            for w in words:
                word = str(getattr(w, "word", "") or "").strip()
                if not word:
                    continue
                spans.append(WordSpan(
                    word=word,
                    start_ms=int(base + float(getattr(w, "start", 0.0)) * 1000.0),
                    end_ms=int(base + float(getattr(w, "end", 0.0)) * 1000.0),
                    confidence=float(getattr(w, "probability", 0.9) or 0.9),
                ))
        return "".join(texts).strip(), spans

    async def _decode(self) -> tuple[str, list[WordSpan]]:
        """异步解码当前窗口（线程池 + 单实例串行）。"""
        await self._ensure_model()
        audio = self._buffer.copy()
        if audio.size < self.sample_rate // 20:  # 不足 50ms 不解码
            return "", []
        started = time.perf_counter()
        async with self._model_lock:
            try:
                text, spans = await asyncio.to_thread(self._decode_sync, audio)
            except Exception as exc:  # noqa: BLE001
                raise ASRError(f"whisper 解码失败: {exc}", cause=exc) from exc
        cost = (time.perf_counter() - started) * 1000.0
        self.decode_calls += 1
        self.decode_ms_total += cost
        if cost > self.stride_ms * 2:
            log.warning("asr.whisper_slow", cost_ms=round(cost, 1),
                        stride_ms=self.stride_ms, window_s=round(audio.size / self.sample_rate, 2))
        return text, spans

    # ------------------------------------------------------------------
    # 缓冲管理
    # ------------------------------------------------------------------

    def _append(self, chunk: AudioChunk) -> None:
        try:
            samples = np.asarray(chunk.to_ndarray(), dtype=np.float32).reshape(-1)
        except ValueError as exc:
            raise ASRError(f"ASR 只接受 PCM 输入，收到 {chunk.encoding}", cause=exc) from exc
        if samples.size == 0:
            return
        if chunk.sample_rate != self.sample_rate:
            samples = _resample(samples, chunk.sample_rate, self.sample_rate)
        self._buffer = np.concatenate([self._buffer, samples])
        span_ms = samples.size * 1000.0 / self.sample_rate
        self._elapsed_ms += span_ms
        self._pending_ms += span_ms
        if _dbfs(samples) < _SILENCE_DBFS:
            self._silence_ms += span_ms
        else:
            self._silence_ms = 0.0
        # 窗口硬上限保护
        max_samples = int(self.window_s * self.sample_rate)
        if self._buffer.size > max_samples:
            drop = self._buffer.size - max_samples
            self._buffer = self._buffer[drop:]
            self._buffer_origin_ms += drop * 1000.0 / self.sample_rate

    def _trim_committed(self) -> None:
        """已提交部分足够长时裁掉窗口左端。"""
        committed_ms = self._elapsed_ms - self._buffer_origin_ms
        if committed_ms < self.trim_after_s * 1000.0:
            return
        keep_ms = self.trim_after_s * 1000.0 * 0.5
        drop_ms = max(0.0, committed_ms - keep_ms)
        drop_samples = int(drop_ms * self.sample_rate / 1000.0)
        if drop_samples <= 0 or drop_samples >= self._buffer.size:
            return
        self._buffer = self._buffer[drop_samples:]
        self._buffer_origin_ms += drop_samples * 1000.0 / self.sample_rate

    def _make_delta(self, text: str, spans: list[WordSpan], *,
                    is_final: bool, stability: float) -> TranscriptDelta:
        rendered = text
        if self.enable_punctuation and text:
            rendered = self._punct.restore(text, is_final=is_final)
        words = []
        if self.enable_word_timestamps:
            words = [s.to_model() for s in (spans or interpolate_words(
                rendered, int(self._segment_start_ms), int(self._elapsed_ms)))]
        delta = build_delta(
            turn_id=self.turn_id,
            text=rendered,
            is_final=is_final,
            stability=stability,
            start_ms=int(self._segment_start_ms),
            end_ms=int(self._elapsed_ms),
            speaker="user",
            words=words,
            language=self.language,
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
        self._buffer_origin_ms = self._elapsed_ms
        self._segment_start_ms = self._elapsed_ms
        self._committed = ""
        self._pending_ms = 0.0
        self._delta_index = 0
        self._stabilizer.reset()

    # ------------------------------------------------------------------
    # BaseStreamingASR 接口
    # ------------------------------------------------------------------

    async def stream(self, chunks: AsyncIterator[AudioChunk]) -> AsyncIterator[TranscriptDelta]:
        """滑窗解码 + 稳定前缀提交。"""
        last_text = ""
        last_spans: list[WordSpan] = []
        emitted_any = False

        async for chunk in chunks:
            self._append(chunk)

            end_of_utterance = self._silence_ms >= DEFAULT_UTTERANCE_SILENCE_MS
            if not end_of_utterance and self._pending_ms < self.stride_ms:
                continue
            self._pending_ms = 0.0

            text, spans = await self._decode()
            if text:
                last_text, last_spans = text, spans

            if end_of_utterance:
                if last_text:
                    corrected, _ = self._bias.correct(last_text)
                    yield self._make_delta(corrected, last_spans,
                                           is_final=True, stability=1.0)
                    emitted_any = True
                last_text, last_spans = "", []
                self._reset_segment()
                self._silence_ms = 0.0
                continue

            if not text:
                continue
            stabilized = self._stabilizer.update(text)
            corrected, _ = self._bias.correct(stabilized.text)
            self._committed = stabilized.committed
            yield self._make_delta(corrected, spans, is_final=False,
                                   stability=stabilized.stability)
            emitted_any = True
            self._trim_committed()

        # 收尾 final
        if last_text or not emitted_any:
            if not last_text and self._buffer.size:
                last_text, last_spans = await self._decode()
            corrected, _ = self._bias.correct(last_text)
            yield self._make_delta(corrected, last_spans, is_final=True, stability=1.0)
        self._reset_segment()

    async def reset(self) -> None:
        """清空解码状态（保留已加载模型）。"""
        self._reset_segment()
        self._elapsed_ms = 0.0
        self._buffer_origin_ms = 0.0
        self._segment_start_ms = 0.0
        self._silence_ms = 0.0
        self._punct.reset()

    async def finalize(self) -> TranscriptDelta | None:
        """强制解码剩余缓冲并返回最终结果。"""
        if self._buffer.size == 0:
            return None
        text, spans = await self._decode()
        if not text:
            self._reset_segment()
            return None
        corrected, _ = self._bias.correct(text)
        delta = self._make_delta(corrected, spans, is_final=True, stability=1.0)
        self._reset_segment()
        return delta

    def set_hotwords(self, words: Iterable[str]) -> None:
        """更新热词表（同时更新 initial_prompt）。"""
        super().set_hotwords(words)
        self._bias = HotwordBias(self.hotwords)

    async def aclose(self) -> None:
        """释放模型引用。"""
        self._model = None
        self._buffer = np.zeros(0, dtype=np.float32)

    def stats(self) -> dict[str, Any]:
        data = super().stats()
        avg = self.decode_ms_total / self.decode_calls if self.decode_calls else 0.0
        data.update({
            "decode_calls": self.decode_calls,
            "decode_avg_ms": round(avg, 1),
            "window_samples": int(self._buffer.size),
            "model_loaded": self._model is not None,
        })
        return data


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """线性重采样（ASR 前处理够用，媒体层有更高质量的实现）。"""
    if src_rate == dst_rate or samples.size == 0:
        return samples
    target_len = max(1, int(round(samples.size * dst_rate / src_rate)))
    src_idx = np.linspace(0.0, samples.size - 1, num=target_len, dtype=np.float64)
    return np.interp(src_idx, np.arange(samples.size, dtype=np.float64),
                     samples).astype(np.float32)

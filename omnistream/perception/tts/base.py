"""流式语音合成抽象接口。

冻结契约（§2.20）::

    class BaseStreamingTTS:
        async def synthesize(text_stream) -> AsyncIterator[AudioChunk]
        async def cancel() -> None

设计要点
--------
1. **首包最快**（TTFA）：``synthesize`` 接收的是**文本流**而非整段文本，
   实现必须在第一个完整小句到手时就开始出音频，不等 LLM 说完。
   切句由 :mod:`omnistream.perception.tts.sentencizer` 负责。
2. **可即时取消**：barge-in 时 :meth:`BaseStreamingTTS.cancel` 必须让
   ``synthesize`` 在下一帧边界停止吐音频（<50ms），且不抛异常。
3. **pts 连续**：输出 ``AudioChunk.pts_ms`` 从 0 开始按帧递增，
   这是播放队列 ``cancel_from(pts)`` 的定位依据，绝不能乱序或跳变。
4. **定长帧**：统一切成 ``frame_ms``（默认 20ms），
   让播放队列的撤销粒度与媒体层一致。
"""
from __future__ import annotations

import abc
import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import structlog

from omnistream.core.constants import DEFAULT_FRAME_MS, DEFAULT_OUTPUT_SAMPLE_RATE
from omnistream.core.streams import BoundedStream, OverflowPolicy
from omnistream.perception.errors import TTSError
from omnistream.schemas.audio import AudioChunk

log = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_TTS_SAMPLE_RATE",
    "TTS_FRAME_MS",
    "BaseStreamingTTS",
    "SynthesisRequest",
    "TTSCapabilities",
    "TTSSession",
    "TTSUnavailable",
    "frame_pcm",
    "pcm_to_chunks",
    "resample_f32",
]

#: TTS 输出默认采样率（与 §常量 DEFAULT_OUTPUT_SAMPLE_RATE 一致）
DEFAULT_TTS_SAMPLE_RATE: Final[int] = DEFAULT_OUTPUT_SAMPLE_RATE

#: 输出帧长（毫秒），与播放队列撤销粒度一致
TTS_FRAME_MS: Final[int] = DEFAULT_FRAME_MS

#: 文本流内部队列深度
_TEXT_QUEUE: Final[int] = 128

#: 音频输出队列深度（20ms × 400 = 8s）
_AUDIO_QUEUE: Final[int] = 400


class TTSUnavailable(RuntimeError):
    """引擎依赖缺失时由构造函数抛出，供工厂降级捕获。"""


@dataclass(slots=True)
class TTSCapabilities:
    """引擎能力声明，路由器据此选型。"""

    streaming: bool = True
    ssml: bool = False
    voice_cloning: bool = False
    emotion: bool = False
    #: 支持的音色数量（0 表示单一音色）
    voices: int = 1
    needs_gpu: bool = False
    needs_network: bool = False
    #: 典型首包音频延迟（毫秒），决定 TTFA
    typical_ttfa_ms: int = 200
    languages: tuple[str, ...] = ("zh", "en")
    sample_rates: tuple[int, ...] = (16000, 24000)


@dataclass(slots=True)
class SynthesisRequest:
    """一次合成请求的参数快照。"""

    text: str = ""
    voice: str = "default"
    language: str = "zh-CN"
    speed: float = 1.0
    pitch: float = 0.0
    volume_db: float = 0.0
    emotion: str = ""
    #: 该句在整轮回复中的序号，用于日志与缓存键
    index: int = 0
    #: 该句起始 pts（由会话累计），实现不应自己维护全局 pts
    pts_offset_ms: int = 0

    def cache_key(self) -> str:
        """稳定的缓存键（不含 pts，因为同样的文本在任何位置都该复用）。"""
        return (f"{self.language}|{self.voice}|{self.speed:.2f}|{self.pitch:.1f}"
                f"|{self.volume_db:.1f}|{self.emotion}|{self.text}")


def frame_pcm(samples: np.ndarray, sample_rate: int,
              frame_ms: int = TTS_FRAME_MS) -> list[np.ndarray]:
    """把连续 PCM 切成定长帧，末帧补零对齐。

    Args:
        samples: 单声道 float32 [-1,1]。
        sample_rate: 采样率。
        frame_ms: 帧长（毫秒）。

    Returns:
        等长帧列表；输入为空时返回空列表。
    """
    if samples.size == 0:
        return []
    per_frame = max(1, int(sample_rate * frame_ms / 1000))
    total = int(np.ceil(samples.size / per_frame)) * per_frame
    if total > samples.size:
        samples = np.concatenate([samples, np.zeros(total - samples.size, dtype=np.float32)])
    return [samples[i:i + per_frame] for i in range(0, samples.size, per_frame)]


def resample_f32(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """把 float32 PCM 从 ``src_rate`` 重采样到 ``dst_rate``。

    优先复用媒体层的多相/soxr 重采样器（质量更好），
    该模块不可用时退化为线性插值——精度略低但绝不失败，
    这符合"感知层永远不能因为重采样挂掉"的降级原则。

    Args:
        samples: 单声道 float32 [-1,1]。
        src_rate: 源采样率。
        dst_rate: 目标采样率。

    Returns:
        重采样后的单声道 float32。
    """
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    if arr.size == 0 or src_rate == dst_rate or src_rate <= 0 or dst_rate <= 0:
        return arr
    try:
        from omnistream.media.audio.pcm import bytes_to_float32, float32_to_bytes
        from omnistream.media.audio.resampler import resample_once

        raw = resample_once(float32_to_bytes(arr), src_rate, dst_rate, channels=1)
        return bytes_to_float32(raw)
    except Exception as exc:  # noqa: BLE001 - 任何原因都退化为线性插值
        log.debug("tts.resample_fallback", error=str(exc),
                  src_rate=src_rate, dst_rate=dst_rate)
        target = max(1, int(round(arr.size * dst_rate / src_rate)))
        src_idx = np.linspace(0.0, arr.size - 1, num=target, dtype=np.float64)
        return np.interp(src_idx, np.arange(arr.size, dtype=np.float64),
                         arr).astype(np.float32)


def pcm_to_chunks(
    samples: np.ndarray,
    *,
    sample_rate: int = DEFAULT_TTS_SAMPLE_RATE,
    pts_offset_ms: int = 0,
    frame_ms: int = TTS_FRAME_MS,
    speaker: str = "assistant",
) -> list[AudioChunk]:
    """把一段 PCM 切成带连续 pts 的 :class:`AudioChunk` 列表。

    Args:
        samples: 单声道 float32。
        sample_rate: 采样率。
        pts_offset_ms: 起始 pts。
        frame_ms: 帧长。
        speaker: 说话人标签。

    Returns:
        音频帧列表。
    """
    chunks: list[AudioChunk] = []
    pts = int(pts_offset_ms)
    for frame in frame_pcm(np.asarray(samples, dtype=np.float32).reshape(-1),
                           sample_rate, frame_ms):
        chunk = AudioChunk.from_ndarray(frame, pts_ms=pts, sample_rate=sample_rate)
        chunk.speaker = speaker
        chunks.append(chunk)
        pts += frame_ms
    return chunks


class BaseStreamingTTS(abc.ABC):
    """流式语音合成抽象基类。"""

    #: 注册名，用于 ``create_tts("piper")``
    name: str = "base"

    #: 能力声明，子类覆盖
    capabilities: TTSCapabilities = TTSCapabilities()

    def __init__(
        self,
        *,
        voice: str = "default",
        language: str = "zh-CN",
        sample_rate: int = DEFAULT_TTS_SAMPLE_RATE,
        speed: float = 1.0,
        pitch: float = 0.0,
        volume_db: float = 0.0,
        frame_ms: int = TTS_FRAME_MS,
        session_id: str = "",
    ) -> None:
        self.voice = voice
        self.language = language
        self.sample_rate = int(sample_rate)
        self.speed = float(speed)
        self.pitch = float(pitch)
        self.volume_db = float(volume_db)
        self.frame_ms = int(frame_ms)
        self.session_id = session_id

        self._cancelled = asyncio.Event()
        self.sentences_synthesized = 0
        self.audio_ms_emitted = 0.0
        self.first_chunk_ms: float | None = None

    # ------------------------------------------------------------------
    # 子类实现
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def synthesize(self, text_stream: AsyncIterator[str]) -> AsyncIterator[AudioChunk]:
        """消费文本流，产出音频帧。

        实现约定：

        * 每收到一个完整小句就立即合成并吐帧，不等待整段文本；
        * ``pts_ms`` 从 0 开始按 ``frame_ms`` 严格递增；
        * :meth:`cancel` 被调用后必须尽快 ``return``，不得抛异常；
        * 不得抛出除 :class:`~omnistream.perception.errors.TTSError` 以外的异常。
        """
        raise NotImplementedError
        yield  # pragma: no cover - 让类型检查器识别这是异步生成器

    @abc.abstractmethod
    async def synthesize_text(self, request: SynthesisRequest) -> np.ndarray:
        """合成一小句，返回单声道 float32 PCM。

        这是引擎唯一必须实现的"真正干活"的方法；
        :meth:`synthesize` 的流式编排由基类的 :meth:`_stream_sentences` 复用。
        """

    # ------------------------------------------------------------------
    # 通用能力
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """引擎依赖是否就绪。"""
        return True

    @property
    def cancelled(self) -> bool:
        """是否已被取消。"""
        return self._cancelled.is_set()

    async def cancel(self) -> None:
        """取消当前合成（barge-in 调用，必须立即返回）。"""
        self._cancelled.set()

    def rearm(self) -> None:
        """新一轮开始前重置取消标记。"""
        self._cancelled.clear()
        self.first_chunk_ms = None

    def available_voices(self) -> list[str]:
        """可用音色列表。"""
        return [self.voice]

    def set_voice(self, voice: str) -> None:
        """切换音色。"""
        self.voice = voice

    async def aclose(self) -> None:
        """释放资源，幂等。"""
        return None

    def _build_request(self, text: str, *, index: int,
                       pts_offset_ms: int) -> SynthesisRequest:
        """按当前引擎参数构造合成请求。"""
        return SynthesisRequest(
            text=text,
            voice=self.voice,
            language=self.language,
            speed=self.speed,
            pitch=self.pitch,
            volume_db=self.volume_db,
            index=index,
            pts_offset_ms=pts_offset_ms,
        )

    async def _stream_sentences(
        self,
        text_stream: AsyncIterator[str],
        *,
        first_chunk_chars: int = 8,
    ) -> AsyncIterator[AudioChunk]:
        """通用流式编排：切句 -> 逐句合成 -> 切帧 -> 连续 pts。

        子类的 :meth:`synthesize` 通常只需 ``async for c in
        self._stream_sentences(stream): yield c``。

        Args:
            text_stream: 文本增量流。
            first_chunk_chars: 首句最小字符数，越小 TTFA 越低。

        Yields:
            带连续 pts 的音频帧。
        """
        from omnistream.perception.tts.sentencizer import StreamingSentencizer

        sentencizer = StreamingSentencizer(
            language=self.language, first_chunk_chars=first_chunk_chars)
        pts = 0
        index = 0
        started = asyncio.get_running_loop().time()

        async def _emit(sentence: str) -> AsyncIterator[AudioChunk]:
            nonlocal pts, index
            if not sentence.strip() or self.cancelled:
                return
            request = self._build_request(sentence, index=index, pts_offset_ms=pts)
            index += 1
            try:
                samples = await self.synthesize_text(request)
            except TTSError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise TTSError(f"{self.name} 合成失败: {exc}", cause=exc) from exc
            self.sentences_synthesized += 1
            for chunk in pcm_to_chunks(samples, sample_rate=self.sample_rate,
                                       pts_offset_ms=pts, frame_ms=self.frame_ms):
                if self.cancelled:
                    return
                pts += self.frame_ms
                self.audio_ms_emitted += self.frame_ms
                if self.first_chunk_ms is None:
                    self.first_chunk_ms = (
                        asyncio.get_running_loop().time() - started) * 1000.0
                yield chunk

        async for piece in text_stream:
            if self.cancelled:
                return
            for sentence in sentencizer.feed(piece):
                async for chunk in _emit(sentence):
                    yield chunk
                if self.cancelled:
                    return
        for sentence in sentencizer.flush():
            async for chunk in _emit(sentence):
                yield chunk
            if self.cancelled:
                return

    def stats(self) -> dict[str, Any]:
        """统计快照。"""
        return {
            "engine": self.name,
            "voice": self.voice,
            "language": self.language,
            "sentences": self.sentences_synthesized,
            "audio_ms": round(self.audio_ms_emitted, 1),
            "ttfa_ms": round(self.first_chunk_ms, 1) if self.first_chunk_ms else None,
            "cancelled": self.cancelled,
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name} voice={self.voice}>"


class TTSSession:
    """把拉模式的 :class:`BaseStreamingTTS` 适配成推模式。

    编排器侧是"LLM 文本增量随时到达"，而引擎接口是"给我一个文本流"，
    这个适配器用两条 :class:`BoundedStream` 把二者接起来。

    Example::

        session = TTSSession(tts)
        await session.start()
        await session.push_text("你好，")
        await session.push_text("今天天气不错。")
        async for chunk in session.audio():
            await playback.enqueue(chunk)
        await session.finish()
    """

    def __init__(
        self,
        tts: BaseStreamingTTS,
        *,
        text_queue: int = _TEXT_QUEUE,
        audio_queue: int = _AUDIO_QUEUE,
        name: str = "tts",
    ) -> None:
        self.tts = tts
        self._text: BoundedStream[str] = BoundedStream(
            maxsize=text_queue, policy=OverflowPolicy.BLOCK, name=f"{name}.text")
        self._audio: BoundedStream[AudioChunk] = BoundedStream(
            maxsize=audio_queue, policy=OverflowPolicy.BLOCK, name=f"{name}.audio")
        self._task: asyncio.Task[None] | None = None
        self._error: BaseException | None = None

    @property
    def running(self) -> bool:
        """后台合成协程是否在跑。"""
        return self._task is not None and not self._task.done()

    @property
    def error(self) -> BaseException | None:
        """后台协程捕获到的异常。"""
        return self._error

    async def start(self) -> None:
        """启动后台合成协程。"""
        if self.running:
            return
        self._error = None
        self.tts.rearm()
        self._task = asyncio.create_task(self._run(), name=f"tts-{self.tts.name}")

    async def _run(self) -> None:
        try:
            async for chunk in self.tts.synthesize(self._text.__aiter__()):
                await self._audio.put(chunk)
        except asyncio.CancelledError:
            raise
        except TTSError as exc:
            self._error = exc
            log.warning("tts.synthesize_failed", engine=self.tts.name, error=str(exc))
        except Exception as exc:  # noqa: BLE001
            self._error = TTSError(f"TTS 引擎 {self.tts.name} 异常: {exc}", cause=exc)
            log.warning("tts.synthesize_crashed", engine=self.tts.name, error=str(exc))
        finally:
            self._audio.close()

    async def push_text(self, text: str) -> None:
        """推入一段文本增量。"""
        if not text or self._text.closed:
            return
        await self._text.put(text)

    async def push_all(self, pieces: Iterable[str]) -> None:
        """批量推入。"""
        for piece in pieces:
            await self.push_text(piece)

    def audio(self) -> AsyncIterator[AudioChunk]:
        """异步迭代音频帧。"""
        return self._audio.__aiter__()

    async def finish(self, *, timeout_s: float = 10.0) -> None:
        """关闭文本输入并等待合成收尾。"""
        self._text.close()
        if self._task is not None:
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout_s)

    async def cancel(self) -> int:
        """barge-in：立即取消合成并清空待播音频，返回丢弃的帧数。"""
        await self.tts.cancel()
        self._text.close()
        dropped = self._audio.clear()
        self._audio.close()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
        return dropped

    async def aclose(self) -> None:
        """强制关闭并释放引擎。"""
        await self.cancel()
        await self.tts.aclose()

    def __repr__(self) -> str:
        return f"<TTSSession engine={self.tts.name} running={self.running}>"

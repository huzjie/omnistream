"""流式 ASR 抽象接口。

冻结契约（§2.20）::

    class BaseStreamingASR:
        async def stream(chunks) -> AsyncIterator[TranscriptDelta]
        async def reset() -> None
        async def finalize() -> TranscriptDelta | None

``stream`` 是"拉模式"：调用方给一个音频异步迭代器，实现吐出增量转写。
但全双工编排器是"推模式"（音频帧随时到达，转写要尽早上屏），
所以本模块额外提供 :class:`StreamingASRSession` —— 把任意
``BaseStreamingASR`` 包装成推模式：``await session.push(chunk)`` +
``async for d in session.deltas()``，内部用 :class:`BoundedStream` 做背压。

字段兼容性说明：``TranscriptDelta`` / ``WordTiming`` 由 WS1 定义。
为避免跨工作流的字段漂移，这里统一通过 :func:`build_delta` /
:func:`build_word_timing` 构造，两者会按模型实际声明的字段过滤入参。
"""
from __future__ import annotations

import abc
import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

import structlog

from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import PerceptionError
from omnistream.core.streams import BoundedStream, OverflowPolicy
from omnistream.schemas.audio import AudioChunk
from omnistream.schemas.transcript import TranscriptDelta, WordTiming

log = structlog.get_logger(__name__)

#: ASR 输入统一采样率
ASR_SAMPLE_RATE: Final[int] = 16000

#: 推模式内部音频队列深度（20ms 帧 × 200 = 4s 缓冲）
DEFAULT_INPUT_QUEUE: Final[int] = 200

#: 推模式内部结果队列深度
DEFAULT_OUTPUT_QUEUE: Final[int] = 64


class ASRUnavailable(RuntimeError):
    """引擎依赖缺失时由构造函数抛出，供工厂降级捕获。"""


@dataclass(slots=True)
class ASRCapabilities:
    """引擎能力声明，路由器据此选型。"""

    streaming: bool = True
    word_timestamps: bool = False
    language_detection: bool = False
    hotwords: bool = False
    punctuation: bool = False
    #: 是否需要 GPU 才能达到实时率
    needs_gpu: bool = False
    #: 是否需要网络
    needs_network: bool = False
    #: 典型首个 partial 的延迟（毫秒），用于延迟预算路由
    typical_partial_ms: int = 300
    languages: tuple[str, ...] = ("zh", "en")


def build_word_timing(word: str, start_ms: int, end_ms: int,
                      confidence: float = 1.0) -> WordTiming:
    """构造 :class:`WordTiming`，按模型实际字段过滤。"""
    payload: dict[str, Any] = {
        "word": word,
        "start_ms": int(start_ms),
        "end_ms": int(end_ms),
        "confidence": float(confidence),
        "probability": float(confidence),
        "text": word,
    }
    return _construct(WordTiming, payload)


def build_delta(
    *,
    turn_id: str,
    text: str,
    is_final: bool = False,
    stability: float = 1.0,
    start_ms: int = 0,
    end_ms: int = 0,
    speaker: str = "",
    words: Sequence[WordTiming] | None = None,
    language: str = "",
    engine: str = "",
) -> TranscriptDelta:
    """构造 :class:`TranscriptDelta`，按模型实际字段过滤。"""
    payload: dict[str, Any] = {
        "turn_id": turn_id,
        "text": text,
        "is_final": bool(is_final),
        "stability": float(max(0.0, min(1.0, stability))),
        "start_ms": int(start_ms),
        "end_ms": int(end_ms),
        "speaker": speaker,
        "words": list(words or []),
        "language": language,
        "engine": engine,
    }
    return _construct(TranscriptDelta, payload)


def _construct(model: type, payload: dict[str, Any]) -> Any:
    """按 pydantic 模型声明的字段过滤后实例化。"""
    fields = getattr(model, "model_fields", None)
    if isinstance(fields, dict) and fields:
        payload = {k: v for k, v in payload.items() if k in fields}
    return model(**payload)


class BaseStreamingASR(abc.ABC):
    """流式语音识别抽象基类。"""

    #: 注册名，用于 ``create_asr("mock")``
    name: str = "base"

    #: 能力声明，子类覆盖
    capabilities: ASRCapabilities = ASRCapabilities()

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
    ) -> None:
        self.language = language
        self.sample_rate = int(sample_rate)
        self.turn_id = turn_id
        self.session_id = session_id
        self.hotwords: list[str] = [w for w in (hotwords or []) if w]
        self.enable_punctuation = bool(enable_punctuation)
        self.enable_word_timestamps = bool(enable_word_timestamps)

        self.audio_ms = 0.0
        self.partials_emitted = 0
        self.finals_emitted = 0

    # ------------------------------------------------------------------
    # 子类实现
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def stream(self, chunks: AsyncIterator[AudioChunk]) -> AsyncIterator[TranscriptDelta]:
        """消费音频流，产出增量转写。

        实现约定：

        * 每个非最终 delta 的 ``is_final=False``，``stability`` 反映稳定度；
        * 输入流结束后必须补吐一条 ``is_final=True`` 的 delta（可为空文本）；
        * 不得抛出除 :class:`PerceptionError` 以外的异常。
        """
        raise NotImplementedError
        yield  # pragma: no cover - 让类型检查器识别这是异步生成器

    @abc.abstractmethod
    async def reset(self) -> None:
        """清空解码状态，准备下一轮。"""

    @abc.abstractmethod
    async def finalize(self) -> TranscriptDelta | None:
        """强制结束当前 utterance，返回最终结果（无内容时返回 ``None``）。"""

    # ------------------------------------------------------------------
    # 通用能力
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """引擎依赖是否就绪。"""
        return True

    def set_turn(self, turn_id: str) -> None:
        """切换轮次 id（新一轮开始时调用）。"""
        self.turn_id = turn_id

    def set_hotwords(self, words: Iterable[str]) -> None:
        """更新热词表。"""
        self.hotwords = [w for w in words if w]

    async def aclose(self) -> None:
        """释放资源，幂等。"""
        return None

    def _account_audio(self, chunk: AudioChunk) -> None:
        self.audio_ms += float(chunk.duration_ms)

    def _account_delta(self, delta: TranscriptDelta) -> None:
        if getattr(delta, "is_final", False):
            self.finals_emitted += 1
        else:
            self.partials_emitted += 1

    def stats(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "language": self.language,
            "audio_ms": round(self.audio_ms, 1),
            "partials": self.partials_emitted,
            "finals": self.finals_emitted,
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name} lang={self.language}>"


class StreamingASRSession:
    """把拉模式的 :class:`BaseStreamingASR` 适配成推模式。

    用法::

        session = StreamingASRSession(asr)
        await session.start()
        ...
        await session.push(chunk)          # 采集侧
        async for delta in session.deltas():   # 消费侧
            ...
        final = await session.finish()
    """

    def __init__(
        self,
        asr: BaseStreamingASR,
        *,
        input_queue: int = DEFAULT_INPUT_QUEUE,
        output_queue: int = DEFAULT_OUTPUT_QUEUE,
        name: str = "asr",
    ) -> None:
        self.asr = asr
        self._in: BoundedStream[AudioChunk] = BoundedStream(
            maxsize=input_queue, policy=OverflowPolicy.BLOCK, name=f"{name}.in")
        self._out: BoundedStream[TranscriptDelta] = BoundedStream(
            maxsize=output_queue, policy=OverflowPolicy.DROP_OLDEST, name=f"{name}.out")
        self._task: asyncio.Task[None] | None = None
        self._error: BaseException | None = None
        self._finished = asyncio.Event()

    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def error(self) -> BaseException | None:
        return self._error

    async def start(self) -> None:
        """启动后台解码协程。"""
        if self.running:
            return
        self._finished.clear()
        self._error = None
        self._task = asyncio.create_task(self._run(), name=f"asr-{self.asr.name}")

    async def _run(self) -> None:
        try:
            async for delta in self.asr.stream(self._in.__aiter__()):
                self.asr._account_delta(delta)
                await self._out.put(delta)
        except asyncio.CancelledError:
            raise
        except PerceptionError as exc:
            self._error = exc
            log.warning("asr.stream_failed", engine=self.asr.name, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - 统一包装成 PerceptionError 语义
            self._error = PerceptionError(f"ASR 引擎 {self.asr.name} 异常: {exc}",
                                          code=ErrorCode.ASR_FAILED, cause=exc)
            log.warning("asr.stream_crashed", engine=self.asr.name, error=str(exc))
        finally:
            self._out.close()
            self._finished.set()

    async def push(self, chunk: AudioChunk) -> None:
        """推入一帧音频（队列满时背压等待）。"""
        if self._in.closed:
            return
        self.asr._account_audio(chunk)
        await self._in.put(chunk)

    def deltas(self) -> AsyncIterator[TranscriptDelta]:
        """异步迭代转写增量。"""
        return self._out.__aiter__()

    async def finish(self, *, timeout_s: float = 3.0) -> TranscriptDelta | None:
        """关闭输入并等待收尾，返回最终 delta（若引擎另行给出）。"""
        self._in.close()
        if self._task is not None:
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout_s)
        try:
            return await self.asr.finalize()
        except Exception as exc:  # noqa: BLE001
            log.warning("asr.finalize_failed", engine=self.asr.name, error=str(exc))
            return None

    async def aclose(self) -> None:
        """强制关闭。"""
        self._in.close()
        self._out.close()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
        await self.asr.aclose()

    def __repr__(self) -> str:
        return f"<StreamingASRSession engine={self.asr.name} running={self.running}>"


__all__ = [
    "ASR_SAMPLE_RATE",
    "DEFAULT_INPUT_QUEUE",
    "DEFAULT_OUTPUT_QUEUE",
    "ASRCapabilities",
    "ASRUnavailable",
    "BaseStreamingASR",
    "StreamingASRSession",
    "build_delta",
    "build_word_timing",
]

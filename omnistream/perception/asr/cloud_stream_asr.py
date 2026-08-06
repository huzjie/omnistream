"""云端流式 ASR 通用适配器（WebSocket 双向流）。

各家云 ASR（Azure Speech / Deepgram / 火山 / 阿里 / 腾讯）的协议差异集中在三处：

1. **握手**：URL 参数、鉴权头、启动帧；
2. **音频帧**：原始 PCM 二进制 vs base64 JSON；
3. **结果帧**：JSON 字段名不同（``text`` / ``transcript`` / ``result.text``）。

本模块把这三处抽成 :class:`CloudASRDialect`，**协议本身只写一遍**。
新增一家云厂商 = 写 20 行 dialect，不改任何流程代码。

内置方言：``generic`` / ``deepgram`` / ``azure`` / ``volc``。

依赖 ``websockets``；缺失时构造抛 :class:`ASRUnavailable`，由工厂降级到本地引擎。
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Final, Iterable
from urllib.parse import urlencode

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
from omnistream.perception.asr.word_timing import interpolate_words
from omnistream.perception.errors import ASRError, ASRTimeout
from omnistream.schemas.audio import AudioChunk
from omnistream.schemas.transcript import TranscriptDelta

log = structlog.get_logger(__name__)

__all__ = [
    "CloudASRDialect",
    "CloudStreamASR",
    "DIALECTS",
    "cloud_asr_available",
    "get_dialect",
]

try:  # pragma: no cover - 依赖是否存在取决于部署环境
    import websockets  # type: ignore
    from websockets.asyncio.client import connect as _ws_connect  # type: ignore

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    try:  # websockets < 13 的旧入口
        import websockets  # type: ignore

        _ws_connect = websockets.connect  # type: ignore[attr-defined]
        _IMPORT_ERROR = None
    except Exception as exc2:  # noqa: BLE001
        websockets = None  # type: ignore[assignment]
        _ws_connect = None  # type: ignore[assignment]
        _IMPORT_ERROR = exc2 or exc

#: 连接超时（秒）
DEFAULT_CONNECT_TIMEOUT_S: Final[float] = 6.0

#: 无结果的最长等待（秒），超过判定链路异常
DEFAULT_IDLE_TIMEOUT_S: Final[float] = 20.0

#: 结果队列深度
_RESULT_QUEUE: Final[int] = 128


def cloud_asr_available() -> bool:
    """``websockets`` 是否可用。"""
    return _ws_connect is not None


@dataclass(slots=True)
class CloudASRDialect:
    """一家云厂商的协议方言。"""

    name: str
    #: WS 端点模板，``{region}`` 会被替换
    endpoint: str = ""
    #: 是否以二进制发送原始 PCM（False 表示 base64 JSON）
    binary_audio: bool = True
    #: 音频采样率
    sample_rate: int = ASR_SAMPLE_RATE
    #: 结束标记：发送该文本帧表示音频结束
    finish_frame: str | None = None
    #: 构造查询参数
    build_params: Callable[["CloudStreamASR"], dict[str, str]] | None = None
    #: 构造鉴权头
    build_headers: Callable[["CloudStreamASR"], dict[str, str]] | None = None
    #: 构造启动帧（None 表示无需启动帧）
    build_start_frame: Callable[["CloudStreamASR"], dict[str, Any]] | None = None
    #: 解析结果帧 -> ``(text, is_final, confidence)``；返回 None 表示忽略该帧
    parse_result: Callable[[dict[str, Any]], tuple[str, bool, float] | None] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 内置方言
# ---------------------------------------------------------------------------


def _generic_params(asr: "CloudStreamASR") -> dict[str, str]:
    return {
        "language": asr.language,
        "sample_rate": str(asr.sample_rate),
        "encoding": "pcm16",
    }


def _generic_headers(asr: "CloudStreamASR") -> dict[str, str]:
    return {"Authorization": f"Bearer {asr.api_key}"} if asr.api_key else {}


def _generic_start(asr: "CloudStreamASR") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "start",
        "language": asr.language,
        "sample_rate": asr.sample_rate,
        "encoding": "pcm16",
        "interim_results": True,
        "punctuate": asr.enable_punctuation,
    }
    hotwords = asr.bias.as_hotword_list()
    if hotwords:
        payload["keywords"] = hotwords
    return payload


def _generic_parse(msg: dict[str, Any]) -> tuple[str, bool, float] | None:
    for key in ("text", "transcript", "result"):
        value = msg.get(key)
        if isinstance(value, str) and value:
            return value, bool(msg.get("is_final") or msg.get("final")), \
                float(msg.get("confidence", 0.8) or 0.8)
        if isinstance(value, dict):
            inner = value.get("text") or value.get("transcript")
            if isinstance(inner, str) and inner:
                return inner, bool(value.get("is_final") or msg.get("is_final")), \
                    float(value.get("confidence", 0.8) or 0.8)
    return None


def _deepgram_params(asr: "CloudStreamASR") -> dict[str, str]:
    return {
        "encoding": "linear16",
        "sample_rate": str(asr.sample_rate),
        "channels": "1",
        "language": asr.language,
        "interim_results": "true",
        "punctuate": "true" if asr.enable_punctuation else "false",
        "model": asr.model or "nova-2",
    }


def _deepgram_headers(asr: "CloudStreamASR") -> dict[str, str]:
    return {"Authorization": f"Token {asr.api_key}"} if asr.api_key else {}


def _deepgram_parse(msg: dict[str, Any]) -> tuple[str, bool, float] | None:
    channel = msg.get("channel")
    if not isinstance(channel, dict):
        return None
    alts = channel.get("alternatives")
    if not isinstance(alts, list) or not alts:
        return None
    first = alts[0]
    if not isinstance(first, dict):
        return None
    text = str(first.get("transcript", "") or "")
    if not text:
        return None
    return text, bool(msg.get("is_final")), float(first.get("confidence", 0.8) or 0.8)


def _azure_parse(msg: dict[str, Any]) -> tuple[str, bool, float] | None:
    kind = str(msg.get("RecognitionStatus", "") or msg.get("status", ""))
    text = str(msg.get("DisplayText") or msg.get("Text") or "")
    if not text:
        return None
    return text, kind.lower() in ("success", "final"), float(msg.get("Confidence", 0.85) or 0.85)


def _volc_start(asr: "CloudStreamASR") -> dict[str, Any]:
    return {
        "app": {"appid": asr.app_id, "token": asr.api_key, "cluster": asr.model or "volc_asr"},
        "user": {"uid": asr.session_id or "omnistream"},
        "audio": {"format": "raw", "rate": asr.sample_rate, "bits": 16, "channel": 1},
        "request": {"reqid": asr.session_id or "omnistream",
                    "nbest": 1, "show_utterances": True,
                    "result_type": "single", "sequence": 1},
    }


def _volc_parse(msg: dict[str, Any]) -> tuple[str, bool, float] | None:
    result = msg.get("result")
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            text = str(first.get("text", "") or "")
            if text:
                return text, bool(msg.get("is_final") or first.get("is_final")), 0.85
    return _generic_parse(msg)


#: 内置方言注册表
DIALECTS: Final[dict[str, CloudASRDialect]] = {
    "generic": CloudASRDialect(
        name="generic",
        endpoint="",
        binary_audio=True,
        finish_frame=json.dumps({"type": "finish"}),
        build_params=_generic_params,
        build_headers=_generic_headers,
        build_start_frame=_generic_start,
        parse_result=_generic_parse,
    ),
    "deepgram": CloudASRDialect(
        name="deepgram",
        endpoint="wss://api.deepgram.com/v1/listen",
        binary_audio=True,
        finish_frame=json.dumps({"type": "CloseStream"}),
        build_params=_deepgram_params,
        build_headers=_deepgram_headers,
        build_start_frame=None,
        parse_result=_deepgram_parse,
    ),
    "azure": CloudASRDialect(
        name="azure",
        endpoint="wss://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1",
        binary_audio=True,
        finish_frame=None,
        build_params=lambda asr: {"language": asr.language, "format": "detailed"},
        build_headers=lambda asr: ({"Ocp-Apim-Subscription-Key": asr.api_key}
                                   if asr.api_key else {}),
        build_start_frame=None,
        parse_result=_azure_parse,
    ),
    "volc": CloudASRDialect(
        name="volc",
        endpoint="wss://openspeech.bytedance.com/api/v2/asr",
        binary_audio=True,
        finish_frame=None,
        build_params=lambda asr: {},
        build_headers=lambda asr: ({"Authorization": f"Bearer; {asr.api_key}"}
                                   if asr.api_key else {}),
        build_start_frame=_volc_start,
        parse_result=_volc_parse,
    ),
}


def get_dialect(name: str) -> CloudASRDialect:
    """按名取方言，未知名回落到 ``generic``。"""
    return DIALECTS.get(name, DIALECTS["generic"])


class CloudStreamASR(BaseStreamingASR):
    """云端 WebSocket 流式识别器。

    Raises:
        ASRUnavailable: ``websockets`` 未安装，或未配置端点/密钥。
    """

    name = "cloud"
    capabilities = ASRCapabilities(
        streaming=True,
        word_timestamps=False,
        language_detection=True,
        hotwords=True,
        punctuation=True,
        needs_gpu=False,
        needs_network=True,
        typical_partial_ms=200,
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
        dialect: str = "generic",
        endpoint: str = "",
        api_key: str = "",
        app_id: str = "",
        region: str = "eastasia",
        model: str = "",
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
        idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S,
    ) -> None:
        """
        Args:
            dialect: 方言名，见 :data:`DIALECTS`。
            endpoint: 覆盖方言默认端点。
            api_key: 鉴权密钥。
            app_id: 部分厂商需要的应用 id。
            region: Azure 等按区域路由的厂商使用。
            model: 云端模型名。
            connect_timeout_s: 握手超时。
            idle_timeout_s: 无结果最长等待。

        Raises:
            ASRUnavailable: 依赖缺失或端点为空。
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
        if _ws_connect is None:
            raise ASRUnavailable(f"websockets 不可用: {_IMPORT_ERROR}")
        self.dialect = get_dialect(dialect)
        self.endpoint = endpoint or self.dialect.endpoint
        if not self.endpoint:
            raise ASRUnavailable(f"云端 ASR 方言 {dialect} 未配置 endpoint")
        self.api_key = api_key
        self.app_id = app_id
        self.region = region
        self.model = model
        self.connect_timeout_s = float(connect_timeout_s)
        self.idle_timeout_s = float(idle_timeout_s)
        self.bias = HotwordBias(self.hotwords)

        self._ws: Any = None
        self._results: asyncio.Queue[tuple[str, bool, float] | None] = \
            asyncio.Queue(maxsize=_RESULT_QUEUE)
        self._reader: asyncio.Task[None] | None = None
        self._elapsed_ms = 0.0
        self._segment_start_ms = 0.0
        self._delta_index = 0
        self._last_final = ""
        self.frames_sent = 0
        self.bytes_sent = 0
        self.reconnects = 0

    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        return cloud_asr_available()

    def _build_url(self) -> str:
        url = self.endpoint.replace("{region}", self.region)
        builder = self.dialect.build_params
        params = builder(self) if builder else {}
        if params:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urlencode(params)}"
        return url

    async def _connect(self) -> Any:
        """建立 WS 连接并发送启动帧。"""
        url = self._build_url()
        builder = self.dialect.build_headers
        headers = builder(self) if builder else {}
        try:
            ws = await asyncio.wait_for(
                _ws_connect(url, additional_headers=headers or None,
                            max_size=8 * 1024 * 1024),
                timeout=self.connect_timeout_s,
            )
        except TypeError:
            # 旧版 websockets 用 extra_headers
            ws = await asyncio.wait_for(
                _ws_connect(url, extra_headers=headers or None,
                            max_size=8 * 1024 * 1024),
                timeout=self.connect_timeout_s,
            )
        except TimeoutError as exc:
            raise ASRTimeout(f"云端 ASR 握手超时: {url}", cause=exc) from exc
        except Exception as exc:  # noqa: BLE001
            raise ASRError(f"云端 ASR 连接失败: {exc}", cause=exc) from exc

        start_builder = self.dialect.build_start_frame
        if start_builder is not None:
            payload = start_builder(self)
            await ws.send(json.dumps(payload, ensure_ascii=False))
        log.info("asr.cloud_connected", dialect=self.dialect.name,
                 session_id=self.session_id)
        return ws

    async def _read_loop(self, ws: Any) -> None:
        """后台读取云端结果帧。"""
        parser = self.dialect.parse_result or _generic_parse
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    try:
                        raw = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(msg, dict):
                    continue
                if msg.get("error") or msg.get("code") not in (None, 0, 1000, "0"):
                    log.warning("asr.cloud_error_frame", payload=str(msg)[:200])
                parsed = parser(msg)
                if parsed is None:
                    continue
                with contextlib.suppress(asyncio.QueueFull):
                    self._results.put_nowait(parsed)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("asr.cloud_read_failed", error=str(exc))
        finally:
            with contextlib.suppress(asyncio.QueueFull):
                self._results.put_nowait(None)

    async def _send_audio(self, chunk: AudioChunk) -> None:
        """发送一帧音频。"""
        ws = self._ws
        if ws is None:
            return
        payload = self._encode(chunk)
        try:
            if self.dialect.binary_audio:
                await ws.send(payload)
            else:
                await ws.send(json.dumps({
                    "type": "audio",
                    "audio": base64.b64encode(payload).decode("ascii"),
                }))
        except Exception as exc:  # noqa: BLE001
            raise ASRError(f"云端 ASR 发送失败: {exc}", cause=exc) from exc
        self.frames_sent += 1
        self.bytes_sent += len(payload)

    def _encode(self, chunk: AudioChunk) -> bytes:
        """把 chunk 转成目标采样率的 pcm16 字节。"""
        if chunk.sample_rate == self.sample_rate:
            try:
                samples = np.asarray(chunk.to_ndarray(), dtype=np.float32).reshape(-1)
            except ValueError:
                return chunk.data
            return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        samples = np.asarray(chunk.to_ndarray(), dtype=np.float32).reshape(-1)
        target = max(1, int(round(samples.size * self.sample_rate / chunk.sample_rate)))
        resampled = np.interp(
            np.linspace(0.0, samples.size - 1, num=target, dtype=np.float64),
            np.arange(samples.size, dtype=np.float64), samples).astype(np.float32)
        return (np.clip(resampled, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    def _make_delta(self, text: str, *, is_final: bool,
                    confidence: float) -> TranscriptDelta:
        corrected, _ = self.bias.correct(text)
        words = []
        if self.enable_word_timestamps and corrected:
            words = [s.to_model() for s in interpolate_words(
                corrected, int(self._segment_start_ms), int(self._elapsed_ms))]
        delta = build_delta(
            turn_id=self.turn_id,
            text=corrected,
            is_final=is_final,
            stability=1.0 if is_final else max(0.0, min(1.0, confidence)),
            start_ms=int(self._segment_start_ms),
            end_ms=int(self._elapsed_ms),
            speaker="user",
            words=words,
            language=self.language,
            engine=f"{self.name}:{self.dialect.name}",
        )
        try:
            delta.index = self._delta_index
        except (AttributeError, ValueError):
            pass
        self._delta_index += 1
        return delta

    async def _drain_results(self) -> list[TranscriptDelta]:
        """非阻塞地取出已到达的结果。"""
        out: list[TranscriptDelta] = []
        while True:
            try:
                item = self._results.get_nowait()
            except asyncio.QueueEmpty:
                return out
            if item is None:
                return out
            text, is_final, confidence = item
            if not text:
                continue
            if is_final:
                self._last_final = text
                self._segment_start_ms = self._elapsed_ms
                self._delta_index = 0
            out.append(self._make_delta(text, is_final=is_final, confidence=confidence))

    # ------------------------------------------------------------------

    async def stream(self, chunks: AsyncIterator[AudioChunk]) -> AsyncIterator[TranscriptDelta]:
        """把音频转发到云端并透传结果。"""
        self._ws = await self._connect()
        self._reader = asyncio.create_task(
            self._read_loop(self._ws), name=f"cloud-asr-{self.dialect.name}")
        last_result_at = time.monotonic()
        emitted_final = False
        try:
            async for chunk in chunks:
                await self._send_audio(chunk)
                self._elapsed_ms += float(chunk.duration_ms)
                for delta in await self._drain_results():
                    last_result_at = time.monotonic()
                    emitted_final = bool(getattr(delta, "is_final", False))
                    yield delta
                if time.monotonic() - last_result_at > self.idle_timeout_s:
                    raise ASRTimeout(
                        f"云端 ASR {self.idle_timeout_s}s 无结果，判定链路异常")

            # 音频结束：发结束帧并等待收尾结果
            await self._finish_stream()
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                try:
                    item = await asyncio.wait_for(self._results.get(), timeout=0.5)
                except TimeoutError:
                    break
                if item is None:
                    break
                text, is_final, confidence = item
                if not text:
                    continue
                emitted_final = is_final
                yield self._make_delta(text, is_final=is_final, confidence=confidence)
                if is_final:
                    self._last_final = text
                    break
            if not emitted_final:
                yield self._make_delta(self._last_final, is_final=True, confidence=1.0)
        finally:
            await self._teardown()

    async def _finish_stream(self) -> None:
        """告知云端音频已结束。"""
        ws, frame = self._ws, self.dialect.finish_frame
        if ws is None:
            return
        with contextlib.suppress(Exception):
            if frame is not None:
                await ws.send(frame)
            elif self.dialect.binary_audio:
                await ws.send(b"")  # 空帧作为 EOS，多数厂商接受

    async def _teardown(self) -> None:
        """关闭连接与读协程。"""
        if self._reader is not None and not self._reader.done():
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader
        self._reader = None
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
        self._ws = None

    async def reset(self) -> None:
        """断开并清空状态，下一轮重新握手。"""
        await self._teardown()
        while not self._results.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._results.get_nowait()
        self._elapsed_ms = 0.0
        self._segment_start_ms = 0.0
        self._delta_index = 0
        self._last_final = ""

    async def finalize(self) -> TranscriptDelta | None:
        """返回最后一条 final（若有）。"""
        if not self._last_final:
            return None
        delta = self._make_delta(self._last_final, is_final=True, confidence=1.0)
        self._last_final = ""
        return delta

    def set_hotwords(self, words: Iterable[str]) -> None:
        """更新热词（下次握手生效）。"""
        super().set_hotwords(words)
        self.bias = HotwordBias(self.hotwords)

    async def aclose(self) -> None:
        """关闭连接。"""
        await self._teardown()

    def stats(self) -> dict[str, Any]:
        data = super().stats()
        data.update({
            "dialect": self.dialect.name,
            "frames_sent": self.frames_sent,
            "bytes_sent": self.bytes_sent,
            "connected": self._ws is not None,
            "reconnects": self.reconnects,
        })
        return data

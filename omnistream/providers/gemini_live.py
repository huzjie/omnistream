"""Gemini Live API 适配器。

对接 Google Gemini Live（``wss://generativelanguage.googleapis.com/ws/...``），
原生音频全双工 + 视觉理解（video_in），支持 server VAD 与打断。
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError:  # pragma: no cover
    websockets = None
    ConnectionClosed = Exception

from omnistream.core.errors import (
    CapabilityUnsupported,
    ProviderError,
    ProviderTimeout,
    ProviderUnauthorized,
)
from omnistream.core.clock import now_ms
from omnistream.core.streams import BoundedStream
from omnistream.protocol.enums import AudioEncoding, StopReason, TurnRole
from omnistream.providers.base import BaseRealtimeProvider, ProviderEvent
from omnistream.schemas.audio import AudioChunk
from omnistream.schemas.provider import (
    ProviderCapabilities,
    ProviderCredential,
    ProviderHealth,
)
from omnistream.schemas.session import SessionConfig
from omnistream.schemas.tool import ToolCall, ToolResult
from omnistream.schemas.video import VideoFrame

logger = logging.getLogger(__name__)


class GeminiLiveProvider(BaseRealtimeProvider):
    """Gemini Live —— 原生全双工 + 实时视觉。"""

    name = "gemini_live"
    display_name = "Gemini Live"
    vendor = "Google"
    supported_models = ["gemini-2.5-live", "gemini-3-live"]
    default_model = "gemini-3-live"
    docs_url = "https://ai.google.dev/gemini-api/docs/live"

    capabilities = ProviderCapabilities(
        native_duplex=True,
        audio_in=True,
        audio_out=True,
        video_in=True,
        text_in=True,
        text_out=True,
        server_vad=True,
        function_calling=True,
        parallel_tools=True,
        interruption=True,
        proactive=True,
        streaming_transcript=True,
        max_context_tokens=1_000_000,
        input_audio_formats=[AudioEncoding.PCM16],
        output_audio_formats=[AudioEncoding.PCM16],
        input_sample_rates=[16000],
        output_sample_rates=[24000],
        max_video_fps=8.0,
    )

    def __init__(self, credentials: ProviderCredential | None, settings: Any) -> None:
        if websockets is None:
            raise CapabilityUnsupported(
                "gemini_live 需要 websockets 包", provider=self.name)
        self._api_key: str = (credentials.api_key or "") if credentials else ""
        self._base_url: str = (
            (credentials.base_url or "wss://generativelanguage.googleapis.com/ws") if credentials
            else "wss://generativelanguage.googleapis.com/ws"
        )
        self._ws: Any = None
        self._stream = BoundedStream[ProviderEvent](maxsize=1024)
        self._reader_task: asyncio.Task | None = None
        self._closed = False

    async def connect(self, config: SessionConfig) -> None:
        if not self._api_key:
            raise ProviderUnauthorized("Gemini Live 未配置 API Key", provider=self.name)
        url = (
            f"{self._base_url.rstrip('/')}/google.ai.generativelanguage.v1beta.GenerativeService/"
            f"BidiGenerateContent?key={self._api_key}"
        )
        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(url, max_size=16 * 1024 * 1024), timeout=15)
        except asyncio.TimeoutError as exc:
            raise ProviderTimeout("连接 Gemini Live 超时", provider=self.name) from exc
        except Exception as exc:
            raise ProviderError(f"连接失败: {exc}", provider=self.name) from exc

        setup = {
            "setup": {
                "model": f"models/{config.model or self.default_model}",
                "generationConfig": {
                    "temperature": config.temperature,
                    "responseModalities": ["AUDIO", "TEXT"],
                },
                "systemInstruction": {
                    "parts": [{"text": config.instructions}],
                },
            }
        }
        if config.tools:
            setup["setup"]["tools"] = [{
                "functionDeclarations": [
                    {"name": t.name, "description": t.description,
                     "parameters": t.parameters or {}} for t in config.tools]
            }]
        await self._ws.send(json.dumps(setup))
        self._stream.put_nowait(ProviderEvent(kind="connected", role=TurnRole.SYSTEM))
        self._reader_task = asyncio.create_task(self._reader_loop())

    async def send_audio(self, chunk: AudioChunk) -> None:
        await self._ws.send(json.dumps({
            "realtimeInput": {
                "audio": {"data": base64.b64encode(chunk.data).decode("ascii")},
            }
        }))

    async def send_video(self, frame: VideoFrame) -> None:
        await self._ws.send(json.dumps({
            "realtimeInput": {
                "mediaChunks": [{
                    "mimeType": frame.encoding.value,
                    "data": base64.b64encode(frame.data).decode("ascii"),
                }]
            }
        }))

    async def send_text(self, text: str, *, role: TurnRole = TurnRole.USER) -> None:
        await self._ws.send(json.dumps({
            "clientContent": {"turns": [{"role": "user", "parts": [{"text": text}]}]}
        }))

    async def request_response(self, *, instructions_suffix: str | None = None) -> None:
        # Gemini Live 是连续流式，无需显式触发；后缀指令走 serverContent 注入
        if instructions_suffix:
            await self._ws.send(json.dumps({
                "clientContent": {"turns": [{
                    "role": "user",
                    "parts": [{"text": f"[系统临时指示] {instructions_suffix}"}],
                }]}
            }))

    async def cancel_response(self, reason: StopReason = StopReason.INTERRUPTED) -> None:
        # Gemini 通过发送 audio 输入即可打断；这里保留空实现并发出取消事件
        self._stream.put_nowait(
            ProviderEvent(kind="response_cancelled", stop_reason=reason))

    async def submit_tool_result(self, result: ToolResult) -> None:
        await self._ws.send(json.dumps({
            "clientContent": {"turns": [{
                "role": "user",
                "parts": [{"functionResponse": {
                    "name": result.call_id,
                    "response": result.content,
                }}],
            }]}
        }))

    def events(self):
        return self._stream.__aiter__()

    async def _reader_loop(self) -> None:
        try:
            async for raw in self._ws:
                ev = self._parse(raw)
                if ev is not None:
                    self._stream.put_nowait(ev)
        except ConnectionClosed:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.exception("gemini reader error: %s", exc)
            self._stream.put_nowait(ProviderEvent(kind="error", error=exc))
        finally:
            self._stream.put_nowait(ProviderEvent(kind="closed", role=TurnRole.SYSTEM))

    def _parse(self, raw: Any) -> ProviderEvent | None:
        try:
            msg = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None
        if "serverContent" not in msg:
            return None
        sc = msg["serverContent"]
        if "interrupted" in sc and sc["interrupted"]:
            return ProviderEvent(kind="response_cancelled",
                                 stop_reason=StopReason.INTERRUPTED)
        parts = sc.get("modelTurn", {}).get("parts", [])
        for p in parts:
            if "inlineData" in p:
                b64 = p["inlineData"].get("data", "")
                try:
                    data = base64.b64decode(b64)
                except Exception:  # noqa: BLE001
                    data = b""
                return ProviderEvent(
                    kind="audio_delta",
                    audio=AudioChunk(pts_ms=0, data=data, sample_rate=24000,
                                     encoding=AudioEncoding.PCM16))
            if "text" in p and p.get("text"):
                return ProviderEvent(kind="text_delta", text=p["text"])
            if "functionCall" in p:
                fc = p["functionCall"]
                return ProviderEvent(
                    kind="tool_call",
                    tool_call=ToolCall(
                        id=fc.get("id", ""),
                        name=fc.get("name", ""),
                        arguments=fc.get("args", {}),
                    ))
        if sc.get("turnComplete"):
            return ProviderEvent(kind="response_done",
                                 stop_reason=StopReason.COMPLETED, is_final=True)
        return None

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except Exception:  # noqa: BLE001
                pass
        if self._ws is not None:
            await self._ws.close()
        self._stream.put_nowait(ProviderEvent(kind="closed", role=TurnRole.SYSTEM))

    async def health(self) -> ProviderHealth:
        return ProviderHealth(
            name=self.name, healthy=self._ws is not None and not self._closed,
            checked_at_ms=now_ms())

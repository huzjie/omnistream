"""OpenAI Realtime API 适配器。

对接 OpenAI 实时语音 API（``/v1/realtime``，WebSocket，原生全双工）。
协议与 Doubao 高度相似（同源于 OpenAI 兼容协议），但独立实现以便独立演进。
"""
from __future__ import annotations

import asyncio
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


class OpenAIRealtimeProvider(BaseRealtimeProvider):
    """OpenAI Realtime API —— 原生音频全双工 + function calling。"""

    name = "openai_realtime"
    display_name = "OpenAI Realtime"
    vendor = "OpenAI"
    supported_models = ["gpt-realtime", "gpt-realtime-mini"]
    default_model = "gpt-realtime"
    docs_url = "https://platform.openai.com/docs/guides/realtime"

    capabilities = ProviderCapabilities(
        native_duplex=True,
        audio_in=True,
        audio_out=True,
        video_in=False,
        text_in=True,
        text_out=True,
        server_vad=True,
        function_calling=True,
        parallel_tools=True,
        interruption=True,
        proactive=True,
        streaming_transcript=True,
        max_context_tokens=128_000,
        input_audio_formats=[AudioEncoding.PCM16],
        output_audio_formats=[AudioEncoding.PCM16],
        input_sample_rates=[24000, 16000],
        output_sample_rates=[24000],
    )

    def __init__(self, credentials: ProviderCredential | None, settings: Any) -> None:
        if websockets is None:
            raise CapabilityUnsupported(
                "openai_realtime 需要 websockets 包：pip install websockets",
                provider=self.name,
            )
        self._api_key: str = (credentials.api_key or "") if credentials else ""
        self._base_url: str = (credentials.base_url or "https://api.openai.com/v1") if credentials else ""
        self._ws: Any = None
        self._stream = BoundedStream[ProviderEvent](maxsize=1024)
        self._reader_task: asyncio.Task | None = None
        self._closed = False

    async def connect(self, config: SessionConfig) -> None:
        if not self._api_key:
            raise ProviderUnauthorized("OpenAI Realtime 未配置 API Key", provider=self.name)
        url = f"{self._base_url.rstrip('/')}/realtime?model={config.model or self.default_model}"
        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(
                    url,
                    extra_headers={"Authorization": f"Bearer {self._api_key}",
                                   "OpenAI-Beta": "realtime=v1"},
                    max_size=16 * 1024 * 1024,
                ),
                timeout=15,
            )
        except asyncio.TimeoutError as exc:
            raise ProviderTimeout("连接 OpenAI Realtime 超时", provider=self.name) from exc
        except Exception as exc:
            raise ProviderError(f"连接失败: {exc}", provider=self.name) from exc

        await self._ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions": config.instructions,
                "voice": config.voice,
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": config.vad.threshold,
                    "prefix_padding_ms": config.vad.prefix_padding_ms,
                    "silence_duration_ms": config.vad.min_silence_ms,
                },
                "modalities": [m.value for m in config.modalities],
                "temperature": config.temperature,
                "tools": [
                    {"type": "function", "name": t.name, "description": t.description,
                     "parameters": t.parameters or {}} for t in config.tools
                ] or [],
            },
        }))
        self._stream.put_nowait(ProviderEvent(kind="connected", role=TurnRole.SYSTEM))
        self._reader_task = asyncio.create_task(self._reader_loop())

    async def send_audio(self, chunk: AudioChunk) -> None:
        await self._ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": chunk.data.decode("latin-1"),
        }))

    async def send_text(self, text: str, *, role: TurnRole = TurnRole.USER) -> None:
        await self._ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": role.value,
                     "content": [{"type": "input_text", "text": text}]},
        }))

    async def commit_input(self) -> None:
        await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

    async def clear_input(self) -> None:
        await self._ws.send(json.dumps({"type": "input_audio_buffer.clear"}))

    async def request_response(self, *, instructions_suffix: str | None = None) -> None:
        payload: dict[str, Any] = {"type": "response.create"}
        if instructions_suffix:
            payload["instructions"] = instructions_suffix
        await self._ws.send(json.dumps(payload))

    async def cancel_response(self, reason: StopReason = StopReason.INTERRUPTED) -> None:
        await self._ws.send(json.dumps({"type": "response.cancel"}))
        self._stream.put_nowait(
            ProviderEvent(kind="response_cancelled", stop_reason=reason))

    async def submit_tool_result(self, result: ToolResult) -> None:
        await self._ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {"type": "function_call_output",
                     "call_id": result.call_id,
                     "output": json.dumps(result.content, ensure_ascii=False)},
        }))
        await self._ws.send(json.dumps({"type": "response.create"}))

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
            logger.exception("openai reader error: %s", exc)
            self._stream.put_nowait(ProviderEvent(kind="error", error=exc))
        finally:
            self._stream.put_nowait(ProviderEvent(kind="closed", role=TurnRole.SYSTEM))

    def _parse(self, raw: Any) -> ProviderEvent | None:
        try:
            msg = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None
        t = msg.get("type", "")
        if t == "session.created":
            return ProviderEvent(kind="connected", role=TurnRole.SYSTEM)
        if t == "response.audio.delta":
            return ProviderEvent(
                kind="audio_delta",
                audio=AudioChunk(
                    pts_ms=0,
                    data=msg.get("delta", "").encode("latin-1"),
                    sample_rate=24000, encoding=AudioEncoding.PCM16,
                ))
        if t == "response.audio.done":
            return ProviderEvent(kind="audio_done", is_final=True)
        if t == "response.text.delta":
            return ProviderEvent(kind="text_delta", text=msg.get("delta", ""))
        if t == "response.text.done":
            return ProviderEvent(kind="text_done", text=msg.get("text", ""), is_final=True)
        if t == "response.function_call_arguments.done":
            return ProviderEvent(
                kind="tool_call",
                tool_call=ToolCall(
                    id=msg.get("call_id", ""),
                    name=msg.get("name", ""),
                    arguments=json.loads(msg.get("arguments", "{}") or "{}"),
                ))
        if t == "response.done":
            return ProviderEvent(
                kind="response_done", stop_reason=StopReason.COMPLETED,
                usage=msg.get("usage", {}), is_final=True)
        if t == "error":
            return ProviderEvent(kind="error", error=RuntimeError(msg.get("error", {})))
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

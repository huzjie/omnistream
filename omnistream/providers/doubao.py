"""Doubao SeedRealtime 适配器。

对接字节跳动 SeedRealtime（原生音视频全双工大模型）的实时 API。
采用 OpenAI Realtime 兼容协议（``/v1/realtime``），通过 WebSocket 双向流式交互。

注意：示例/默认密钥一律为脱敏格式，真实使用时从环境变量注入。
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError:  # pragma: no cover - 可选依赖
    websockets = None  # type: ignore[assignment]
    ConnectionClosed = Exception  # type: ignore[assignment,misc]

from omnistream.core.errors import (
    CapabilityUnsupported,
    ProviderError,
    ProviderTimeout,
    ProviderUnauthorized,
)
from omnistream.core.clock import now_ms
from omnistream.core.streams import BoundedStream
from omnistream.protocol.enums import AudioEncoding, Modality, StopReason, TurnRole
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


class DoubaoSeedRealtimeProvider(BaseRealtimeProvider):
    """字节 SeedRealtime —— 原生音视频全双工多模态。

    能力：audio_in/out、video_in、native_duplex、interruption、proactive。
    """

    name = "doubao_seedrealtime"
    display_name = "Doubao SeedRealtime"
    vendor = "ByteDance"
    supported_models = ["seed-realtime", "seed-realtime-1.5"]
    default_model = "seed-realtime"
    docs_url = "https://www.volcengine.com/docs/6561"

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
        input_audio_formats=[AudioEncoding.OPUS, AudioEncoding.PCM16],
        output_audio_formats=[AudioEncoding.OPUS, AudioEncoding.PCM16],
        input_sample_rates=[16000, 24000, 48000],
        output_sample_rates=[24000, 48000],
        max_video_fps=10.0,
    )

    def __init__(self, credentials: ProviderCredential | None, settings: Any) -> None:
        if websockets is None:
            raise CapabilityUnsupported(
                "doubao_seedrealtime 需要 websockets 包：pip install websockets",
                provider=self.name,
            )
        self._api_key: str = (credentials.api_key or "") if credentials else ""
        self._base_url: str = (credentials.base_url or "https://api.doubao.com/v1") if credentials else ""
        self._ws: Any = None
        self._stream = BoundedStream[ProviderEvent](maxsize=1024)
        self._reader_task: asyncio.Task | None = None
        self._closed = False

    # ---------------- 连接 ----------------

    async def connect(self, config: SessionConfig) -> None:
        if not self._api_key:
            raise ProviderUnauthorized(
                "Doubao SeedRealtime 未配置 API Key", provider=self.name
            )
        url = self._realtime_url(config)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(url, extra_headers=headers, max_size=16 * 1024 * 1024),
                timeout=15,
            )
        except asyncio.TimeoutError as exc:
            raise ProviderTimeout("连接 SeedRealtime 超时", provider=self.name) from exc
        except Exception as exc:
            raise ProviderError(
                f"连接失败: {exc}", provider=self.name
            ) from exc

        await self._session_update(config)
        self._stream.put_nowait(
            ProviderEvent(kind="connected", role=TurnRole.SYSTEM, is_final=True)
        )
        self._reader_task = asyncio.create_task(self._reader_loop())

    async def _realtime_url(self, config: SessionConfig) -> str:
        base = self._base_url.rstrip("/")
        return f"{base}/realtime?model={config.model or self.default_model}"

    async def _session_update(self, config: SessionConfig) -> None:
        payload: dict[str, Any] = {
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
            },
        }
        if config.tools:
            payload["session"]["tools"] = [
                {"type": "function", "name": t.name, "description": t.description,
                 "parameters": t.parameters or {}} for t in config.tools
            ]
        await self._ws.send(json.dumps(payload))

    # ---------------- 输入 ----------------

    async def send_audio(self, chunk: AudioChunk) -> None:
        await self._ensure_ws()
        await self._ws.send(
            json.dumps({
                "type": "input_audio_buffer.append",
                "audio": chunk.data.decode("latin-1"),
            })
        )

    async def send_video(self, frame: VideoFrame) -> None:
        await self._ensure_ws()
        payload = {
            "type": "input_image.append",
            "image": frame.data.decode("latin-1"),
            "mime": frame.encoding.value,
        }
        if frame.pts_ms is not None:
            payload["pts_ms"] = frame.pts_ms
        await self._ws.send(json.dumps(payload))

    async def send_text(self, text: str, *, role: TurnRole = TurnRole.USER) -> None:
        await self._ensure_ws()
        await self._ws.send(
            json.dumps({"type": "conversation.item.create", "item": {
                "type": "message", "role": role.value, "content": [
                    {"type": "input_text", "text": text}]}})
        )

    async def request_response(self, *, instructions_suffix: str | None = None) -> None:
        await self._ensure_ws()
        payload: dict[str, Any] = {"type": "response.create"}
        if instructions_suffix:
            payload["instructions"] = instructions_suffix
        await self._ws.send(json.dumps(payload))

    async def cancel_response(self, reason: StopReason = StopReason.INTERRUPTED) -> None:
        if self._ws is None:
            return
        await self._ws.send(json.dumps({"type": "response.cancel"}))
        self._stream.put_nowait(
            ProviderEvent(kind="response_cancelled", stop_reason=reason)
        )

    async def submit_tool_result(self, result: ToolResult) -> None:
        await self._ensure_ws()
        await self._ws.send(
            json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": result.call_id,
                    "output": json.dumps(result.content, ensure_ascii=False),
                },
            })
        )
        await self._ws.send(json.dumps({"type": "response.create"}))

    async def update_session(self, patch: dict[str, Any]) -> None:
        await self._ensure_ws()
        await self._ws.send(json.dumps({"type": "session.update", "session": patch}))

    # ---------------- 输出 ----------------

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
            logger.exception("doubao reader error: %s", exc)
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
                    pts_ms=int(msg.get("response_id", "0").replace("resp_", "") or 0) % 1_000_000,
                    data=msg.get("delta", "").encode("latin-1"),
                    sample_rate=24000,
                    encoding=AudioEncoding.PCM16,
                ),
                is_final=False,
            )
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
                ),
            )
        if t == "response.output_audio_done" or t == "response.audio_transcript.done":
            return None
        if t == "response.done":
            return ProviderEvent(
                kind="response_done",
                stop_reason=StopReason.COMPLETED,
                usage=msg.get("usage", {}),
                is_final=True,
            )
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
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if self._ws is not None:
            await self._ws.close()
        self._stream.put_nowait(ProviderEvent(kind="closed", role=TurnRole.SYSTEM))

    async def health(self) -> ProviderHealth:
        return ProviderHealth(
            name=self.name,
            healthy=self._ws is not None and not self._closed,
            checked_at_ms=now_ms(),
        )

    async def _ensure_ws(self) -> None:
        if self._ws is None or self._closed:
            raise ProviderError(
                "Doubao SeedRealtime 尚未连接", provider=self.name
            )

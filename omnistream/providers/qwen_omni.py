"""Qwen-Omni / Qwen3.8-Max 适配器。

对接阿里通义千问 Omni 系列（2.4T MoE / 1M 上下文 / 原生多模态）的实时 API，
通过 DashScope WebSocket 通道实现音视频全双工交互。
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


class QwenOmniProvider(BaseRealtimeProvider):
    """Qwen-Omni / Qwen3.8-Max —— 1M 上下文、原生音视频多模态。"""

    name = "qwen_omni"
    display_name = "Qwen Omni / Qwen3.8-Max"
    vendor = "Alibaba"
    supported_models = ["qwen-omni-turbo", "qwen3.8-max", "qwen-omni-flash"]
    default_model = "qwen-omni-turbo"
    docs_url = "https://help.aliyun.com/zh/model-studio/realtime-api"

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
        input_audio_formats=[AudioEncoding.PCM16, AudioEncoding.OPUS],
        output_audio_formats=[AudioEncoding.PCM16, AudioEncoding.OPUS],
        input_sample_rates=[16000, 24000, 48000],
        output_sample_rates=[24000, 48000],
        max_video_fps=10.0,
    )

    def __init__(self, credentials: ProviderCredential | None, settings: Any) -> None:
        if websockets is None:
            raise CapabilityUnsupported("qwen_omni 需要 websockets 包", provider=self.name)
        self._api_key: str = (credentials.api_key or "") if credentials else ""
        self._base_url: str = (credentials.base_url or "wss://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation") if credentials else "wss://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        self._ws: Any = None
        self._stream = BoundedStream[ProviderEvent](maxsize=1024)
        self._reader_task: asyncio.Task | None = None
        self._closed = False

    async def connect(self, config: SessionConfig) -> None:
        if not self._api_key:
            raise ProviderUnauthorized("Qwen-Omni 未配置 API Key", provider=self.name)
        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(
                    self._base_url,
                    extra_headers={"Authorization": f"Bearer {self._api_key}"},
                    max_size=16 * 1024 * 1024,
                ),
                timeout=15,
            )
        except asyncio.TimeoutError as exc:
            raise ProviderTimeout("连接 Qwen-Omni 超时", provider=self.name) from exc
        except Exception as exc:
            raise ProviderError(f"连接失败: {exc}", provider=self.name) from exc

        await self._ws.send(json.dumps({
            "header": {"message_id": "session-start", "task_id": "session",
                       "action": "run"},
            "parameter": {"model": config.model or self.default_model,
                          "incremental_output": True, "enable_audio": True,
                          "enable_video": Modality.VIDEO.value in
                          [m.value for m in config.modalities]},
            "payload": {"input": {"messages": [{
                "role": "system",
                "content": [{"text": config.instructions}],
            }]}},
        }))
        self._stream.put_nowait(ProviderEvent(kind="connected", role=TurnRole.SYSTEM))
        self._reader_task = asyncio.create_task(self._reader_loop())

    async def send_audio(self, chunk: AudioChunk) -> None:
        await self._ws.send(json.dumps({
            "header": {"message_id": f"audio-{chunk.pts_ms}",
                       "task_id": "session", "action": "run"},
            "payload": {"input": {"audio": chunk.data.decode("latin-1")}},
        }))

    async def send_video(self, frame: VideoFrame) -> None:
        await self._ws.send(json.dumps({
            "header": {"message_id": f"video-{frame.pts_ms}",
                       "task_id": "session", "action": "run"},
            "payload": {"input": {"image": frame.data.decode("latin-1")}},
        }))

    async def send_text(self, text: str, *, role: TurnRole = TurnRole.USER) -> None:
        await self._ws.send(json.dumps({
            "header": {"message_id": "text", "task_id": "session", "action": "run"},
            "payload": {"input": {"messages": [{
                "role": role.value,
                "content": [{"text": text}],
            }]}},
        }))

    async def request_response(self, *, instructions_suffix: str | None = None) -> None:
        if instructions_suffix:
            await self.send_text(f"[系统临时指示] {instructions_suffix}",
                                 role=TurnRole.SYSTEM)

    async def cancel_response(self, reason: StopReason = StopReason.INTERRUPTED) -> None:
        try:
            await self._ws.send(json.dumps({
                "header": {"message_id": "cancel", "task_id": "session",
                           "action": "cancel"},
                "payload": {},
            }))
        except Exception:  # noqa: BLE001
            pass
        self._stream.put_nowait(
            ProviderEvent(kind="response_cancelled", stop_reason=reason))

    async def submit_tool_result(self, result: ToolResult) -> None:
        await self.send_text(json.dumps({
            "tool_result": {"call_id": result.call_id,
                            "output": result.content}},
            ensure_ascii=False), role=TurnRole.USER)

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
            logger.exception("qwen reader error: %s", exc)
            self._stream.put_nowait(ProviderEvent(kind="error", error=exc))
        finally:
            self._stream.put_nowait(ProviderEvent(kind="closed", role=TurnRole.SYSTEM))

    def _parse(self, raw: Any) -> ProviderEvent | None:
        try:
            msg = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None
        if msg.get("header", {}).get("status_code", 0) >= 400:
            return ProviderEvent(kind="error",
                                 error=RuntimeError(msg.get("payload", {})))
        payload = msg.get("payload", {})
        output = payload.get("output", {})
        choices = output.get("choices", [])
        if not choices:
            return None
        content = choices[0].get("message", {}).get("content", [])
        for part in content:
            if part.get("type") == "audio":
                return ProviderEvent(
                    kind="audio_delta",
                    audio=AudioChunk(pts_ms=0,
                                     data=part.get("audio", "").encode("latin-1"),
                                     sample_rate=24000, encoding=AudioEncoding.PCM16))
            if part.get("type") == "text" and part.get("text"):
                return ProviderEvent(kind="text_delta", text=part["text"])
            if part.get("type") == "image":
                return None  # 视频输出暂不转发
        usage = output.get("usage", {})
        if output.get("finish_reason") == "stop":
            return ProviderEvent(kind="response_done",
                                 stop_reason=StopReason.COMPLETED,
                                 usage=usage, is_final=True)
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

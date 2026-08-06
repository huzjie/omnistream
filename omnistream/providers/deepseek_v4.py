"""DeepSeek V4 适配器。

DeepSeek V4-Flash / V4-Pro 提供 OpenAI 兼容的流式文本 + Agent 能力，
本身非实时语音接口；本适配器采用「流式文本 + 端侧 TTS 组合」模式接入，
将 DeepSeek 的能力映射到 OmniStream 全双工框架中（composite kind）。
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from omnistream.core.errors import (
    ProviderError,
    ProviderTimeout,
    ProviderUnauthorized,
)
from omnistream.core.clock import now_ms
from omnistream.core.streams import BoundedStream
from omnistream.protocol.enums import StopReason, TurnRole
from omnistream.providers.base import BaseRealtimeProvider, ProviderEvent
from omnistream.schemas.audio import AudioChunk
from omnistream.schemas.provider import (
    ProviderCapabilities,
    ProviderCredential,
    ProviderHealth,
)
from omnistream.schemas.session import SessionConfig
from omnistream.schemas.tool import ToolCall, ToolResult

logger = logging.getLogger(__name__)


class DeepSeekV4Provider(BaseRealtimeProvider):
    """DeepSeek V4 —— 流式文本（低成本 Agent 主力）。"""

    name = "deepseek_v4"
    display_name = "DeepSeek V4"
    vendor = "DeepSeek"
    supported_models = ["deepseek-v4-flash", "deepseek-v4-pro"]
    default_model = "deepseek-v4-flash"
    docs_url = "https://api-docs.deepseek.com"

    capabilities = ProviderCapabilities(
        native_duplex=False,
        audio_in=False,
        audio_out=False,
        text_in=True,
        text_out=True,
        server_vad=False,
        function_calling=True,
        parallel_tools=True,
        interruption=True,
        proactive=False,
        streaming_transcript=True,
        max_context_tokens=128_000,
    )

    def __init__(self, credentials: ProviderCredential | None, settings: Any) -> None:
        self._api_key: str = (credentials.api_key or "") if credentials else ""
        self._base_url: str = (credentials.base_url or "https://api.deepseek.com") if credentials else "https://api.deepseek.com"
        self._client = httpx.AsyncClient(timeout=60.0)
        self._stream = BoundedStream[ProviderEvent](maxsize=256)
        self._messages: list[dict[str, Any]] = []
        self._gen_task: asyncio.Task | None = None
        self._cancel_flag = asyncio.Event()
        self._closed = False
        self._connected = False

    async def connect(self, config: SessionConfig) -> None:
        if not self._api_key:
            raise ProviderUnauthorized("DeepSeek V4 未配置 API Key", provider=self.name)
        self._messages = [{"role": "system", "content": config.instructions}]
        self._connected = True
        self._stream.put_nowait(ProviderEvent(kind="connected", role=TurnRole.SYSTEM))

    async def send_audio(self, chunk: AudioChunk) -> None:
        # DeepSeek 无原生音频输入：交由感知管线转写为文本后调用 send_text
        return None

    async def send_text(self, text: str, *, role: TurnRole = TurnRole.USER) -> None:
        self._messages.append({"role": role.value, "content": text})

    async def request_response(self, *, instructions_suffix: str | None = None) -> None:
        self._cancel_flag.clear()
        messages = self._messages
        if instructions_suffix:
            messages = [*messages, {"role": "system", "content": instructions_suffix}]
        self._gen_task = asyncio.create_task(self._generate(messages))

    async def _generate(self, messages: list[dict[str, Any]]) -> None:
        try:
            payload = {
                "model": self.default_model,
                "messages": messages,
                "stream": True,
                "temperature": 0.8,
            }
            async with self._client.stream(
                "POST", f"{self._base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            ) as resp:
                if resp.status_code == 401:
                    raise ProviderUnauthorized("DeepSeek 鉴权失败", provider=self.name)
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if self._cancel_flag.is_set():
                        self._stream.put_nowait(
                            ProviderEvent(kind="response_cancelled",
                                          stop_reason=StopReason.INTERRUPTED))
                        return
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        self._stream.put_nowait(
                            ProviderEvent(kind="text_delta", text=delta["content"]))
                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            fn = tc.get("function", {})
                            self._stream.put_nowait(
                                ProviderEvent(kind="tool_call",
                                              tool_call=ToolCall(
                                                  id=tc.get("id", ""),
                                                  name=fn.get("name", ""),
                                                  arguments=json.loads(
                                                      fn.get("arguments", "{}") or "{}"),
                                              )))
            self._stream.put_nowait(
                ProviderEvent(kind="response_done",
                              stop_reason=StopReason.COMPLETED, is_final=True))
            self._messages.append({"role": "assistant", "content": "(completed)"})
        except asyncio.CancelledError:
            self._stream.put_nowait(
                ProviderEvent(kind="response_cancelled",
                              stop_reason=StopReason.CANCELLED))
        except Exception as exc:  # noqa: BLE001
            logger.exception("deepseek generate error: %s", exc)
            self._stream.put_nowait(ProviderEvent(kind="error", error=exc))

    async def cancel_response(self, reason: StopReason = StopReason.INTERRUPTED) -> None:
        self._cancel_flag.set()
        if self._gen_task:
            self._gen_task.cancel()

    async def submit_tool_result(self, result: ToolResult) -> None:
        self._messages.append({
            "role": "tool",
            "tool_call_id": result.call_id,
            "content": json.dumps(result.content, ensure_ascii=False),
        })
        await self.request_response()

    def events(self):
        return self._stream.__aiter__()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._gen_task:
            self._gen_task.cancel()
            try:
                await self._gen_task
            except Exception:  # noqa: BLE001
                pass
        await self._client.aclose()
        self._stream.put_nowait(ProviderEvent(kind="closed", role=TurnRole.SYSTEM))

    async def health(self) -> ProviderHealth:
        return ProviderHealth(
            name=self.name, healthy=self._connected and not self._closed,
            checked_at_ms=now_ms())

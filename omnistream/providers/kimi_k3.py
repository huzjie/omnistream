"""Kimi K3 适配器。

Moonshot Kimi K3（2.8T 开源 MoE）—— 通过 OpenAI 兼容 API 提供流式文本
与超长上下文（K3 原生 1M+），本适配器采用「流式文本 + 组合」模式。
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


class KimiK3Provider(BaseRealtimeProvider):
    """Kimi K3 —— 超长上下文流式文本。"""

    name = "kimi_k3"
    display_name = "Kimi K3"
    vendor = "Moonshot AI"
    supported_models = ["kimi-k3", "kimi-k3-2.8t"]
    default_model = "kimi-k3"
    docs_url = "https://platform.moonshot.cn/docs"

    capabilities = ProviderCapabilities(
        native_duplex=False,
        audio_in=False,
        audio_out=False,
        text_in=True,
        text_out=True,
        function_calling=True,
        parallel_tools=True,
        interruption=True,
        streaming_transcript=True,
        max_context_tokens=1_000_000,
    )

    def __init__(self, credentials: ProviderCredential | None, settings: Any) -> None:
        self._api_key: str = (credentials.api_key or "") if credentials else ""
        self._base_url: str = (credentials.base_url or "https://api.moonshot.cn/v1") if credentials else "https://api.moonshot.cn/v1"
        self._client = httpx.AsyncClient(timeout=60.0)
        self._stream = BoundedStream[ProviderEvent](maxsize=256)
        self._messages: list[dict[str, Any]] = []
        self._gen_task: asyncio.Task | None = None
        self._cancel_flag = asyncio.Event()
        self._closed = False
        self._connected = False

    async def connect(self, config: SessionConfig) -> None:
        if not self._api_key:
            raise ProviderUnauthorized("Kimi K3 未配置 API Key", provider=self.name)
        self._messages = [{"role": "system", "content": config.instructions}]
        self._connected = True
        self._stream.put_nowait(ProviderEvent(kind="connected", role=TurnRole.SYSTEM))

    async def send_audio(self, chunk: AudioChunk) -> None:
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
                    raise ProviderUnauthorized("Kimi 鉴权失败", provider=self.name)
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
                    if delta.get("content"):
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
        except asyncio.CancelledError:
            self._stream.put_nowait(
                ProviderEvent(kind="response_cancelled",
                              stop_reason=StopReason.CANCELLED))
        except Exception as exc:  # noqa: BLE001
            logger.exception("kimi generate error: %s", exc)
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

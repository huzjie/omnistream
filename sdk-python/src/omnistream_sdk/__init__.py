"""OmniStream Python SDK。

用法：:

    import asyncio
    from omnistream_sdk import RealtimeClient

    async def main():
        client = RealtimeClient(provider="mock")
        await client.connect()
        await client.send_text("你好")
        async for ev in client.events():
            if ev.kind == "text_delta":
                print(ev.text, end="")
            if ev.kind == "response_done":
                break
        await client.close()

    asyncio.run(main())
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

API_ROOT = "http://localhost:8000"


@dataclass(slots=True)
class ClientEvent:
    kind: str
    ts_ms: int = 0
    text: str | None = None
    audio: bytes | None = None
    tool_call: dict[str, Any] | None = None
    stop_reason: str | None = None
    role: str = "assistant"
    is_final: bool = False
    usage: dict[str, int] | None = None
    error: str | None = None


@dataclass(slots=True)
class RealtimeClient:
    """实时全双工多模态客户端。"""

    provider: str = "mock"
    model: str | None = None
    modalities: list[str] = field(default_factory=lambda: ["text", "audio"])
    instructions: str = "你是一个自然、简洁、可被随时打断的语音助手。"
    voice: str = "default"
    base_url: str = API_ROOT
    api_key: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)

    session_id: str | None = field(default=None, init=False, repr=False)
    _ws: Any = field(default=None, init=False, repr=False)
    _queue: asyncio.Queue[ClientEvent] = field(
        default_factory=lambda: asyncio.Queue(maxsize=256), init=False, repr=False)
    _reader: asyncio.Task | None = field(default=None, init=False, repr=False)

    async def connect(self) -> str:
        """创建会话并建立实时通道，返回 session_id。"""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{self.base_url}/api/sessions",
                headers={**headers, "Content-Type": "application/json"},
                json={
                    "provider": self.provider,
                    "model": self.model,
                    "modalities": self.modalities,
                    "instructions": self.instructions,
                    "voice": self.voice,
                    "tools": self.tools,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            self.session_id = data.get("id") or data.get("session_id")

        import websockets

        ws_url = (self.base_url.replace("http", "ws")
                  + f"/api/sessions/{self.session_id}/realtime")
        self._ws = await websockets.connect(ws_url)
        self._reader = asyncio.create_task(self._read_loop())
        return self.session_id  # type: ignore[return-value]

    async def _read_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                ev = self._parse(raw)
                if ev is not None:
                    await self._queue.put(ev)
        except Exception as exc:  # noqa: BLE001
            await self._queue.put(ClientEvent(kind="error", error=str(exc)))
        finally:
            await self._queue.put(ClientEvent(kind="closed"))

    def _parse(self, raw: Any) -> ClientEvent | None:
        try:
            msg = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        except (TypeError, json.JSONDecodeError):
            return None
        kind = msg.get("kind", msg.get("type", ""))
        if not kind:
            return None
        return ClientEvent(
            kind=kind,
            ts_ms=msg.get("ts_ms", 0),
            text=msg.get("text") or msg.get("delta"),
            stop_reason=msg.get("stop_reason"),
            role=msg.get("role", "assistant"),
            is_final=bool(msg.get("is_final")),
            usage=msg.get("usage"),
            error=msg.get("error"),
            tool_call=msg.get("tool_call"),
        )

    async def send_text(self, text: str) -> None:
        await self._send({"type": "text", "text": text})

    async def request_response(self, instructions_suffix: str | None = None) -> None:
        payload: dict[str, Any] = {"type": "request_response"}
        if instructions_suffix:
            payload["instructions_suffix"] = instructions_suffix
        await self._send(payload)

    async def cancel_response(self) -> None:
        await self._send({"type": "cancel_response"})

    async def submit_tool_result(self, call_id: str, result: Any) -> None:
        await self._send({"type": "tool_result", "call_id": call_id, "result": result})

    async def events(self) -> AsyncIterator[ClientEvent]:
        while True:
            ev = await self._queue.get()
            yield ev
            if ev.kind == "closed":
                break

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("未连接，请先调用 connect()")
        await self._ws.send(json.dumps(payload))

    async def close(self) -> None:
        if self._reader:
            self._reader.cancel()
            try:
                await self._reader
            except asyncio.CancelledError:
                pass
        if self._ws is not None:
            await self._ws.close()
        self._ws = None

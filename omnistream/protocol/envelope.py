"""统一消息外壳。所有 WS 文本消息 = 一个 Envelope 的 JSON。"""
from __future__ import annotations

from typing import Any

import orjson
from pydantic import BaseModel, ConfigDict, Field

from omnistream.core.clock import now_ms
from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import ProtocolError
from omnistream.core.ids import new_event_id
from omnistream.protocol.enums import PROTOCOL_VERSION


class Envelope(BaseModel):
    """线上消息统一结构。

    JSON 形态::

        {
          "v": "oms.v1",
          "type": "response.audio.delta",
          "event_id": "evt_01J...",
          "ts": 1770000000123,
          "session_id": "sess_01J...",
          "seq": 42,
          "payload": {...}
        }
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    v: str = Field(default=PROTOCOL_VERSION, description="协议版本")
    type: str = Field(description="事件类型，取自 ClientEventType/ServerEventType")
    event_id: str = Field(default_factory=lambda: new_event_id(), description="全局唯一事件 ID")
    ts: int = Field(default_factory=now_ms, description="服务端 UTC 毫秒时间戳")
    session_id: str | None = Field(default=None)
    seq: int | None = Field(default=None, description="同会话内单调递增序号，用于断线续传")
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_wire(self) -> bytes:
        return orjson.dumps(self.model_dump(exclude_none=True))

    @classmethod
    def from_wire(cls, raw: str | bytes) -> "Envelope":
        try:
            data = orjson.loads(raw)
        except orjson.JSONDecodeError as exc:
            raise ProtocolError("消息不是合法 JSON", code=ErrorCode.PROTOCOL_MALFORMED, cause=exc) from exc
        if not isinstance(data, dict):
            raise ProtocolError("消息根必须是对象", code=ErrorCode.PROTOCOL_MALFORMED)
        version = data.get("v", PROTOCOL_VERSION)
        if version != PROTOCOL_VERSION:
            raise ProtocolError(
                f"协议版本不匹配: 期望 {PROTOCOL_VERSION}, 收到 {version}",
                code=ErrorCode.PROTOCOL_VERSION_MISMATCH,
                detail={"expected": PROTOCOL_VERSION, "got": version},
            )
        try:
            return cls.model_validate(data)
        except Exception as exc:  # pydantic ValidationError
            raise ProtocolError(f"消息结构非法: {exc}", code=ErrorCode.PROTOCOL_MALFORMED, cause=exc) from exc

    @classmethod
    def server(
        cls,
        type_: str,
        payload: dict[str, Any] | BaseModel | None = None,
        *,
        session_id: str | None = None,
        seq: int | None = None,
    ) -> "Envelope":
        if isinstance(payload, BaseModel):
            body = payload.model_dump(mode="json", exclude_none=True)
        else:
            body = payload or {}
        return cls(type=type_, payload=body, session_id=session_id, seq=seq)

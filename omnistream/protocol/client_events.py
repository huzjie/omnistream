"""上行事件模型（客户端 → 服务端）。

本模块**刻意只依赖 core 与 protocol.enums**，不引用 ``omnistream.schemas``：
协议层要能被 SDK 单独抽取，也避免与领域模型形成导入环。凡是"整块配置"的
字段（如 ``session.update`` 的 session 体）一律用 ``dict[str, Any]`` 承载，
由编排层用 :class:`omnistream.schemas.session.SessionConfig` 做二次校验。
"""
from __future__ import annotations

import base64
import binascii
from typing import Any, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import ProtocolError
from omnistream.protocol.enums import (
    AudioEncoding,
    ClientEventType,
    MemoryLayer,
    Modality,
    StopReason,
    TurnRole,
    VideoEncoding,
)
from omnistream.protocol.envelope import Envelope

__all__ = [
    "CLIENT_PAYLOAD_MODELS",
    "ClientEvent",
    "ClientPayload",
    "InputAudioAppend",
    "InputAudioClear",
    "InputAudioCommit",
    "InputTextSend",
    "InputVideoFrame",
    "MemoryDelete",
    "MemoryUpsert",
    "Ping",
    "ResponseCancel",
    "ResponseCreate",
    "SessionClose",
    "SessionUpdate",
    "ToolResultIn",
    "parse_client_event",
]


class ProtoModel(BaseModel):
    """协议层模型基类：未知字段一律拒绝，防止客户端拼错字段名却静默生效。"""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, use_enum_values=False)


def _decode_b64(value: str, field_name: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ProtocolError(
            f"字段 {field_name} 不是合法 base64",
            code=ErrorCode.PROTOCOL_MALFORMED,
            detail={"field": field_name},
            cause=exc,
        ) from exc


# ---------------------------------------------------------------- 会话
class SessionUpdate(ProtoModel):
    """``session.update`` —— 中途更新会话配置（换音色、改指令、增删工具）。"""

    session: dict[str, Any] = Field(default_factory=dict, description="SessionConfig 的部分字段")


class SessionClose(ProtoModel):
    """``session.close`` —— 客户端主动结束会话。"""

    reason: str = Field(default="client_request", max_length=200)


class Ping(ProtoModel):
    """``ping`` —— 心跳，服务端回 ``pong`` 并原样带回 nonce。"""

    nonce: str = Field(default="", max_length=64)


# ---------------------------------------------------------------- 输入
class InputAudioAppend(ProtoModel):
    """``input_audio.append`` —— 追加一段用户音频。

    高频音频建议走二进制帧（见 :mod:`omnistream.protocol.binary`）；
    本事件用于低频、调试或不便发二进制的客户端。
    """

    audio: str = Field(description="base64 编码的音频负载")
    pts_ms: int = Field(default=0, ge=0)
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    channels: int = Field(default=1, ge=1, le=2)
    encoding: AudioEncoding = AudioEncoding.PCM16

    @field_validator("audio")
    @classmethod
    def _validate_audio(cls, v: str) -> str:
        _decode_b64(v, "audio")
        return v

    @property
    def data(self) -> bytes:
        """解码后的原始音频字节。"""
        return _decode_b64(self.audio, "audio")


class InputAudioCommit(ProtoModel):
    """``input_audio.commit`` —— 手动 VAD 模式下标记本轮输入结束。"""

    create_response: bool = Field(default=True, description="提交后是否立即请求生成")


class InputAudioClear(ProtoModel):
    """``input_audio.clear`` —— 丢弃尚未提交的输入缓冲。"""

    reason: str = Field(default="user_cancel", max_length=120)


class InputVideoFrame(ProtoModel):
    """``input_video.frame`` —— 上传一帧画面。"""

    image: str = Field(description="base64 编码的图像数据")
    pts_ms: int = Field(default=0, ge=0)
    width: int = Field(default=0, ge=0, le=8192)
    height: int = Field(default=0, ge=0, le=8192)
    encoding: VideoEncoding = VideoEncoding.JPEG

    @field_validator("image")
    @classmethod
    def _validate_image(cls, v: str) -> str:
        _decode_b64(v, "image")
        return v

    @property
    def data(self) -> bytes:
        """解码后的原始图像字节。"""
        return _decode_b64(self.image, "image")


class InputTextSend(ProtoModel):
    """``input_text.send`` —— 纯文本输入（打字模式 / 无麦环境）。"""

    text: str = Field(min_length=1, max_length=32_000)
    role: TurnRole = TurnRole.USER
    create_response: bool = True


# ---------------------------------------------------------------- 生成控制
class ResponseCreate(ProtoModel):
    """``response.create`` —— 请求模型生成一轮回复。"""

    instructions_suffix: str | None = Field(default=None, max_length=8_000)
    modalities: list[Modality] | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1, le=32_768)


class ResponseCancel(ProtoModel):
    """``response.cancel`` —— 取消当前生成（客户端侧打断）。"""

    reason: StopReason = StopReason.INTERRUPTED
    truncate_at_ms: int | None = Field(default=None, ge=0,
                                       description="已播放到的位置，用于服务端裁剪上下文")


class ToolResultIn(ProtoModel):
    """``tool.result`` —— 回填客户端侧工具的执行结果。"""

    call_id: str = Field(min_length=1, max_length=64)
    ok: bool = True
    content: Any = None
    error: str | None = Field(default=None, max_length=2_000)
    elapsed_ms: float = Field(default=0.0, ge=0.0)


# ---------------------------------------------------------------- 记忆
class MemoryUpsert(ProtoModel):
    """``memory.upsert`` —— 客户端显式写入一条记忆。"""

    content: str = Field(min_length=1, max_length=8_000)
    layer: MemoryLayer = MemoryLayer.SESSION
    key: str | None = Field(default=None, max_length=200)
    subject: str = Field(default="user", max_length=64)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    ttl_s: int | None = Field(default=None, ge=1)


class MemoryDelete(ProtoModel):
    """``memory.delete`` —— 按 id 或 (layer, key) 删除记忆。"""

    id: str | None = Field(default=None, max_length=64)
    layer: MemoryLayer | None = None
    key: str | None = Field(default=None, max_length=200)


ClientPayload = Union[  # noqa: UP007 - pydantic 需要显式 Union 以便 TS SDK 生成
    SessionUpdate,
    SessionClose,
    Ping,
    InputAudioAppend,
    InputAudioCommit,
    InputAudioClear,
    InputVideoFrame,
    InputTextSend,
    ResponseCreate,
    ResponseCancel,
    ToolResultIn,
    MemoryUpsert,
    MemoryDelete,
]

#: 事件类型 -> 负载模型。新增上行事件必须在此登记。
CLIENT_PAYLOAD_MODELS: dict[ClientEventType, type[ProtoModel]] = {
    ClientEventType.SESSION_UPDATE: SessionUpdate,
    ClientEventType.SESSION_CLOSE: SessionClose,
    ClientEventType.PING: Ping,
    ClientEventType.INPUT_AUDIO_APPEND: InputAudioAppend,
    ClientEventType.INPUT_AUDIO_COMMIT: InputAudioCommit,
    ClientEventType.INPUT_AUDIO_CLEAR: InputAudioClear,
    ClientEventType.INPUT_VIDEO_FRAME: InputVideoFrame,
    ClientEventType.INPUT_TEXT: InputTextSend,
    ClientEventType.RESPONSE_CREATE: ResponseCreate,
    ClientEventType.RESPONSE_CANCEL: ResponseCancel,
    ClientEventType.TOOL_RESULT: ToolResultIn,
    ClientEventType.MEMORY_UPSERT: MemoryUpsert,
    ClientEventType.MEMORY_DELETE: MemoryDelete,
}


class ClientEvent(BaseModel):
    """已解析的上行事件：外壳字段 + 强类型负载。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: ClientEventType
    data: ClientPayload
    event_id: str = ""
    ts: int = 0
    session_id: str | None = None
    seq: int | None = None

    def as_envelope(self) -> Envelope:
        """还原为线上 Envelope（SDK 回放与录制归档用）。"""
        return Envelope(
            type=str(self.type),
            payload=self.data.model_dump(mode="json", exclude_none=True),
            session_id=self.session_id,
            seq=self.seq,
        )


def parse_client_event(source: Envelope | str | bytes) -> ClientEvent:
    """把线上消息解析成强类型 :class:`ClientEvent`。

    Args:
        source: 已解析的 :class:`Envelope`，或原始 JSON 文本/字节。

    Raises:
        ProtocolError: 版本不符（``OMS-3000``）、结构非法（``OMS-3001``）、
            未知事件类型（``OMS-3002``）。
    """
    envelope = source if isinstance(source, Envelope) else Envelope.from_wire(source)
    try:
        event_type = ClientEventType(envelope.type)
    except ValueError as exc:
        raise ProtocolError(
            f"未知的上行事件类型: {envelope.type!r}",
            code=ErrorCode.PROTOCOL_UNKNOWN_EVENT,
            detail={"type": envelope.type, "known": [str(t) for t in ClientEventType]},
            cause=exc,
        ) from exc

    model = CLIENT_PAYLOAD_MODELS[event_type]
    try:
        payload = model.model_validate(envelope.payload)
    except ProtocolError:
        raise
    except Exception as exc:
        raise ProtocolError(
            f"事件 {event_type} 的负载校验失败: {exc}",
            code=ErrorCode.PROTOCOL_MALFORMED,
            detail={"type": str(event_type)},
            cause=exc,
        ) from exc

    return ClientEvent(
        type=event_type,
        data=payload,  # type: ignore[arg-type]
        event_id=envelope.event_id,
        ts=envelope.ts,
        session_id=envelope.session_id,
        seq=envelope.seq,
    )

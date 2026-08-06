"""``ProtocolCodec`` —— 文本 JSON 与二进制帧的统一编解码入口。

WebSocket 处理器只需要跟这个类打交道：

    codec = ProtocolCodec(session_id=sid)
    decoded = codec.decode(raw)          # raw 可以是 str 也可以是 bytes
    if isinstance(decoded, ClientEvent): ...
    elif isinstance(decoded, BinaryFrame): ...

    await ws.send_bytes(codec.encode_audio(turn_id, pcm, pts_ms=pts))
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import ProtocolError
from omnistream.protocol.binary import (
    BinaryFrame,
    decode_frame,
    encode_audio,
    encode_video,
    is_binary_frame,
)
from omnistream.protocol.client_events import ClientEvent, parse_client_event
from omnistream.protocol.enums import ServerEventType
from omnistream.protocol.envelope import Envelope
from omnistream.protocol.server_events import ServerModel, dump_server_event

__all__ = ["DecodedMessage", "ProtocolCodec"]

#: ``decode`` 的返回类型：要么是强类型上行事件，要么是一帧二进制媒体。
DecodedMessage = ClientEvent | BinaryFrame


@dataclass(slots=True)
class ProtocolCodec:
    """一个 WS 连接一个实例，负责序号维护与帧分流。

    Attributes:
        session_id: 当前会话 ID，编码二进制帧时写入帧头。
        strict_seq: 为 True 时，上行序号出现空洞会抛 ``OMS-3003``。
    """

    session_id: str | None = None
    strict_seq: bool = False
    #: 下行序号，每编码一条下行消息自增。
    out_seq: int = 0
    #: 最近一次收到的上行序号。
    last_in_seq: int | None = field(default=None)
    #: 统计信息。
    decoded_text: int = 0
    decoded_binary: int = 0
    encoded_text: int = 0
    encoded_binary: int = 0

    # ------------------------------------------------------------- 解码
    def decode(self, raw: str | bytes | bytearray) -> DecodedMessage:
        """解码一条上行消息。

        ``bytes`` 且以 magic ``OMS1`` 开头 → 二进制媒体帧；
        其余一律按 JSON Envelope 解析成 :class:`ClientEvent`。
        """
        if isinstance(raw, (bytes, bytearray)):
            payload = bytes(raw)
            if is_binary_frame(payload):
                self.decoded_binary += 1
                return decode_frame(payload)
            return self._decode_text(payload)
        return self._decode_text(raw)

    def _decode_text(self, raw: str | bytes) -> ClientEvent:
        event = parse_client_event(raw)
        self.decoded_text += 1
        self._check_seq(event.seq)
        if event.session_id and self.session_id and event.session_id != self.session_id:
            raise ProtocolError(
                "消息中的 session_id 与连接所属会话不一致",
                code=ErrorCode.PROTOCOL_MALFORMED,
                detail={"expected": self.session_id, "got": event.session_id},
            )
        return event

    def _check_seq(self, seq: int | None) -> None:
        if seq is None:
            return
        if self.last_in_seq is not None and seq != self.last_in_seq + 1 and self.strict_seq:
            raise ProtocolError(
                f"上行序号出现空洞: 期望 {self.last_in_seq + 1}, 收到 {seq}",
                code=ErrorCode.PROTOCOL_SEQ_GAP,
                detail={"expected": self.last_in_seq + 1, "got": seq},
            )
        self.last_in_seq = seq

    # ------------------------------------------------------------- 编码
    def next_seq(self) -> int:
        """取下一个下行序号。"""
        self.out_seq += 1
        return self.out_seq

    def encode(
        self,
        event_type: ServerEventType,
        payload: ServerModel | dict[str, Any] | None = None,
    ) -> bytes:
        """编码一条下行 JSON 事件为 UTF-8 字节。"""
        envelope = dump_server_event(
            event_type, payload, session_id=self.session_id, seq=self.next_seq()
        )
        self.encoded_text += 1
        return envelope.to_wire()

    def encode_envelope(self, envelope: Envelope) -> bytes:
        """编码一个已经构造好的 Envelope（补齐 session_id 与序号）。"""
        if envelope.session_id is None:
            envelope.session_id = self.session_id
        if envelope.seq is None:
            envelope.seq = self.next_seq()
        self.encoded_text += 1
        return envelope.to_wire()

    def encode_audio(self, data: bytes, *, pts_ms: int, sample_rate: int = 24000,
                     channels: int = 1, encoding: str = "pcm16") -> bytes:
        """编码一帧下行音频为二进制帧。"""
        self.encoded_binary += 1
        return encode_audio(
            self.session_id or "",
            pts_ms,
            data,
            sample_rate=sample_rate,
            channels=channels,
            encoding=encoding,
            seq=self.next_seq(),
        )

    def encode_video(self, data: bytes, *, pts_ms: int, width: int, height: int,
                     encoding: str = "jpeg", keyframe_score: float = 0.0) -> bytes:
        """编码一帧下行视频为二进制帧。"""
        self.encoded_binary += 1
        return encode_video(
            self.session_id or "",
            pts_ms,
            data,
            width=width,
            height=height,
            encoding=encoding,
            keyframe_score=keyframe_score,
        )

    # ------------------------------------------------------------- 观测
    def stats(self) -> dict[str, int | str | None]:
        return {
            "session_id": self.session_id,
            "out_seq": self.out_seq,
            "last_in_seq": self.last_in_seq,
            "decoded_text": self.decoded_text,
            "decoded_binary": self.decoded_binary,
            "encoded_text": self.encoded_text,
            "encoded_binary": self.encoded_binary,
        }

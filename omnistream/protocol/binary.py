"""二进制媒体帧格式 —— 避免 base64 带来的 33% 膨胀与 GC 压力。

线格式（全部小端）::

    offset  size  field
    0       4     magic  = b"OMS1"
    4       1     kind   (1=audio, 2=video)
    5       2     header_len (uint16)
    7       H     header (JSON, UTF-8)
    7+H     N     payload (raw bytes)

header 示例（audio）: {"sid":"sess_x","pts":1234,"sr":16000,"ch":1,"enc":"pcm16","seq":42}
header 示例（video）: {"sid":"sess_x","pts":1240,"w":768,"h":432,"enc":"jpeg","ks":0.82}
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, Literal

import orjson

from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import ProtocolError

MAGIC = b"OMS1"
KIND_AUDIO = 1
KIND_VIDEO = 2
_HEADER_STRUCT = struct.Struct("<4sBH")
MAX_HEADER = 4096
MAX_PAYLOAD = 4 * 1024 * 1024


@dataclass(slots=True)
class BinaryFrame:
    kind: Literal[1, 2]
    header: dict[str, Any]
    payload: bytes

    @property
    def is_audio(self) -> bool:
        return self.kind == KIND_AUDIO

    @property
    def is_video(self) -> bool:
        return self.kind == KIND_VIDEO


def encode_frame(kind: int, header: dict[str, Any], payload: bytes) -> bytes:
    head = orjson.dumps(header)
    if len(head) > MAX_HEADER:
        raise ProtocolError("二进制帧头过大", code=ErrorCode.PROTOCOL_MALFORMED)
    if len(payload) > MAX_PAYLOAD:
        raise ProtocolError("二进制帧负载过大", code=ErrorCode.PROTOCOL_MALFORMED)
    return _HEADER_STRUCT.pack(MAGIC, kind, len(head)) + head + payload


def decode_frame(raw: bytes) -> BinaryFrame:
    if len(raw) < _HEADER_STRUCT.size:
        raise ProtocolError("二进制帧过短", code=ErrorCode.PROTOCOL_MALFORMED)
    magic, kind, header_len = _HEADER_STRUCT.unpack_from(raw, 0)
    if magic != MAGIC:
        raise ProtocolError("二进制帧 magic 非法", code=ErrorCode.PROTOCOL_MALFORMED)
    if kind not in (KIND_AUDIO, KIND_VIDEO):
        raise ProtocolError(f"未知帧类型 {kind}", code=ErrorCode.PROTOCOL_MALFORMED)
    start = _HEADER_STRUCT.size
    end = start + header_len
    if len(raw) < end:
        raise ProtocolError("二进制帧头被截断", code=ErrorCode.PROTOCOL_MALFORMED)
    try:
        header = orjson.loads(raw[start:end])
    except orjson.JSONDecodeError as exc:
        raise ProtocolError("二进制帧头不是合法 JSON", code=ErrorCode.PROTOCOL_MALFORMED, cause=exc) from exc
    return BinaryFrame(kind=kind, header=header, payload=raw[end:])  # type: ignore[arg-type]


def encode_audio(session_id: str, pts_ms: int, data: bytes, *, sample_rate: int = 16000,
                 channels: int = 1, encoding: str = "pcm16", seq: int | None = None) -> bytes:
    header: dict[str, Any] = {"sid": session_id, "pts": pts_ms, "sr": sample_rate,
                              "ch": channels, "enc": encoding}
    if seq is not None:
        header["seq"] = seq
    return encode_frame(KIND_AUDIO, header, data)


def encode_video(session_id: str, pts_ms: int, data: bytes, *, width: int, height: int,
                 encoding: str = "jpeg", keyframe_score: float = 0.0) -> bytes:
    header = {"sid": session_id, "pts": pts_ms, "w": width, "h": height,
              "enc": encoding, "ks": round(keyframe_score, 4)}
    return encode_frame(KIND_VIDEO, header, data)


def is_binary_frame(raw: bytes) -> bool:
    """快速判断一段字节是否是本协议的二进制帧（供 codec 分流）。"""
    return len(raw) >= len(MAGIC) and raw[: len(MAGIC)] == MAGIC

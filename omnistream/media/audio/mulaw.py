"""G.711 μ-law 编解码（电话侧接入预留）。

纯 numpy 查表实现，零外部依赖，性能约 100M 样本/秒量级，实时链路开销可忽略。

算法严格遵循 ITU-T G.711：

* 偏置 ``BIAS = 0x84``，削顶 ``CLIP = 32635``
* 8 位输出 = ``~(sign | exponent<<4 | mantissa)``
* 解码时补回半个量化步长，保证往返误差最小
"""
from __future__ import annotations

import numpy as np

from omnistream.media.audio import pcm
from omnistream.protocol.enums import AudioEncoding
from omnistream.schemas.audio import AudioChunk

__all__ = [
    "MULAW_BIAS",
    "MULAW_CLIP",
    "decode",
    "decode_to_float32",
    "decode_chunk",
    "encode",
    "encode_chunk",
    "encode_from_float32",
]

MULAW_BIAS: int = 0x84
MULAW_CLIP: int = 32635


def _build_encode_table() -> np.ndarray:
    """构造 int16 → μ-law 的 65536 项查表。"""
    samples = np.arange(-32768, 32768, dtype=np.int32)
    sign = np.where(samples < 0, 0x80, 0x00).astype(np.int32)
    magnitude = np.abs(samples)
    magnitude = np.minimum(magnitude, MULAW_CLIP) + MULAW_BIAS
    # exponent = 位段序号，等价于标准 exp_lut[(magnitude >> 7) & 0xFF]：
    # exp = k  <=>  magnitude >= (0x80 << k)，k 取 1..7；magnitude < 0x100 时 exp = 0
    exponent = np.zeros_like(magnitude)
    for seg in range(1, 8):
        exponent = np.where(magnitude >= (0x80 << seg), seg, exponent)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    encoded = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return encoded.astype(np.uint8)


def _build_decode_table() -> np.ndarray:
    """构造 μ-law → int16 的 256 项查表。"""
    codes = np.arange(256, dtype=np.int32)
    inverted = ~codes & 0xFF
    sign = inverted & 0x80
    exponent = (inverted >> 4) & 0x07
    mantissa = inverted & 0x0F
    magnitude = ((mantissa << 3) + MULAW_BIAS) << exponent
    magnitude = magnitude - MULAW_BIAS
    values = np.where(sign != 0, -magnitude, magnitude)
    return np.clip(values, -32768, 32767).astype(np.int16)


_ENCODE_TABLE: np.ndarray = _build_encode_table()
_DECODE_TABLE: np.ndarray = _build_decode_table()


def encode(pcm16: bytes) -> bytes:
    """pcm16 小端字节 → μ-law 字节（长度减半）。"""
    if not pcm16:
        return b""
    if len(pcm16) % 2:
        pcm16 = pcm16[:-1]
    samples = np.frombuffer(pcm16, dtype="<i2").astype(np.int32)
    return _ENCODE_TABLE[samples + 32768].tobytes()


def decode(mulaw: bytes) -> bytes:
    """μ-law 字节 → pcm16 小端字节（长度翻倍）。"""
    if not mulaw:
        return b""
    codes = np.frombuffer(mulaw, dtype=np.uint8)
    return _DECODE_TABLE[codes].astype("<i2").tobytes()


def encode_from_float32(arr: np.ndarray) -> bytes:
    """``float32`` [-1,1] 数组 → μ-law 字节。"""
    return encode(pcm.float32_to_bytes(arr, AudioEncoding.PCM16))


def decode_to_float32(mulaw: bytes) -> np.ndarray:
    """μ-law 字节 → ``float32`` [-1,1] 数组。"""
    return pcm.bytes_to_float32(decode(mulaw), AudioEncoding.PCM16)


def encode_chunk(chunk: AudioChunk) -> AudioChunk:
    """把 pcm16 的 :class:`AudioChunk` 编码为 μ-law。"""
    if chunk.encoding == AudioEncoding.MULAW:
        return chunk
    raw = chunk.data
    if chunk.encoding == AudioEncoding.PCM_F32:
        raw = pcm.float32_to_bytes(pcm.bytes_to_float32(raw, AudioEncoding.PCM_F32),
                                   AudioEncoding.PCM16)
    return AudioChunk(
        pts_ms=chunk.pts_ms,
        data=encode(raw),
        sample_rate=chunk.sample_rate,
        channels=chunk.channels,
        encoding=AudioEncoding.MULAW,
        capture_ts_ns=chunk.capture_ts_ns,
        speaker=chunk.speaker,
    )


def decode_chunk(chunk: AudioChunk) -> AudioChunk:
    """把 μ-law 的 :class:`AudioChunk` 解码为 pcm16。"""
    if chunk.encoding != AudioEncoding.MULAW:
        return chunk
    return AudioChunk(
        pts_ms=chunk.pts_ms,
        data=decode(chunk.data),
        sample_rate=chunk.sample_rate,
        channels=chunk.channels,
        encoding=AudioEncoding.PCM16,
        capture_ts_ns=chunk.capture_ts_ns,
        speaker=chunk.speaker,
    )

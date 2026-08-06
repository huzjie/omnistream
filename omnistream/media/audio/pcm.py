"""PCM 工具：字节 ↔ ndarray、声道混合、分帧、拼接、电平计算。

约定
----
* 内部统一用 ``float32`` 且范围 ``[-1, 1]`` 表示音频（与 ``AudioChunk.to_ndarray()`` 一致）。
* 多声道内存布局统一为 **interleaved**（L R L R ...），与 WebAudio / WebRTC 一致。
* 所有函数纯函数无状态，只依赖 numpy。
"""
from __future__ import annotations

import numpy as np

from omnistream.core.clock import now_ns
from omnistream.media.errors import CodecError
from omnistream.protocol.enums import AudioEncoding
from omnistream.schemas.audio import AudioChunk

__all__ = [
    "INT16_MAX",
    "bytes_to_float32",
    "chunk_from_float32",
    "concat",
    "db_to_linear",
    "dbfs",
    "deinterleave",
    "float32_to_bytes",
    "interleave",
    "linear_to_db",
    "mix_to_mono",
    "mono_to_stereo",
    "peak",
    "remix",
    "rms",
    "slice_frames",
    "split_fixed",
    "to_float32",
    "trim_silence_edges",
    "zeros",
]

INT16_MAX: float = 32767.0
_INT16_SCALE: float = 32768.0
_EPS: float = 1e-12


def bytes_to_float32(data: bytes, encoding: AudioEncoding = AudioEncoding.PCM16) -> np.ndarray:
    """把裸 PCM 字节转成 ``float32`` 一维数组（interleaved 保持不变）。

    Args:
        data: 原始字节。
        encoding: 仅支持 ``pcm16`` / ``pcm_f32``。

    Returns:
        一维 ``float32`` 数组，范围 ``[-1, 1]``。

    Raises:
        CodecError: 传入压缩编码或字节长度不对齐。
    """
    if encoding == AudioEncoding.PCM16:
        if len(data) % 2:
            data = data[: len(data) - 1]
        return np.frombuffer(data, dtype="<i2").astype(np.float32) / _INT16_SCALE
    if encoding == AudioEncoding.PCM_F32:
        if len(data) % 4:
            data = data[: len(data) - (len(data) % 4)]
        return np.frombuffer(data, dtype="<f4").astype(np.float32, copy=True)
    raise CodecError(f"{encoding} must be decoded before converting to ndarray")


def float32_to_bytes(arr: np.ndarray, encoding: AudioEncoding = AudioEncoding.PCM16) -> bytes:
    """把 ``float32`` 数组转回裸 PCM 字节（含硬限幅）。"""
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    clipped = np.clip(flat, -1.0, 1.0)
    if encoding == AudioEncoding.PCM16:
        return (clipped * INT16_MAX).astype("<i2").tobytes()
    if encoding == AudioEncoding.PCM_F32:
        return clipped.astype("<f4").tobytes()
    raise CodecError(f"cannot serialise float32 samples to {encoding}")


def to_float32(chunk: AudioChunk) -> np.ndarray:
    """从 :class:`AudioChunk` 取出 interleaved ``float32`` 一维数组。"""
    return bytes_to_float32(chunk.data, chunk.encoding)


def chunk_from_float32(
    arr: np.ndarray,
    *,
    pts_ms: int,
    sample_rate: int,
    channels: int = 1,
    encoding: AudioEncoding = AudioEncoding.PCM16,
    capture_ts_ns: int | None = None,
    speaker: str | None = None,
) -> AudioChunk:
    """由 ``float32`` 数组构造 :class:`AudioChunk`。"""
    return AudioChunk(
        pts_ms=pts_ms,
        data=float32_to_bytes(arr, encoding),
        sample_rate=sample_rate,
        channels=channels,
        encoding=encoding,
        capture_ts_ns=capture_ts_ns if capture_ts_ns is not None else now_ns(),
        speaker=speaker,
    )


def deinterleave(arr: np.ndarray, channels: int) -> np.ndarray:
    """interleaved 一维数组 → ``(frames, channels)`` 二维数组。"""
    if channels <= 1:
        return arr.reshape(-1, 1)
    usable = (arr.size // channels) * channels
    return arr[:usable].reshape(-1, channels)


def interleave(arr2d: np.ndarray) -> np.ndarray:
    """``(frames, channels)`` → interleaved 一维数组。"""
    return np.asarray(arr2d, dtype=np.float32).reshape(-1)


def mix_to_mono(arr: np.ndarray, channels: int) -> np.ndarray:
    """多声道下混为单声道（等权平均，不做相位对齐）。"""
    if channels <= 1:
        return arr.reshape(-1)
    return deinterleave(arr, channels).mean(axis=1).astype(np.float32)


def mono_to_stereo(arr: np.ndarray) -> np.ndarray:
    """单声道复制为立体声（interleaved）。"""
    mono = arr.reshape(-1)
    return np.repeat(mono, 2).astype(np.float32)


def remix(arr: np.ndarray, src_channels: int, dst_channels: int) -> np.ndarray:
    """任意声道数转换（当前支持 1↔2，其余按取前 N / 复制第 0 路处理）。"""
    if src_channels == dst_channels:
        return arr.reshape(-1)
    if dst_channels == 1:
        return mix_to_mono(arr, src_channels)
    if src_channels == 1 and dst_channels == 2:
        return mono_to_stereo(arr)
    planar = deinterleave(arr, src_channels)
    if dst_channels < src_channels:
        return interleave(planar[:, :dst_channels])
    pad = np.repeat(planar[:, :1], dst_channels - src_channels, axis=1)
    return interleave(np.concatenate((planar, pad), axis=1))


def split_fixed(data: bytes, frame_bytes: int) -> tuple[list[bytes], bytes]:
    """把字节流切成定长块。

    Args:
        data: 输入字节。
        frame_bytes: 每块字节数，必须 > 0。

    Returns:
        ``(完整块列表, 尾部残留字节)``。
    """
    if frame_bytes <= 0:
        raise CodecError("frame_bytes must be positive", detail={"frame_bytes": frame_bytes})
    count = len(data) // frame_bytes
    frames = [data[i * frame_bytes : (i + 1) * frame_bytes] for i in range(count)]
    return frames, data[count * frame_bytes :]


def slice_frames(arr: np.ndarray, samples_per_frame: int) -> tuple[list[np.ndarray], np.ndarray]:
    """把样本数组切成定长帧，返回 ``(帧列表, 残留)``。"""
    if samples_per_frame <= 0:
        raise CodecError("samples_per_frame must be positive")
    count = arr.size // samples_per_frame
    frames = [arr[i * samples_per_frame : (i + 1) * samples_per_frame] for i in range(count)]
    return frames, arr[count * samples_per_frame :]


def concat(parts: list[np.ndarray]) -> np.ndarray:
    """拼接多段样本，空输入返回空数组。"""
    usable = [p for p in parts if p is not None and p.size]
    if not usable:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(usable).astype(np.float32, copy=False)


def zeros(num_samples: int, channels: int = 1) -> np.ndarray:
    """生成静音样本（interleaved）。"""
    return np.zeros(max(0, num_samples) * max(1, channels), dtype=np.float32)


def rms(arr: np.ndarray) -> float:
    """均方根电平（线性）。"""
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr, dtype=np.float64))))


def peak(arr: np.ndarray) -> float:
    """峰值绝对幅度（线性）。"""
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))


def dbfs(arr: np.ndarray) -> float:
    """RMS 电平转 dBFS，静音返回 ``-120.0``。"""
    value = rms(arr)
    if value <= _EPS:
        return -120.0
    return float(20.0 * np.log10(value))


def linear_to_db(value: float) -> float:
    """线性增益 → 分贝。"""
    return float(20.0 * np.log10(max(value, _EPS)))


def db_to_linear(db: float) -> float:
    """分贝 → 线性增益。"""
    return float(10.0 ** (db / 20.0))


def trim_silence_edges(arr: np.ndarray, threshold_dbfs: float = -55.0, window: int = 160) -> np.ndarray:
    """裁掉首尾静音（按窗口 RMS 判定），用于示例音频清洗。"""
    if arr.size == 0:
        return arr
    limit = db_to_linear(threshold_dbfs)
    num_windows = max(1, arr.size // window)
    trimmed = arr[: num_windows * window].reshape(num_windows, window)
    energy = np.sqrt(np.mean(np.square(trimmed, dtype=np.float64), axis=1))
    active = np.flatnonzero(energy > limit)
    if active.size == 0:
        return np.zeros(0, dtype=np.float32)
    start = int(active[0]) * window
    end = min(arr.size, (int(active[-1]) + 1) * window)
    return arr[start:end]

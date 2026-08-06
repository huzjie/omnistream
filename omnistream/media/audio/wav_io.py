"""WAV 读写工具。

只依赖标准库 :mod:`wave` 与 numpy —— **任何环境下都可用**，因此被示例脚本、
压测脚本、录音落盘路径共同复用。

约定
----
* 对外统一暴露 ``float32`` interleaved 样本（范围 ``[-1, 1]``），与
  :mod:`omnistream.media.audio.pcm` 一致。
* 写盘统一 16-bit PCM（sampwidth=2），兼容性最好。
* 读取时支持 8/16/24/32-bit PCM，自动归一化。
"""
from __future__ import annotations

import io
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from omnistream.media.audio import pcm
from omnistream.media.constants import DEFAULT_FRAME_MS, DEFAULT_INPUT_SAMPLE_RATE
from omnistream.media.errors import CodecError
from omnistream.protocol.enums import AudioEncoding
from omnistream.schemas.audio import AudioChunk

__all__ = [
    "WavInfo",
    "WavStreamWriter",
    "chunks_to_wav",
    "encode_wav_bytes",
    "iter_chunks",
    "probe_wav",
    "read_wav",
    "read_wav_bytes",
    "write_wav",
]


@dataclass(slots=True, frozen=True)
class WavInfo:
    """WAV 文件头信息。"""

    sample_rate: int
    channels: int
    sample_width: int
    num_frames: int

    @property
    def duration_ms(self) -> float:
        """总时长（毫秒）。"""
        if self.sample_rate <= 0:
            return 0.0
        return self.num_frames * 1000.0 / self.sample_rate

    def as_dict(self) -> dict[str, float]:
        """转为可序列化字典。"""
        return {
            "sample_rate": float(self.sample_rate),
            "channels": float(self.channels),
            "sample_width": float(self.sample_width),
            "num_frames": float(self.num_frames),
            "duration_ms": round(self.duration_ms, 3),
        }


def _decode_pcm(raw: bytes, sample_width: int, num_channels: int) -> np.ndarray:
    """把 ``wave`` 读出的裸帧转成 float32 interleaved 数组。"""
    if sample_width == 1:
        # 8-bit WAV 是无符号的，偏置 128
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (arr - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        # 24-bit：手工拼成 int32 再右移
        usable = (len(raw) // 3) * 3
        byte_view = np.frombuffer(raw[:usable], dtype=np.uint8).reshape(-1, 3)
        as_int = (
            byte_view[:, 0].astype(np.int32)
            | (byte_view[:, 1].astype(np.int32) << 8)
            | (byte_view[:, 2].astype(np.int32) << 16)
        )
        # 符号扩展
        as_int = np.where(as_int & 0x800000, as_int - 0x1000000, as_int)
        return as_int.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise CodecError(
        f"unsupported WAV sample width: {sample_width} bytes",
        detail={"sample_width": sample_width, "channels": num_channels},
    )


def _read_from(reader: wave.Wave_read) -> tuple[np.ndarray, WavInfo]:
    """从一个已打开的 ``Wave_read`` 读取全部样本。"""
    channels = reader.getnchannels()
    sample_width = reader.getsampwidth()
    sample_rate = reader.getframerate()
    num_frames = reader.getnframes()
    raw = reader.readframes(num_frames)
    samples = _decode_pcm(raw, sample_width, channels)
    info = WavInfo(
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        num_frames=num_frames,
    )
    return samples, info


def read_wav(path: str | Path) -> tuple[np.ndarray, WavInfo]:
    """读取 WAV 文件。

    Args:
        path: 文件路径。

    Returns:
        ``(float32 interleaved 样本, WavInfo)``。

    Raises:
        CodecError: 文件不是合法 WAV 或位深不受支持。
    """
    target = Path(path)
    try:
        with wave.open(str(target), "rb") as reader:
            return _read_from(reader)
    except wave.Error as exc:
        raise CodecError(f"cannot read WAV file: {target}", cause=exc) from exc
    except OSError as exc:
        raise CodecError(f"cannot open WAV file: {target}", cause=exc) from exc


def read_wav_bytes(data: bytes) -> tuple[np.ndarray, WavInfo]:
    """从内存字节读取 WAV。"""
    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            return _read_from(reader)
    except wave.Error as exc:
        raise CodecError("cannot parse WAV bytes", cause=exc) from exc


def probe_wav(path: str | Path) -> WavInfo:
    """只读文件头，不加载样本（用于快速校验素材）。"""
    target = Path(path)
    try:
        with wave.open(str(target), "rb") as reader:
            return WavInfo(
                sample_rate=reader.getframerate(),
                channels=reader.getnchannels(),
                sample_width=reader.getsampwidth(),
                num_frames=reader.getnframes(),
            )
    except (wave.Error, OSError) as exc:
        raise CodecError(f"cannot probe WAV file: {target}", cause=exc) from exc


def write_wav(
    path: str | Path,
    arr: np.ndarray,
    *,
    sample_rate: int = DEFAULT_INPUT_SAMPLE_RATE,
    channels: int = 1,
) -> WavInfo:
    """把 float32 样本写成 16-bit PCM WAV。

    Args:
        path: 目标路径，父目录会自动创建。
        arr: interleaved float32 样本。
        sample_rate: 采样率。
        channels: 声道数。

    Returns:
        写入结果的 :class:`WavInfo`。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = pcm.float32_to_bytes(arr, AudioEncoding.PCM16)
    with wave.open(str(target), "wb") as writer:
        writer.setnchannels(max(1, channels))
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(payload)
    frames = len(payload) // (2 * max(1, channels))
    return WavInfo(sample_rate=sample_rate, channels=channels, sample_width=2, num_frames=frames)


def encode_wav_bytes(
    arr: np.ndarray,
    *,
    sample_rate: int = DEFAULT_INPUT_SAMPLE_RATE,
    channels: int = 1,
) -> bytes:
    """把 float32 样本编码成完整 WAV 字节（含 RIFF 头），便于 HTTP 直接返回。"""
    buffer = io.BytesIO()
    payload = pcm.float32_to_bytes(arr, AudioEncoding.PCM16)
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(max(1, channels))
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(payload)
    return buffer.getvalue()


def iter_chunks(
    path: str | Path,
    *,
    frame_ms: int = DEFAULT_FRAME_MS,
    target_sample_rate: int | None = None,
    mono: bool = True,
    start_pts_ms: int = 0,
) -> Iterator[AudioChunk]:
    """按固定帧长把 WAV 文件切成 :class:`AudioChunk` 序列。

    这是把磁盘素材灌进管线的标准入口（示例、压测、回归素材都用它）。

    Args:
        path: WAV 路径。
        frame_ms: 每帧毫秒数。
        target_sample_rate: 目标采样率；``None`` 表示保持原采样率。
        mono: 是否下混为单声道。
        start_pts_ms: 首帧 pts。

    Yields:
        连续的 :class:`AudioChunk`（最后一帧不足会补零）。
    """
    samples, info = read_wav(path)
    channels = info.channels
    if mono and channels > 1:
        samples = pcm.mix_to_mono(samples, channels)
        channels = 1

    sample_rate = info.sample_rate
    if target_sample_rate is not None and target_sample_rate != sample_rate:
        from omnistream.media.audio.resampler import Resampler

        converter = Resampler(sample_rate, target_sample_rate, channels=channels)
        samples = converter.process_array(samples)
        tail = pcm.bytes_to_float32(converter.flush(), AudioEncoding.PCM16)
        if tail.size:
            samples = pcm.concat([samples, tail])
        sample_rate = target_sample_rate

    samples_per_frame = max(1, int(round(sample_rate * frame_ms / 1000.0))) * max(1, channels)
    total = samples.size
    index = 0
    emitted = 0
    while index < total:
        block = samples[index : index + samples_per_frame]
        if block.size < samples_per_frame:
            block = np.concatenate(
                (block, np.zeros(samples_per_frame - block.size, dtype=np.float32))
            )
        pts = start_pts_ms + int(round(emitted * frame_ms))
        yield pcm.chunk_from_float32(
            block, pts_ms=pts, sample_rate=sample_rate, channels=max(1, channels)
        )
        index += samples_per_frame
        emitted += 1


def chunks_to_wav(chunks: list[AudioChunk], path: str | Path) -> WavInfo:
    """把一串 :class:`AudioChunk` 顺序落盘为 WAV。

    Raises:
        CodecError: 传入空列表，或块之间采样率/声道不一致。
    """
    if not chunks:
        raise CodecError("cannot write WAV from an empty chunk list")
    sample_rate = chunks[0].sample_rate
    channels = chunks[0].channels
    parts: list[np.ndarray] = []
    for chunk in chunks:
        if chunk.sample_rate != sample_rate or chunk.channels != channels:
            raise CodecError(
                "inconsistent audio format across chunks",
                detail={
                    "expected": f"{sample_rate}Hz/{channels}ch",
                    "actual": f"{chunk.sample_rate}Hz/{chunk.channels}ch",
                    "pts_ms": chunk.pts_ms,
                },
            )
        parts.append(pcm.to_float32(chunk))
    return write_wav(path, pcm.concat(parts), sample_rate=sample_rate, channels=channels)


class WavStreamWriter:
    """增量 WAV 落盘器：边收边写，不需要把整段音频驻留内存。

    典型用法（录制会话）::

        with WavStreamWriter("rec.wav", sample_rate=16000) as writer:
            async for chunk in stream:
                writer.append_chunk(chunk)
    """

    def __init__(
        self,
        path: str | Path,
        *,
        sample_rate: int = DEFAULT_INPUT_SAMPLE_RATE,
        channels: int = 1,
    ) -> None:
        """打开目标文件并写入 WAV 头。"""
        self.path = Path(path)
        self.sample_rate = int(sample_rate)
        self.channels = max(1, int(channels))
        self.frames_written = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer: wave.Wave_write | None = wave.open(str(self.path), "wb")
        self._writer.setnchannels(self.channels)
        self._writer.setsampwidth(2)
        self._writer.setframerate(self.sample_rate)

    @property
    def closed(self) -> bool:
        """是否已关闭。"""
        return self._writer is None

    @property
    def duration_ms(self) -> float:
        """已写入时长（毫秒）。"""
        if self.sample_rate <= 0:
            return 0.0
        return self.frames_written * 1000.0 / self.sample_rate

    def append(self, arr: np.ndarray) -> int:
        """追加一段 float32 样本，返回本次写入的帧数。"""
        if self._writer is None:
            raise CodecError("WavStreamWriter is already closed", detail={"path": str(self.path)})
        payload = pcm.float32_to_bytes(arr, AudioEncoding.PCM16)
        self._writer.writeframes(payload)
        frames = len(payload) // (2 * self.channels)
        self.frames_written += frames
        return frames

    def append_chunk(self, chunk: AudioChunk) -> int:
        """追加一个 :class:`AudioChunk`（自动做声道下混，采样率必须匹配）。"""
        if chunk.sample_rate != self.sample_rate:
            raise CodecError(
                "sample rate mismatch in WavStreamWriter",
                detail={"expected": self.sample_rate, "actual": chunk.sample_rate},
            )
        samples = pcm.to_float32(chunk)
        if chunk.channels != self.channels:
            samples = pcm.remix(samples, chunk.channels, self.channels)
        return self.append(samples)

    def close(self) -> WavInfo:
        """关闭文件（幂等），返回最终 :class:`WavInfo`。"""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return WavInfo(
            sample_rate=self.sample_rate,
            channels=self.channels,
            sample_width=2,
            num_frames=self.frames_written,
        )

    def __enter__(self) -> "WavStreamWriter":
        """上下文管理器入口。"""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """上下文管理器出口，确保文件被关闭。"""
        self.close()

    def __repr__(self) -> str:
        """调试用文本表示。"""
        return (
            f"WavStreamWriter(path={self.path.name!r}, {self.sample_rate}Hz/"
            f"{self.channels}ch, {self.duration_ms:.0f}ms)"
        )

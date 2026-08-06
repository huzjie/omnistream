"""任意长度音频流 → 固定 20ms 帧切片器。

VAD / ASR / Opus 编码都要求定长帧；而网络与 TTS 送来的是任意长度的字节块。
本模块负责在两者之间做无损缓冲与重新分帧，并沿时间轴连续分配 ``pts_ms``。

保证：
* 输出帧长严格恒定（字节数 = sample_rate * frame_ms / 1000 * width * channels）
* ``pts_ms`` 连续无空洞（按样本数推进，不用墙钟，避免累积漂移）
* 不足一帧的尾部缓存到下次调用；``flush(pad=True)`` 时补零成整帧
"""
from __future__ import annotations

import structlog

from omnistream.core.clock import now_ns
from omnistream.media.constants import DEFAULT_CHANNELS, DEFAULT_FRAME_MS, DEFAULT_INPUT_SAMPLE_RATE
from omnistream.media.errors import CodecError
from omnistream.protocol.enums import AudioEncoding
from omnistream.schemas.audio import AudioChunk

log = structlog.get_logger(__name__)

__all__ = ["FrameSlicer"]


class FrameSlicer:
    """定长分帧器（非线程安全，每条流独占）。

    Example::

        slicer = FrameSlicer(sample_rate=16000, frame_ms=20)
        for frame in slicer.push_bytes(raw):
            await vad_stream.put(frame)
        for frame in slicer.flush(pad=True):
            await vad_stream.put(frame)
    """

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_INPUT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        frame_ms: int = DEFAULT_FRAME_MS,
        encoding: AudioEncoding = AudioEncoding.PCM16,
        start_pts_ms: int = 0,
    ) -> None:
        """
        Args:
            sample_rate: 采样率。
            channels: 声道数。
            frame_ms: 目标帧长（毫秒）。
            encoding: 仅支持 pcm16 / pcm_f32。
            start_pts_ms: 起始 pts。

        Raises:
            CodecError: 编码不是裸 PCM，或帧长/采样率导致帧字节数为 0。
        """
        if encoding not in (AudioEncoding.PCM16, AudioEncoding.PCM_F32):
            raise CodecError(f"frame slicer requires raw pcm, got {encoding}")
        self.sample_rate = sample_rate
        self.channels = max(1, channels)
        self.frame_ms = max(1, frame_ms)
        self.encoding = encoding
        self._width = 4 if encoding == AudioEncoding.PCM_F32 else 2
        self._samples_per_frame = int(sample_rate * self.frame_ms / 1000)
        self._frame_bytes = self._samples_per_frame * self._width * self.channels
        if self._frame_bytes <= 0:
            raise CodecError(
                "computed frame size is zero",
                detail={"sample_rate": sample_rate, "frame_ms": frame_ms},
            )
        self._buffer = bytearray()
        self._pts_ms = start_pts_ms
        self._samples_emitted = 0
        self._start_pts_ms = start_pts_ms
        self._last_capture_ts_ns = now_ns()
        self._speaker: str | None = None

    # ---------------- 属性 ----------------

    @property
    def frame_bytes(self) -> int:
        """单帧字节数。"""
        return self._frame_bytes

    @property
    def pending_bytes(self) -> int:
        """尚未成帧的残留字节数。"""
        return len(self._buffer)

    @property
    def pts_ms(self) -> int:
        """下一帧将使用的 pts。"""
        return self._pts_ms

    # ---------------- 写入 ----------------

    def push_bytes(self, data: bytes, *, capture_ts_ns: int | None = None) -> list[AudioChunk]:
        """追加裸字节，返回本次可以凑出的完整帧。"""
        if not data:
            return []
        self._last_capture_ts_ns = capture_ts_ns if capture_ts_ns is not None else now_ns()
        self._buffer.extend(data)
        return self._drain()

    def push(self, chunk: AudioChunk) -> list[AudioChunk]:
        """追加一个 :class:`AudioChunk`（会校验格式一致性）。

        Raises:
            CodecError: 采样率 / 声道 / 编码与切片器不一致。
        """
        if chunk.encoding != self.encoding:
            raise CodecError(
                f"encoding mismatch: slicer={self.encoding} chunk={chunk.encoding}"
            )
        if chunk.sample_rate != self.sample_rate:
            raise CodecError(
                f"sample rate mismatch: slicer={self.sample_rate} chunk={chunk.sample_rate}"
            )
        if chunk.channels != self.channels:
            raise CodecError(
                f"channel mismatch: slicer={self.channels} chunk={chunk.channels}"
            )
        self._speaker = chunk.speaker or self._speaker
        return self.push_bytes(chunk.data, capture_ts_ns=chunk.capture_ts_ns)

    def _drain(self) -> list[AudioChunk]:
        out: list[AudioChunk] = []
        while len(self._buffer) >= self._frame_bytes:
            raw = bytes(self._buffer[: self._frame_bytes])
            del self._buffer[: self._frame_bytes]
            out.append(self._make_chunk(raw))
        return out

    def _make_chunk(self, raw: bytes) -> AudioChunk:
        chunk = AudioChunk(
            pts_ms=self._pts_ms,
            data=raw,
            sample_rate=self.sample_rate,
            channels=self.channels,
            encoding=self.encoding,
            capture_ts_ns=self._last_capture_ts_ns,
            speaker=self._speaker,
        )
        self._samples_emitted += self._samples_per_frame
        # 按累计样本数推 pts，避免 frame_ms 非整除采样率时的累积误差
        self._pts_ms = self._start_pts_ms + int(
            self._samples_emitted * 1000 / self.sample_rate
        )
        return chunk

    # ---------------- 收尾 ----------------

    def flush(self, *, pad: bool = True) -> list[AudioChunk]:
        """输出残留数据。

        Args:
            pad: 为 True 时补零到整帧输出；为 False 时直接丢弃残留。

        Returns:
            至多一帧的列表。
        """
        if not self._buffer:
            return []
        if not pad:
            self._buffer.clear()
            return []
        raw = bytes(self._buffer) + b"\x00" * (self._frame_bytes - len(self._buffer))
        self._buffer.clear()
        return [self._make_chunk(raw)]

    def reset(self, *, start_pts_ms: int | None = None) -> None:
        """清空缓冲并可选地重置时间轴。"""
        self._buffer.clear()
        self._samples_emitted = 0
        if start_pts_ms is not None:
            self._start_pts_ms = start_pts_ms
        self._pts_ms = self._start_pts_ms

    def __repr__(self) -> str:
        return (
            f"<FrameSlicer {self.sample_rate}Hz ch={self.channels} "
            f"frame={self.frame_ms}ms pending={len(self._buffer)}B>"
        )

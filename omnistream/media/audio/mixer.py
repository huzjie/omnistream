"""多路音频混音（多说话人 / 背景音 / 提示音场景）。

特性
----
* 每路独立增益、静音开关、可选淡入淡出
* 按 ``pts_ms`` 对齐：迟到的轨道自动补静音，不会把整体时间轴拖歪
* 混音后统一软限幅，避免多路叠加削波
* 支持"独奏"（solo）与动态增删轨道
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from omnistream.media.audio import pcm
from omnistream.media.audio.gain import soft_limit
from omnistream.media.constants import DEFAULT_FRAME_MS, LIMITER_CEILING_DBFS
from omnistream.media.errors import CodecError
from omnistream.protocol.enums import AudioEncoding
from omnistream.schemas.audio import AudioChunk

__all__ = ["AudioMixer", "MixerTrack"]


@dataclass(slots=True)
class MixerTrack:
    """混音器中的一条轨道。

    Attributes:
        track_id: 轨道标识（通常是 speaker id）。
        gain: 线性增益。
        muted: 是否静音。
        buffer: 待混音样本队列。
        pending: 队列中的样本总数。
        submitted: 累计提交样本数。
        starved: 缺样本被补零的次数。
    """

    track_id: str
    gain: float = 1.0
    muted: bool = False
    buffer: deque[np.ndarray] = field(default_factory=deque)
    pending: int = 0
    submitted: int = 0
    starved: int = 0

    def push(self, samples: np.ndarray) -> None:
        """写入样本。"""
        if samples.size == 0:
            return
        self.buffer.append(samples.astype(np.float32, copy=False))
        self.pending += samples.size
        self.submitted += samples.size

    def take(self, count: int) -> np.ndarray:
        """取出 ``count`` 个样本，不足补零并计入 starved。"""
        if count <= 0:
            return np.zeros(0, dtype=np.float32)
        out = np.zeros(count, dtype=np.float32)
        filled = 0
        while filled < count and self.buffer:
            head = self.buffer[0]
            need = count - filled
            if head.size <= need:
                out[filled : filled + head.size] = head
                filled += head.size
                self.buffer.popleft()
                self.pending -= head.size
            else:
                out[filled:] = head[:need]
                self.buffer[0] = head[need:]
                self.pending -= need
                filled = count
        if filled < count:
            self.starved += 1
        return out

    def clear(self) -> None:
        """丢弃全部待混样本。"""
        self.buffer.clear()
        self.pending = 0


class AudioMixer:
    """N 路混音器。

    Example::

        mixer = AudioMixer(sample_rate=24000)
        mixer.add_track("agent")
        mixer.add_track("bgm", gain=0.2)
        mixer.submit("agent", agent_chunk)
        mixer.submit("bgm", bgm_chunk)
        out = mixer.mix_frame()          # 一帧 20ms 的 AudioChunk
    """

    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        channels: int = 1,
        frame_ms: int = DEFAULT_FRAME_MS,
        ceiling_dbfs: float = LIMITER_CEILING_DBFS,
        start_pts_ms: int = 0,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = max(1, channels)
        self.frame_ms = max(1, frame_ms)
        self.ceiling_dbfs = ceiling_dbfs
        self._tracks: dict[str, MixerTrack] = {}
        self._solo: str | None = None
        self._pts_ms = start_pts_ms
        self._start_pts_ms = start_pts_ms
        self._frames_emitted = 0

    # ---------------- 轨道管理 ----------------

    @property
    def track_ids(self) -> list[str]:
        """当前所有轨道 id。"""
        return list(self._tracks)

    @property
    def samples_per_frame(self) -> int:
        """单帧样本数（含所有声道）。"""
        return int(self.sample_rate * self.frame_ms / 1000) * self.channels

    def add_track(self, track_id: str, *, gain: float = 1.0, muted: bool = False) -> MixerTrack:
        """新增或返回已有轨道。"""
        track = self._tracks.get(track_id)
        if track is None:
            track = MixerTrack(track_id=track_id, gain=gain, muted=muted)
            self._tracks[track_id] = track
        else:
            track.gain = gain
            track.muted = muted
        return track

    def remove_track(self, track_id: str) -> None:
        """删除轨道。"""
        self._tracks.pop(track_id, None)
        if self._solo == track_id:
            self._solo = None

    def set_gain(self, track_id: str, gain: float) -> None:
        """设置轨道增益。"""
        track = self._tracks.get(track_id)
        if track is not None:
            track.gain = float(max(0.0, gain))

    def set_muted(self, track_id: str, muted: bool) -> None:
        """静音/取消静音。"""
        track = self._tracks.get(track_id)
        if track is not None:
            track.muted = muted

    def solo(self, track_id: str | None) -> None:
        """独奏某轨（``None`` 表示取消独奏）。"""
        self._solo = track_id

    # ---------------- 数据写入 ----------------

    def submit(self, track_id: str, chunk: AudioChunk) -> None:
        """向轨道写入一个 chunk。

        Raises:
            CodecError: 采样率不匹配或编码不是裸 PCM。
        """
        if chunk.sample_rate != self.sample_rate:
            raise CodecError(
                f"mixer sample rate mismatch: mixer={self.sample_rate} chunk={chunk.sample_rate}"
            )
        if chunk.encoding not in (AudioEncoding.PCM16, AudioEncoding.PCM_F32):
            raise CodecError(f"mixer requires raw pcm, got {chunk.encoding}")
        track = self._tracks.get(track_id) or self.add_track(track_id)
        samples = pcm.bytes_to_float32(chunk.data, chunk.encoding)
        if chunk.channels != self.channels:
            samples = pcm.remix(samples, chunk.channels, self.channels)
        track.push(samples)

    def submit_samples(self, track_id: str, samples: np.ndarray) -> None:
        """直接写入 ``float32`` 样本。"""
        track = self._tracks.get(track_id) or self.add_track(track_id)
        track.push(samples)

    # ---------------- 混音 ----------------

    @property
    def ready_samples(self) -> int:
        """所有非静音轨道中最短的可用样本数。"""
        active = [t for t in self._tracks.values() if not t.muted]
        if not active:
            return 0
        return min(t.pending for t in active)

    def mix_frame(self, *, allow_partial: bool = True) -> AudioChunk | None:
        """混出一帧。

        Args:
            allow_partial: 为 True 时即使部分轨道缺数据也照常出帧（补零）；
                为 False 时所有轨道都必须有足够样本才出帧。

        Returns:
            混音后的 :class:`AudioChunk`；无轨道或数据不足时返回 ``None``。
        """
        if not self._tracks:
            return None
        n = self.samples_per_frame
        if not allow_partial and self.ready_samples < n:
            return None
        if allow_partial and self.ready_samples <= 0:
            has_any = any(t.pending > 0 for t in self._tracks.values())
            if not has_any:
                return None

        acc = np.zeros(n, dtype=np.float32)
        for track in self._tracks.values():
            if track.muted:
                track.take(n)  # 静音轨仍要消费，保持时间轴同步
                continue
            if self._solo is not None and track.track_id != self._solo:
                track.take(n)
                continue
            acc += track.take(n) * track.gain

        mixed = soft_limit(acc, self.ceiling_dbfs)
        chunk = pcm.chunk_from_float32(
            mixed,
            pts_ms=self._pts_ms,
            sample_rate=self.sample_rate,
            channels=self.channels,
            encoding=AudioEncoding.PCM16,
        )
        self._frames_emitted += 1
        self._pts_ms = self._start_pts_ms + int(
            self._frames_emitted * n / self.channels * 1000 / self.sample_rate
        )
        return chunk

    def drain(self) -> list[AudioChunk]:
        """混出全部剩余数据（会话结束时调用）。"""
        out: list[AudioChunk] = []
        while any(t.pending > 0 for t in self._tracks.values()):
            frame = self.mix_frame(allow_partial=True)
            if frame is None:
                break
            out.append(frame)
        return out

    def clear(self) -> None:
        """清空所有轨道缓冲。"""
        for track in self._tracks.values():
            track.clear()

    def snapshot(self) -> dict[str, object]:
        """统计快照。"""
        return {
            "tracks": len(self._tracks),
            "solo": self._solo,
            "pts_ms": self._pts_ms,
            "per_track": {
                tid: {
                    "pending": t.pending,
                    "submitted": t.submitted,
                    "starved": t.starved,
                    "gain": round(t.gain, 3),
                    "muted": t.muted,
                }
                for tid, t in self._tracks.items()
            },
        }

    def __repr__(self) -> str:
        return f"<AudioMixer tracks={len(self._tracks)} pts={self._pts_ms}ms>"

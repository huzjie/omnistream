"""自适应抖动缓冲：目标时延动态调整 + 丢包补偿（PLC）+ 乱序重排。

算法
----
**抖动估计**（RFC 3550 §6.4.1）::

    D(i) = (R(i) - R(i-1)) - (S(i) - S(i-1))
    J    = J + (|D(i)| - J) / 16

其中 ``R`` 为本地到达时刻（单调纳秒转毫秒），``S`` 为帧的 ``pts_ms``。

**目标时延**::

    target = clamp(frame_ms + JITTER_SAFETY_FACTOR * J, min_delay, max_delay)

扩张立即生效（防卡顿），收缩需要连续 ``JITTER_SHRINK_PATIENCE`` 帧无扩张诉求
才允许下调一格（滞回，防止目标值抖动导致周期性爆音）。

**播放决策**：缓冲深度达到 target 后开始出帧；出帧时按 ``pts_ms`` 排序：

* pts 恰好等于期望值 → 正常输出
* pts 大于期望值 → 缺帧，输出 PLC 帧（衰减重复上一帧），期望值前进
* pts 小于期望值 → 迟到帧，丢弃并计数

**PLC**：连续补偿不超过 ``JITTER_MAX_PLC_FRAMES`` 帧，采用"上一帧衰减重复 +
边界交叉淡入淡出"，避免明显的金属声；超出上限后输出静音。
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
import structlog

from omnistream.core.clock import now_ns
from omnistream.media import metrics
from omnistream.media.audio import pcm
from omnistream.media.constants import (
    DEFAULT_FRAME_MS,
    JITTER_FILTER_GAIN,
    JITTER_INITIAL_TARGET_MS,
    JITTER_MAX_DELAY_MS,
    JITTER_MAX_PLC_FRAMES,
    JITTER_MIN_DELAY_MS,
    JITTER_SAFETY_FACTOR,
    JITTER_SHRINK_PATIENCE,
)
from omnistream.protocol.enums import AudioEncoding
from omnistream.schemas.audio import AudioChunk

log = structlog.get_logger(__name__)

__all__ = ["AdaptiveJitterBuffer", "JitterStats"]


@dataclass(slots=True)
class JitterStats:
    """抖动缓冲运行统计，供 ``/v1/media/{sid}/stats`` 与 Prometheus 消费。"""

    received: int = 0
    played: int = 0
    late: int = 0
    duplicated: int = 0
    reordered: int = 0
    concealed: int = 0
    silence_inserted: int = 0
    overflow_dropped: int = 0
    jitter_ms: float = 0.0
    target_ms: float = float(JITTER_INITIAL_TARGET_MS)
    depth_ms: float = 0.0
    max_depth_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        """转成可 JSON 序列化的扁平字典。"""
        return {
            "received": float(self.received),
            "played": float(self.played),
            "late": float(self.late),
            "duplicated": float(self.duplicated),
            "reordered": float(self.reordered),
            "concealed": float(self.concealed),
            "silence_inserted": float(self.silence_inserted),
            "overflow_dropped": float(self.overflow_dropped),
            "jitter_ms": round(self.jitter_ms, 3),
            "target_ms": round(self.target_ms, 1),
            "depth_ms": round(self.depth_ms, 1),
            "max_depth_ms": round(self.max_depth_ms, 1),
            "loss_ratio": round(
                self.concealed / max(1, self.played + self.concealed), 5
            ),
        }


@dataclass(order=True)
class _Entry:
    """堆元素：按 (pts_ms, 到达序号) 排序，保证同 pts 时先到先出。"""

    pts_ms: int
    arrival_seq: int
    chunk: AudioChunk = field(compare=False)


class AdaptiveJitterBuffer:
    """自适应抖动缓冲。

    典型用法（由 ingress 管线以 20ms 节拍驱动）::

        jb = AdaptiveJitterBuffer(sample_rate=16000, frame_ms=20)
        jb.push(chunk)
        frame = jb.pop()          # None 表示尚未攒够目标时延
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        frame_ms: int = DEFAULT_FRAME_MS,
        min_delay_ms: int = JITTER_MIN_DELAY_MS,
        max_delay_ms: int = JITTER_MAX_DELAY_MS,
        initial_target_ms: int = JITTER_INITIAL_TARGET_MS,
        max_frames: int = 128,
        name: str = "jitter",
    ) -> None:
        """
        Args:
            sample_rate: 期望采样率（用于生成 PLC / 静音帧）。
            channels: 声道数。
            frame_ms: 标准帧长。
            min_delay_ms: 目标时延下限。
            max_delay_ms: 目标时延上限。
            initial_target_ms: 初始目标时延。
            max_frames: 堆内最多缓存帧数，超出丢最旧。
            name: 日志标识。
        """
        self.sample_rate = sample_rate
        self.channels = max(1, channels)
        self.frame_ms = max(1, frame_ms)
        self.min_delay_ms = min_delay_ms
        self.max_delay_ms = max_delay_ms
        self.max_frames = max_frames
        self.name = name

        self._heap: list[_Entry] = []
        self._arrival_seq = 0
        self._target_ms = float(initial_target_ms)
        self._jitter_ms = 0.0
        self._primed = False
        self._expected_pts: int | None = None
        self._last_arrival_ms: float | None = None
        self._last_pts_ms: int | None = None
        self._last_good: np.ndarray | None = None
        self._plc_run = 0
        self._shrink_counter = 0
        self._seen_pts: set[int] = set()
        self.stats = JitterStats(target_ms=self._target_ms)

    # ---------------- 属性 ----------------

    @property
    def target_ms(self) -> float:
        """当前目标时延（毫秒）。"""
        return self._target_ms

    @property
    def depth_ms(self) -> float:
        """当前缓冲深度（毫秒）。"""
        return len(self._heap) * self.frame_ms

    @property
    def size(self) -> int:
        """缓冲中的帧数。"""
        return len(self._heap)

    # ---------------- 写入 ----------------

    def push(self, chunk: AudioChunk, *, arrival_ns: int | None = None) -> bool:
        """写入一帧。

        Args:
            chunk: PCM16 音频帧。
            arrival_ns: 到达时刻（单调纳秒），缺省用 ``now_ns()``。

        Returns:
            True 表示已入缓冲；False 表示被丢弃（迟到 / 重复 / 溢出）。
        """
        self.stats.received += 1
        arrival_ms = (arrival_ns if arrival_ns is not None else now_ns()) / 1e6
        self._update_jitter(chunk.pts_ms, arrival_ms)

        if chunk.pts_ms in self._seen_pts:
            self.stats.duplicated += 1
            return False
        if self._expected_pts is not None and chunk.pts_ms < self._expected_pts:
            self.stats.late += 1
            metrics.frames_dropped.labels(media="audio", stage="jitter", reason="late").inc()
            return False
        if self._heap and chunk.pts_ms < self._heap[-1].pts_ms:
            self.stats.reordered += 1

        if len(self._heap) >= self.max_frames:
            evicted = heapq.heappop(self._heap)
            self._seen_pts.discard(evicted.pts_ms)
            self.stats.overflow_dropped += 1
            metrics.frames_dropped.labels(media="audio", stage="jitter", reason="overflow").inc()

        self._arrival_seq += 1
        heapq.heappush(self._heap, _Entry(chunk.pts_ms, self._arrival_seq, chunk))
        self._seen_pts.add(chunk.pts_ms)

        depth = self.depth_ms
        self.stats.depth_ms = depth
        self.stats.max_depth_ms = max(self.stats.max_depth_ms, depth)
        metrics.observe_ms(metrics.jitter_buffer_depth_ms, depth)
        return True

    def _update_jitter(self, pts_ms: int, arrival_ms: float) -> None:
        """RFC 3550 一阶抖动估计 + 目标时延自适应。"""
        if self._last_arrival_ms is not None and self._last_pts_ms is not None:
            transit_delta = (arrival_ms - self._last_arrival_ms) - (pts_ms - self._last_pts_ms)
            self._jitter_ms += (abs(transit_delta) - self._jitter_ms) / JITTER_FILTER_GAIN
        self._last_arrival_ms = arrival_ms
        self._last_pts_ms = pts_ms
        self.stats.jitter_ms = self._jitter_ms

        desired = self.frame_ms + JITTER_SAFETY_FACTOR * self._jitter_ms
        desired = max(self.min_delay_ms, min(self.max_delay_ms, desired))
        if desired > self._target_ms:
            self._target_ms = desired          # 扩张立即生效
            self._shrink_counter = 0
        else:
            self._shrink_counter += 1
            if self._shrink_counter >= JITTER_SHRINK_PATIENCE:
                self._shrink_counter = 0
                # 每次最多收缩一个帧长，平滑下调
                self._target_ms = max(desired, self._target_ms - self.frame_ms)
        self.stats.target_ms = self._target_ms
        metrics.set_gauge(metrics.jitter_buffer_target_ms, self._target_ms)

    # ---------------- 读取 ----------------

    def pop(self) -> AudioChunk | None:
        """取出下一帧。

        Returns:
            * 正常帧 / PLC 帧 / 静音帧
            * ``None`` 表示尚未攒够目标时延（调用方应等待下一个节拍）
        """
        if not self._primed:
            if self.depth_ms < self._target_ms:
                return None
            self._primed = True
            if self._heap:
                self._expected_pts = self._heap[0].pts_ms

        if not self._heap:
            # 缓冲空：链路中断或严重丢包，输出补偿帧并重新蓄水
            if self._expected_pts is None:
                return None
            frame = self._conceal(self._expected_pts)
            self._expected_pts += self.frame_ms
            if self._plc_run > JITTER_MAX_PLC_FRAMES * 2:
                self._primed = False
                self._plc_run = 0
            return frame

        head = self._heap[0]
        expected = self._expected_pts if self._expected_pts is not None else head.pts_ms

        if head.pts_ms <= expected + self.frame_ms // 2:
            entry = heapq.heappop(self._heap)
            self._seen_pts.discard(entry.pts_ms)
            self._plc_run = 0
            self.stats.played += 1
            self._expected_pts = entry.pts_ms + self.frame_ms
            self._remember(entry.chunk)
            self.stats.depth_ms = self.depth_ms
            return entry.chunk

        # 存在空洞：生成补偿帧顶上，等待迟到帧或彻底放弃
        frame = self._conceal(expected)
        self._expected_pts = expected + self.frame_ms
        return frame

    def pop_all_ready(self) -> list[AudioChunk]:
        """一次性取出所有已就绪的帧（批量驱动模式）。"""
        out: list[AudioChunk] = []
        while True:
            frame = self.pop()
            if frame is None:
                return out
            out.append(frame)
            if self.depth_ms <= self._target_ms:
                return out

    # ---------------- 丢包补偿 ----------------

    def _remember(self, chunk: AudioChunk) -> None:
        """记录最近一帧的样本，供 PLC 使用。"""
        if chunk.encoding != AudioEncoding.PCM16:
            return
        try:
            self._last_good = pcm.bytes_to_float32(chunk.data, AudioEncoding.PCM16)
        except Exception:
            self._last_good = None

    def _samples_per_frame(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000) * self.channels

    def _conceal(self, pts_ms: int) -> AudioChunk:
        """生成一帧补偿音频。

        前 ``JITTER_MAX_PLC_FRAMES`` 帧用"上一帧衰减重复 + 淡入淡出"，
        之后输出静音，避免长时间丢包产生刺耳的循环嗡鸣。
        """
        n = self._samples_per_frame()
        self._plc_run += 1
        self.stats.concealed += 1

        if self._last_good is None or self._last_good.size == 0 or self._plc_run > JITTER_MAX_PLC_FRAMES:
            self.stats.silence_inserted += 1
            metrics.jitter_buffer_plc.labels(mode="silence").inc()
            samples = np.zeros(n, dtype=np.float32)
        else:
            source = self._last_good
            if source.size >= n:
                base = source[-n:].copy()
            else:
                reps = int(np.ceil(n / max(1, source.size)))
                base = np.tile(source, reps)[:n].copy()
            decay = float(0.6 ** self._plc_run)
            fade_len = min(n // 4, max(1, n // 8))
            window = np.ones(n, dtype=np.float32)
            window[:fade_len] = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            window[-fade_len:] = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            samples = (base * decay * window).astype(np.float32)
            metrics.jitter_buffer_plc.labels(mode="repeat").inc()

        return pcm.chunk_from_float32(
            samples,
            pts_ms=pts_ms,
            sample_rate=self.sample_rate,
            channels=self.channels,
            encoding=AudioEncoding.PCM16,
        )

    # ---------------- 控制 ----------------

    def reset(self) -> None:
        """清空缓冲并回到未蓄水状态（重协商 / 会话重启）。"""
        self._heap.clear()
        self._seen_pts.clear()
        self._primed = False
        self._expected_pts = None
        self._last_arrival_ms = None
        self._last_pts_ms = None
        self._last_good = None
        self._plc_run = 0
        log.debug("jitter_buffer_reset", name=self.name)

    def snapshot(self) -> dict[str, float]:
        """返回统计快照。"""
        self.stats.depth_ms = self.depth_ms
        return self.stats.as_dict()

    def __repr__(self) -> str:
        return (
            f"<AdaptiveJitterBuffer {self.name} depth={self.depth_ms:.0f}ms "
            f"target={self._target_ms:.0f}ms jitter={self._jitter_ms:.1f}ms>"
        )

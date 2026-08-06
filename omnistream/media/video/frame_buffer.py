"""有界视频帧缓冲。

与 :class:`~omnistream.core.streams.BoundedStream` 的分工
--------------------------------------------------------
``BoundedStream`` 是**跨协程的传输通道**（FIFO，只能顺序消费）；
:class:`VideoFrameBuffer` 是**可回溯的窗口缓存**：关键帧选择器需要"取最近 N 帧"、
"取覆盖某个时间跨度的帧"，这类随机访问 FIFO 队列做不到。

策略固定为 DROP_OLDEST —— 视频帧过期就没有价值，宁可丢旧的也不阻塞采集。
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

from omnistream.media.constants import VIDEO_FRAME_BUFFER_MAXLEN
from omnistream.media.metrics import frames_dropped
from omnistream.media.video.frame import RawVideoFrame

__all__ = ["FrameBufferStats", "VideoFrameBuffer"]


@dataclass(slots=True)
class FrameBufferStats:
    """帧缓冲统计。"""

    pushed: int = 0
    dropped: int = 0
    taken: int = 0
    peak_size: int = 0

    def as_dict(self) -> dict[str, float]:
        """转为可序列化字典。"""
        return {
            "pushed": float(self.pushed),
            "dropped": float(self.dropped),
            "taken": float(self.taken),
            "peak_size": float(self.peak_size),
        }


class VideoFrameBuffer:
    """定长视频帧环形缓冲，支持按时间窗口与均匀间隔取样。"""

    def __init__(self, maxlen: int = VIDEO_FRAME_BUFFER_MAXLEN) -> None:
        """初始化缓冲。

        Args:
            maxlen: 最多保留多少帧，溢出丢最旧。
        """
        self.maxlen = max(1, int(maxlen))
        self._frames: deque[RawVideoFrame] = deque(maxlen=self.maxlen)
        self.stats = FrameBufferStats()

    # ------------------------------------------------------------------
    # 基本容器行为
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """当前帧数。"""
        return len(self._frames)

    def __iter__(self) -> Iterator[RawVideoFrame]:
        """按时间升序迭代。"""
        return iter(tuple(self._frames))

    def __bool__(self) -> bool:
        """缓冲非空则为真。"""
        return bool(self._frames)

    @property
    def size(self) -> int:
        """当前帧数。"""
        return len(self._frames)

    @property
    def span_ms(self) -> int:
        """缓冲覆盖的时间跨度（毫秒）。"""
        if len(self._frames) < 2:
            return 0
        return int(self._frames[-1].pts_ms - self._frames[0].pts_ms)

    @property
    def newest(self) -> RawVideoFrame | None:
        """最新一帧。"""
        return self._frames[-1] if self._frames else None

    @property
    def oldest(self) -> RawVideoFrame | None:
        """最旧一帧。"""
        return self._frames[0] if self._frames else None

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def push(self, frame: RawVideoFrame) -> bool:
        """压入一帧。

        Returns:
            ``True`` 表示未触发淘汰，``False`` 表示挤掉了最旧的一帧。
        """
        evicted = len(self._frames) >= self.maxlen
        self._frames.append(frame)
        self.stats.pushed += 1
        self.stats.peak_size = max(self.stats.peak_size, len(self._frames))
        if evicted:
            self.stats.dropped += 1
            try:
                frames_dropped.labels(media="video", stage="buffer", reason="overflow").inc()
            except Exception:  # 指标绝不能影响主链路
                pass
        return not evicted

    def extend(self, frames: list[RawVideoFrame]) -> int:
        """批量压入，返回被淘汰的帧数。"""
        dropped = 0
        for frame in frames:
            if not self.push(frame):
                dropped += 1
        return dropped

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def latest(self, count: int = 1) -> list[RawVideoFrame]:
        """取最近 ``count`` 帧（时间升序）。"""
        if count <= 0 or not self._frames:
            return []
        items = tuple(self._frames)
        self.stats.taken += min(count, len(items))
        return list(items[-count:])

    def since(self, pts_ms: int) -> list[RawVideoFrame]:
        """取 pts 大于等于给定值的所有帧。"""
        result = [frame for frame in self._frames if frame.pts_ms >= pts_ms]
        self.stats.taken += len(result)
        return result

    def window(self, duration_ms: int) -> list[RawVideoFrame]:
        """取最近 ``duration_ms`` 毫秒内的帧。"""
        if not self._frames:
            return []
        cutoff = self._frames[-1].pts_ms - max(0, int(duration_ms))
        return self.since(cutoff)

    def sample_uniform(self, count: int) -> list[RawVideoFrame]:
        """在缓冲内按时间均匀取 ``count`` 帧（含首尾）。

        关键帧选择器用它来"覆盖时间跨度"，避免连续取到几乎相同的相邻帧。
        """
        items = tuple(self._frames)
        if count <= 0 or not items:
            return []
        if count >= len(items):
            self.stats.taken += len(items)
            return list(items)
        if count == 1:
            self.stats.taken += 1
            return [items[-1]]
        step = (len(items) - 1) / float(count - 1)
        picked = [items[int(round(index * step))] for index in range(count)]
        self.stats.taken += len(picked)
        return picked

    def top_by_score(self, count: int) -> list[RawVideoFrame]:
        """按 ``keyframe_score`` 降序取 ``count`` 帧，结果按时间升序返回。"""
        items = tuple(self._frames)
        if count <= 0 or not items:
            return []
        ranked = sorted(items, key=lambda frame: frame.keyframe_score, reverse=True)[:count]
        self.stats.taken += len(ranked)
        return sorted(ranked, key=lambda frame: frame.pts_ms)

    def nearest(self, pts_ms: int) -> RawVideoFrame | None:
        """取 pts 最接近给定值的帧。"""
        if not self._frames:
            return None
        return min(self._frames, key=lambda frame: abs(frame.pts_ms - pts_ms))

    # ------------------------------------------------------------------
    # 维护
    # ------------------------------------------------------------------

    def prune_before(self, pts_ms: int) -> int:
        """丢弃 pts 小于给定值的帧，返回丢弃数量。"""
        removed = 0
        while self._frames and self._frames[0].pts_ms < pts_ms:
            self._frames.popleft()
            removed += 1
        return removed

    def resize(self, maxlen: int) -> None:
        """调整容量（自适应策略降档时缩小内存占用）。"""
        new_len = max(1, int(maxlen))
        if new_len == self.maxlen:
            return
        self.maxlen = new_len
        self._frames = deque(tuple(self._frames)[-new_len:], maxlen=new_len)

    def clear(self) -> int:
        """清空缓冲，返回被清除的帧数。"""
        count = len(self._frames)
        self._frames.clear()
        return count

    def snapshot(self) -> dict[str, float]:
        """导出统计快照。"""
        data = self.stats.as_dict()
        data["size"] = float(len(self._frames))
        data["maxlen"] = float(self.maxlen)
        data["span_ms"] = float(self.span_ms)
        return data

    def __repr__(self) -> str:
        """调试用文本表示。"""
        return (
            f"VideoFrameBuffer({len(self._frames)}/{self.maxlen}, "
            f"span={self.span_ms}ms, dropped={self.stats.dropped})"
        )

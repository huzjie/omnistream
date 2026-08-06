"""可撤销播放队列 —— barge-in 的核心执行体。

契约（§5.4 冻结，WS4 打断执行器依赖）::

    class PlaybackQueue:
        async def enqueue(self, chunk: AudioChunk) -> None: ...
        def cancel_from(self, pts_ms: int | None = None) -> int:
            \"\"\"丢弃 pts >= pts_ms 的所有未播帧，返回丢弃毫秒数。None 表示全清。
            必须是同步方法且 O(n) 无 await —— barge-in 路径不允许挂起。\"\"\"
        @property
        def played_ms(self) -> int: ...

实现说明
--------
这里**刻意没有**使用 ``core.streams.BoundedStream``（§7.7 的默认要求）：
``cancel_from(pts)`` 需要**按 pts 选择性清除**队列中间及尾部的元素，且必须是同步、
无 await 的 O(n) 操作，``asyncio.Queue`` 语义无法满足。因此改用
``collections.deque`` + ``asyncio.Event`` 手工实现背压与唤醒，
背压语义（满时阻塞生产者）与 ``OverflowPolicy.BLOCK`` 保持一致。

时间轴语义：
* ``pts_ms`` 是**助手本轮语音**的相对时间戳，从 0 开始，由 TTS/Provider 分配。
* ``played_ms`` 统计的是已经**交付给传输层**的音频毫秒数，是 barge-in 时
  ``playback.truncated{played_ms}`` 事件的数据来源。
"""
from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass

import structlog

from omnistream.media import metrics
from omnistream.media.constants import PLAYBACK_QUEUE_MAX_MS
from omnistream.schemas.audio import AudioChunk

log = structlog.get_logger(__name__)

__all__ = ["PlaybackQueue", "PlaybackStats"]


@dataclass(slots=True)
class PlaybackStats:
    """播放队列统计。"""

    enqueued_frames: int = 0
    enqueued_ms: float = 0.0
    played_frames: int = 0
    played_ms: float = 0.0
    cancelled_frames: int = 0
    cancelled_ms: float = 0.0
    cancellations: int = 0
    overflow_waits: int = 0
    max_depth_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        """转成扁平字典。"""
        return {
            "enqueued_frames": float(self.enqueued_frames),
            "enqueued_ms": round(self.enqueued_ms, 1),
            "played_frames": float(self.played_frames),
            "played_ms": round(self.played_ms, 1),
            "cancelled_frames": float(self.cancelled_frames),
            "cancelled_ms": round(self.cancelled_ms, 1),
            "cancellations": float(self.cancellations),
            "overflow_waits": float(self.overflow_waits),
            "max_depth_ms": round(self.max_depth_ms, 1),
        }


class PlaybackQueue:
    """可撤销的 TTS 播放队列。

    Example::

        pq = PlaybackQueue(session_id="sess_x")
        await pq.enqueue(chunk)                 # provider 侧生产
        async for chunk in pq.stream():         # egress 侧消费
            await transport.send_audio(chunk)

        dropped_ms = pq.cancel_from(current_pts)   # barge-in：同步、立即
    """

    def __init__(
        self,
        *,
        session_id: str = "",
        max_buffered_ms: int = PLAYBACK_QUEUE_MAX_MS,
        name: str = "playback",
    ) -> None:
        """
        Args:
            session_id: 所属会话，仅用于日志与指标。
            max_buffered_ms: 队列最大堆积毫秒，超过后 ``enqueue`` 阻塞（背压到 provider）。
            name: 日志标识。
        """
        self.session_id = session_id
        self.name = name
        self.max_buffered_ms = max_buffered_ms
        self._queue: deque[AudioChunk] = deque()
        self._buffered_ms = 0.0
        self._played_ms = 0.0
        self._closed = False
        self._epoch = 0
        self._data_ready = asyncio.Event()
        self._space_ready = asyncio.Event()
        self._space_ready.set()
        self.stats = PlaybackStats()

    # ---------------- 属性 ----------------

    @property
    def played_ms(self) -> int:
        """已交付给传输层的音频毫秒数（barge-in 事件用）。"""
        return int(self._played_ms)

    @property
    def pending_ms(self) -> int:
        """队列中尚未播放的毫秒数。"""
        return int(self._buffered_ms)

    @property
    def size(self) -> int:
        """队列中的帧数。"""
        return len(self._queue)

    @property
    def closed(self) -> bool:
        """队列是否已关闭。"""
        return self._closed

    @property
    def epoch(self) -> int:
        """撤销代次。每次 ``cancel_from`` 自增，消费者可据此丢弃过期帧。"""
        return self._epoch

    @property
    def head_pts_ms(self) -> int | None:
        """队首帧的 pts，空队列返回 ``None``。"""
        return self._queue[0].pts_ms if self._queue else None

    @property
    def tail_pts_ms(self) -> int | None:
        """队尾帧的 pts，空队列返回 ``None``。"""
        return self._queue[-1].pts_ms if self._queue else None

    # ---------------- 生产 ----------------

    async def enqueue(self, chunk: AudioChunk) -> None:
        """入队一帧 TTS 音频。

        队列堆积超过 ``max_buffered_ms`` 时挂起，形成对 provider 的背压。

        Args:
            chunk: 助手音频帧。

        Raises:
            RuntimeError: 队列已关闭。
        """
        if self._closed:
            raise RuntimeError(f"playback queue {self.name} is closed")
        while self._buffered_ms >= self.max_buffered_ms and not self._closed:
            self.stats.overflow_waits += 1
            self._space_ready.clear()
            await self._space_ready.wait()
        if self._closed:
            raise RuntimeError(f"playback queue {self.name} is closed")
        self._queue.append(chunk)
        duration = chunk.duration_ms
        self._buffered_ms += duration
        self.stats.enqueued_frames += 1
        self.stats.enqueued_ms += duration
        self.stats.max_depth_ms = max(self.stats.max_depth_ms, self._buffered_ms)
        metrics.observe_ms(metrics.playback_queue_depth_ms, self._buffered_ms)
        self._data_ready.set()

    def try_enqueue(self, chunk: AudioChunk) -> bool:
        """非阻塞入队，队列满时返回 ``False``（用于不能挂起的调用点）。"""
        if self._closed or self._buffered_ms >= self.max_buffered_ms:
            return False
        self._queue.append(chunk)
        duration = chunk.duration_ms
        self._buffered_ms += duration
        self.stats.enqueued_frames += 1
        self.stats.enqueued_ms += duration
        self._data_ready.set()
        return True

    # ---------------- 消费 ----------------

    async def dequeue(self) -> AudioChunk | None:
        """取出下一帧，队列空则等待。

        Returns:
            音频帧；队列关闭且已排空时返回 ``None``。
        """
        while True:
            if self._queue:
                chunk = self._queue.popleft()
                self._buffered_ms = max(0.0, self._buffered_ms - chunk.duration_ms)
                self._played_ms += chunk.duration_ms
                self.stats.played_frames += 1
                self.stats.played_ms = self._played_ms
                if self._buffered_ms < self.max_buffered_ms:
                    self._space_ready.set()
                if not self._queue:
                    self._data_ready.clear()
                return chunk
            if self._closed:
                return None
            self._data_ready.clear()
            await self._data_ready.wait()

    async def stream(self) -> AsyncIterator[AudioChunk]:
        """异步迭代队列，直到关闭并排空。"""
        while True:
            chunk = await self.dequeue()
            if chunk is None:
                return
            yield chunk

    def peek(self) -> AudioChunk | None:
        """查看队首帧但不取出。"""
        return self._queue[0] if self._queue else None

    # ---------------- 撤销（barge-in 关键路径） ----------------

    def cancel_from(self, pts_ms: int | None = None) -> int:
        """丢弃 ``pts >= pts_ms`` 的所有未播帧。

        **同步、无 await、O(n)** —— barge-in 路径要求 <200ms 完成三路止损，
        这里绝对不能挂起。

        Args:
            pts_ms: 起始 pts；``None`` 表示清空整个队列。

        Returns:
            被丢弃的音频毫秒数（取整）。
        """
        if not self._queue:
            self._epoch += 1
            self.stats.cancellations += 1
            return 0
        dropped_ms = 0.0
        dropped_frames = 0
        if pts_ms is None:
            for chunk in self._queue:
                dropped_ms += chunk.duration_ms
                dropped_frames += 1
            self._queue.clear()
        else:
            kept: deque[AudioChunk] = deque()
            for chunk in self._queue:
                if chunk.pts_ms >= pts_ms:
                    dropped_ms += chunk.duration_ms
                    dropped_frames += 1
                else:
                    kept.append(chunk)
            self._queue = kept
        self._buffered_ms = max(0.0, self._buffered_ms - dropped_ms)
        self._epoch += 1
        self.stats.cancellations += 1
        self.stats.cancelled_frames += dropped_frames
        self.stats.cancelled_ms += dropped_ms
        try:
            metrics.playback_cancelled_ms.inc(dropped_ms)
        except Exception:
            pass
        self._space_ready.set()
        if not self._queue:
            self._data_ready.clear()
        log.info(
            "playback_cancelled",
            session_id=self.session_id,
            from_pts_ms=pts_ms,
            dropped_frames=dropped_frames,
            dropped_ms=int(dropped_ms),
            played_ms=int(self._played_ms),
            epoch=self._epoch,
        )
        return int(dropped_ms)

    def clear(self) -> int:
        """等价于 ``cancel_from(None)``。"""
        return self.cancel_from(None)

    # ---------------- 生命周期 ----------------

    def reset_timeline(self, *, played_ms: float = 0.0) -> None:
        """开始新一轮播放时重置时间轴统计。"""
        self._played_ms = played_ms

    def close(self) -> None:
        """关闭队列：唤醒所有等待者，消费者排空后结束迭代。"""
        if self._closed:
            return
        self._closed = True
        self._data_ready.set()
        self._space_ready.set()
        log.debug("playback_queue_closed", session_id=self.session_id,
                  played_ms=int(self._played_ms))

    def snapshot(self) -> dict[str, float]:
        """统计快照。"""
        data = self.stats.as_dict()
        data["pending_ms"] = round(self._buffered_ms, 1)
        data["epoch"] = float(self._epoch)
        return data

    def __repr__(self) -> str:
        return (
            f"<PlaybackQueue {self.name} frames={len(self._queue)} "
            f"pending={self._buffered_ms:.0f}ms played={self._played_ms:.0f}ms>"
        )

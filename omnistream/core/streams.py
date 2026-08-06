"""有界异步流：全链路唯一允许的跨协程数据通道。"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Generic, TypeVar

T = TypeVar("T")


class OverflowPolicy(StrEnum):
    BLOCK = "block"              # 背压到上游（音频入口用）
    DROP_OLDEST = "drop_oldest"  # 丢最旧（视频帧用）
    DROP_NEWEST = "drop_newest"  # 丢最新（指标 tick 用）


class StreamClosed(RuntimeError):
    pass


class BoundedStream(Generic[T]):
    def __init__(self, maxsize: int = 64, policy: OverflowPolicy = OverflowPolicy.BLOCK,
                 name: str = "stream") -> None:
        self._q: asyncio.Queue[T | None] = asyncio.Queue(maxsize=maxsize)
        self._policy = policy
        self._closed = False
        self.name = name
        self.dropped = 0

    async def put(self, item: T) -> None:
        if self._closed:
            raise StreamClosed(f"{self.name} 已关闭")
        if self._policy is OverflowPolicy.BLOCK:
            await self._q.put(item)
            return
        try:
            self._q.put_nowait(item)
        except asyncio.QueueFull:
            self.dropped += 1
            if self._policy is OverflowPolicy.DROP_OLDEST:
                try:
                    self._q.get_nowait()
                    self._q.task_done()
                except asyncio.QueueEmpty:
                    pass
                self._q.put_nowait(item)
            # DROP_NEWEST: 直接丢弃 item

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._q.put_nowait(None)
            except asyncio.QueueFull:
                pass

    def clear(self) -> int:
        """丢弃全部待处理项（barge-in 时清播放队列用），返回丢弃数量。"""
        n = 0
        while True:
            try:
                self._q.get_nowait()
                self._q.task_done()
                n += 1
            except asyncio.QueueEmpty:
                return n

    @property
    def size(self) -> int:
        return self._q.qsize()

    @property
    def closed(self) -> bool:
        return self._closed

    async def __aiter__(self) -> AsyncIterator[T]:  # type: ignore[override]
        while True:
            item = await self._q.get()
            self._q.task_done()
            if item is None:
                return
            yield item

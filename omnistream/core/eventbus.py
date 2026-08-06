"""事件总线：进程内分发 + Redis Streams 跨进程分发。

实时链路（音频帧、token 流）**不走**事件总线，走 ``BoundedStream``；
事件总线用于低频的控制面事件：会话生命周期、配额变更、Webhook 触发、审计。
"""
from __future__ import annotations

import abc
import asyncio
import contextlib
import fnmatch
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import orjson

from omnistream.core.clock import now_ms
from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import OmniError
from omnistream.core.ids import new_id
from omnistream.core.logging import get_logger

__all__ = [
    "BusEvent",
    "EventBus",
    "EventHandler",
    "InProcEventBus",
    "RedisStreamEventBus",
    "create_event_bus",
]

log = get_logger(__name__)

EventHandler = Callable[["BusEvent"], Awaitable[None]]


@dataclass(slots=True)
class BusEvent:
    """总线事件。``topic`` 用点分层级，支持 ``session.*`` 通配订阅。"""

    topic: str
    payload: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("event"))
    ts_ms: int = field(default_factory=now_ms)
    tenant_id: str | None = None
    source: str = "omnistream"

    def to_wire(self) -> bytes:
        return orjson.dumps({
            "id": self.id,
            "topic": self.topic,
            "ts_ms": self.ts_ms,
            "tenant_id": self.tenant_id,
            "source": self.source,
            "payload": self.payload,
        })

    @classmethod
    def from_wire(cls, raw: bytes | str) -> BusEvent:
        data = orjson.loads(raw)
        return cls(
            topic=str(data.get("topic", "")),
            payload=dict(data.get("payload") or {}),
            id=str(data.get("id") or new_id("event")),
            ts_ms=int(data.get("ts_ms") or now_ms()),
            tenant_id=data.get("tenant_id"),
            source=str(data.get("source") or "omnistream"),
        )


class EventBus(abc.ABC):
    """事件总线抽象。实现必须保证 ``publish`` 不阻塞调用方超过毫秒级。"""

    @abc.abstractmethod
    async def start(self) -> None:
        """建立底层连接并启动消费循环。幂等。"""

    @abc.abstractmethod
    async def stop(self) -> None:
        """停止消费并释放连接。幂等。"""

    @abc.abstractmethod
    async def publish(self, event: BusEvent) -> None:
        """发布一个事件。"""

    @abc.abstractmethod
    def subscribe(self, pattern: str, handler: EventHandler) -> Callable[[], None]:
        """订阅匹配 ``pattern`` 的事件，返回退订函数。

        ``pattern`` 支持 fnmatch 通配，如 ``session.*``、``*``。
        """

    async def emit(self, topic: str, /, **payload: Any) -> None:
        """``publish`` 的语法糖。"""
        await self.publish(BusEvent(topic=topic, payload=payload))


class InProcEventBus(EventBus):
    """进程内事件总线。单副本部署与 mock 模式的默认实现。

    每个订阅者独享一条有界队列 + 一个消费协程，慢订阅者不会阻塞发布方，
    队列打满时丢弃最旧事件并计数告警。
    """

    def __init__(self, queue_size: int = 256) -> None:
        self._queue_size = queue_size
        self._subs: dict[str, tuple[str, EventHandler, asyncio.Queue[BusEvent | None]]] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._running = False
        self.dropped = 0

    async def start(self) -> None:
        self._running = True
        log.info("eventbus_started", backend="inproc", subscribers=len(self._subs))

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for _, _, queue in self._subs.values():
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(None)
        for task in list(self._tasks.values()):
            task.cancel()
        for task in list(self._tasks.values()):
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
        self._subs.clear()
        log.info("eventbus_stopped", backend="inproc")

    async def publish(self, event: BusEvent) -> None:
        for sub_id, (pattern, _handler, queue) in list(self._subs.items()):
            if not fnmatch.fnmatch(event.topic, pattern):
                continue
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                self.dropped += 1
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(event)
                log.warning("eventbus_overflow", subscriber=sub_id, topic=event.topic, dropped=self.dropped)

    def subscribe(self, pattern: str, handler: EventHandler) -> Callable[[], None]:
        sub_id = new_id("event")
        queue: asyncio.Queue[BusEvent | None] = asyncio.Queue(maxsize=self._queue_size)
        self._subs[sub_id] = (pattern, handler, queue)
        self._tasks[sub_id] = asyncio.create_task(
            self._consume(sub_id, handler, queue), name=f"eventbus_sub:{pattern}"
        )

        def unsubscribe() -> None:
            self._subs.pop(sub_id, None)
            task = self._tasks.pop(sub_id, None)
            if task is not None:
                task.cancel()

        return unsubscribe

    async def _consume(self, sub_id: str, handler: EventHandler, queue: asyncio.Queue[BusEvent | None]) -> None:
        while True:
            event = await queue.get()
            if event is None:
                return
            try:
                await handler(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # 单个订阅者异常不得影响总线
                log.error("eventbus_handler_failed", subscriber=sub_id, topic=event.topic, error=str(exc))


class RedisStreamEventBus(EventBus):
    """基于 Redis Streams 的跨进程事件总线。

    所有事件写入同一条 stream（``{namespace}:events``），消费者组保证多副本下
    每条事件只被同组内一个副本处理；订阅侧再按 topic 做本地 fnmatch 派发。
    """

    def __init__(
        self,
        url: str,
        *,
        namespace: str = "oms",
        group: str = "oms-workers",
        consumer: str | None = None,
        maxlen: int = 10_000,
        block_ms: int = 2_000,
        batch: int = 64,
    ) -> None:
        self._url = url
        self._stream = f"{namespace}:events"
        self._group = group
        self._consumer = consumer or new_id("event")
        self._maxlen = maxlen
        self._block_ms = block_ms
        self._batch = batch
        self._redis: Any = None
        self._task: asyncio.Task[None] | None = None
        self._subs: dict[str, tuple[str, EventHandler]] = {}
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        try:
            from redis.asyncio import Redis
        except ImportError as exc:  # pragma: no cover - 依赖缺失
            raise OmniError(
                "RedisStreamEventBus 需要 redis 包，请安装 requirements.txt",
                code=ErrorCode.CONFIG_MISSING,
                cause=exc,
            ) from exc
        self._redis = Redis.from_url(self._url, decode_responses=False)
        await self._redis.ping()
        try:
            await self._redis.xgroup_create(self._stream, self._group, id="$", mkstream=True)
        except Exception as exc:  # BUSYGROUP: 组已存在，属正常情况
            if "BUSYGROUP" not in str(exc):
                raise
        self._running = True
        self._task = asyncio.create_task(self._consume_loop(), name="eventbus_redis_consume")
        log.info("eventbus_started", backend="redis", stream=self._stream, group=self._group)

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
        self._subs.clear()
        log.info("eventbus_stopped", backend="redis")

    async def publish(self, event: BusEvent) -> None:
        if self._redis is None:
            raise OmniError("事件总线未启动", code=ErrorCode.UNAVAILABLE)
        await self._redis.xadd(
            self._stream,
            {b"topic": event.topic.encode(), b"data": event.to_wire()},
            maxlen=self._maxlen,
            approximate=True,
        )

    def subscribe(self, pattern: str, handler: EventHandler) -> Callable[[], None]:
        sub_id = new_id("event")
        self._subs[sub_id] = (pattern, handler)

        def unsubscribe() -> None:
            self._subs.pop(sub_id, None)

        return unsubscribe

    async def _consume_loop(self) -> None:
        while self._running:
            try:
                batches = await self._redis.xreadgroup(
                    self._group, self._consumer, {self._stream: ">"},
                    count=self._batch, block=self._block_ms,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("eventbus_read_failed", error=str(exc))
                await asyncio.sleep(1.0)
                continue
            if not batches:
                continue
            for _stream_name, messages in batches:
                for message_id, fields in messages:
                    raw = fields.get(b"data")
                    if raw:
                        await self._dispatch(raw)
                    with contextlib.suppress(Exception):
                        await self._redis.xack(self._stream, self._group, message_id)

    async def _dispatch(self, raw: bytes) -> None:
        try:
            event = BusEvent.from_wire(raw)
        except Exception as exc:
            log.warning("eventbus_decode_failed", error=str(exc))
            return
        for sub_id, (pattern, handler) in list(self._subs.items()):
            if not fnmatch.fnmatch(event.topic, pattern):
                continue
            try:
                await handler(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("eventbus_handler_failed", subscriber=sub_id, topic=event.topic, error=str(exc))


def create_event_bus(backend: str = "inproc", *, redis_url: str = "", namespace: str = "oms") -> EventBus:
    """按后端名构造事件总线。``inproc`` | ``redis``。"""
    normalized = backend.strip().lower()
    if normalized in ("inproc", "memory", "local"):
        return InProcEventBus()
    if normalized == "redis":
        if not redis_url:
            raise OmniError("redis 事件总线必须提供 redis_url", code=ErrorCode.CONFIG_MISSING)
        return RedisStreamEventBus(redis_url, namespace=namespace)
    raise OmniError(f"未知事件总线后端: {backend}", code=ErrorCode.CONFIG_INVALID)


async def collect(bus: EventBus, pattern: str, limit: int) -> AsyncIterator[BusEvent]:
    """把订阅转成异步迭代器，最多产出 ``limit`` 个事件后自动退订。"""
    queue: asyncio.Queue[BusEvent] = asyncio.Queue(maxsize=max(1, limit))

    async def _handler(event: BusEvent) -> None:
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(event)

    unsubscribe = bus.subscribe(pattern, _handler)
    try:
        for _ in range(limit):
            yield await queue.get()
    finally:
        unsubscribe()

"""WebSocket 传输：生产环境默认链路。

线上格式
--------
* **文本帧** —— 一条 :class:`~omnistream.protocol.envelope.Envelope` 的 JSON。
* **二进制帧** —— OMS1 帧（见 :mod:`omnistream.protocol.binary`），
  音频/视频负载直传，不做 base64，省掉 33% 膨胀与 GC 压力。

适配层
------
本模块**不硬依赖** FastAPI / starlette / websockets。真实的 socket 对象通过
:func:`adapt_websocket` 做鸭子类型识别，包成统一的 :class:`WebSocketAdapter`：

* 服务端（starlette / FastAPI 的 ``WebSocket``）—— 有 ``send_bytes`` / ``receive``。
* 客户端（``websockets`` 库的连接对象）—— 有 ``send`` / ``recv`` / ``ping``。
* 任何自定义对象只要实现 :class:`WebSocketAdapter` 的四个方法即可直接传入。

保活
----
* 服务端每 ``heartbeat_interval_ms`` 发一次心跳：底层支持 WebSocket ping 时用 ping
  并顺带测出真实 RTT；否则退化为一条 ``pong`` 事件（payload 带 ``heartbeat``）。
* 上行静默超过 ``idle_timeout_ms`` 主动断链，避免半开连接占着会话资源。
"""
from __future__ import annotations

import abc
import asyncio
from typing import Any

from omnistream.core.clock import monotonic_s, now_ms
from omnistream.core.logging import get_logger
from omnistream.media.errors import TransportClosed
from omnistream.media.metrics import transport_errors
from omnistream.media.transport.base import BaseTransport, TransportConfig, TransportKind
from omnistream.protocol.enums import ClientEventType, ServerEventType

__all__ = [
    "CallableWebSocketAdapter",
    "ClientWebSocketAdapter",
    "ServerWebSocketAdapter",
    "WebSocketAdapter",
    "WebSocketTransport",
    "adapt_websocket",
]

_log = get_logger(__name__)

#: ``receive()`` 返回值的第一元素。
FRAME_TEXT = "text"
FRAME_BINARY = "binary"


class WebSocketAdapter(abc.ABC):
    """统一的 WebSocket 适配接口。"""

    @abc.abstractmethod
    async def send_bytes(self, data: bytes) -> None:
        """发送二进制帧。"""

    @abc.abstractmethod
    async def send_text(self, data: str) -> None:
        """发送文本帧。"""

    @abc.abstractmethod
    async def receive(self) -> tuple[str, bytes | str] | None:
        """接收一帧。

        Returns:
            ``(FRAME_TEXT, str)`` 或 ``(FRAME_BINARY, bytes)``；
            对端关闭时返回 ``None``。
        """

    @abc.abstractmethod
    async def close(self, code: int = 1000) -> None:
        """关闭底层连接，必须吞掉自身清理异常。"""

    async def heartbeat(self) -> float | None:
        """发一次底层 ping 并返回 RTT（毫秒）。

        Returns:
            RTT 毫秒；底层不支持 ping 时返回 ``None``。
        """
        return None

    @property
    def remote(self) -> str:
        """对端标识，仅用于日志。"""
        return "unknown"


class ServerWebSocketAdapter(WebSocketAdapter):
    """starlette / FastAPI ``WebSocket`` 适配器。"""

    def __init__(self, websocket: Any) -> None:
        """
        Args:
            websocket: starlette 的 ``WebSocket`` 实例。
        """
        self._ws = websocket

    async def send_bytes(self, data: bytes) -> None:
        """发送二进制帧。"""
        await self._ws.send_bytes(data)

    async def send_text(self, data: str) -> None:
        """发送文本帧。"""
        await self._ws.send_text(data)

    async def receive(self) -> tuple[str, bytes | str] | None:
        """从 ASGI 消息中解出一帧。"""
        message = await self._ws.receive()
        kind = message.get("type", "")
        if kind == "websocket.disconnect":
            return None
        payload = message.get("bytes")
        if payload is not None:
            return (FRAME_BINARY, payload)
        text = message.get("text")
        if text is not None:
            return (FRAME_TEXT, text)
        return None

    async def close(self, code: int = 1000) -> None:
        """关闭连接。"""
        try:
            await self._ws.close(code=code)
        except Exception:
            return None

    @property
    def remote(self) -> str:
        """对端地址。"""
        client = getattr(self._ws, "client", None)
        if client is None:
            return "server-socket"
        host = getattr(client, "host", "")
        port = getattr(client, "port", "")
        return f"{host}:{port}" if host else "server-socket"


class ClientWebSocketAdapter(WebSocketAdapter):
    """``websockets`` 库连接对象适配器（客户端 / 压测脚本用）。"""

    def __init__(self, connection: Any) -> None:
        """
        Args:
            connection: ``websockets`` 的连接对象。
        """
        self._ws = connection

    async def send_bytes(self, data: bytes) -> None:
        """发送二进制帧。"""
        await self._ws.send(data)

    async def send_text(self, data: str) -> None:
        """发送文本帧。"""
        await self._ws.send(data)

    async def receive(self) -> tuple[str, bytes | str] | None:
        """接收一帧，连接关闭时返回 ``None``。"""
        try:
            message = await self._ws.recv()
        except Exception:
            return None
        if isinstance(message, (bytes, bytearray, memoryview)):
            return (FRAME_BINARY, bytes(message))
        return (FRAME_TEXT, str(message))

    async def close(self, code: int = 1000) -> None:
        """关闭连接。"""
        try:
            await self._ws.close(code)
        except TypeError:
            try:
                await self._ws.close()
            except Exception:
                return None
        except Exception:
            return None

    async def heartbeat(self) -> float | None:
        """用 WebSocket ping 测量真实 RTT。"""
        ping = getattr(self._ws, "ping", None)
        if ping is None:
            return None
        started = monotonic_s()
        try:
            waiter = await ping()
            if asyncio.isfuture(waiter) or asyncio.iscoroutine(waiter):
                await waiter
        except Exception:
            return None
        return (monotonic_s() - started) * 1000.0

    @property
    def remote(self) -> str:
        """对端地址。"""
        return str(getattr(self._ws, "remote_address", "client-socket"))


class CallableWebSocketAdapter(WebSocketAdapter):
    """由三个可调用对象拼出的适配器，方便测试与自定义链路。"""

    def __init__(
        self,
        *,
        send_bytes: Any,
        send_text: Any,
        receive: Any,
        close: Any | None = None,
        remote: str = "callable",
    ) -> None:
        """
        Args:
            send_bytes: ``async (bytes) -> None``。
            send_text: ``async (str) -> None``。
            receive: ``async () -> tuple[str, bytes | str] | None``。
            close: ``async (int) -> None``，可为 ``None``。
            remote: 对端标识。
        """
        self._send_bytes = send_bytes
        self._send_text = send_text
        self._receive = receive
        self._close = close
        self._remote = remote

    async def send_bytes(self, data: bytes) -> None:
        """发送二进制帧。"""
        await self._send_bytes(data)

    async def send_text(self, data: str) -> None:
        """发送文本帧。"""
        await self._send_text(data)

    async def receive(self) -> tuple[str, bytes | str] | None:
        """接收一帧。"""
        return await self._receive()

    async def close(self, code: int = 1000) -> None:
        """关闭连接。"""
        if self._close is None:
            return None
        try:
            await self._close(code)
        except Exception:
            return None

    @property
    def remote(self) -> str:
        """对端标识。"""
        return self._remote


def adapt_websocket(obj: Any) -> WebSocketAdapter:
    """把任意 WebSocket 对象包装为 :class:`WebSocketAdapter`。

    Args:
        obj: starlette ``WebSocket`` / ``websockets`` 连接 / 已是适配器的对象。

    Returns:
        统一适配器。

    Raises:
        TypeError: 对象既不像服务端也不像客户端 socket。
    """
    if isinstance(obj, WebSocketAdapter):
        return obj
    if hasattr(obj, "send_bytes") and hasattr(obj, "receive"):
        return ServerWebSocketAdapter(obj)
    if hasattr(obj, "send") and hasattr(obj, "recv"):
        return ClientWebSocketAdapter(obj)
    raise TypeError(
        "unsupported websocket object; expected starlette WebSocket, "
        f"websockets connection or WebSocketAdapter, got {type(obj).__name__}"
    )


class WebSocketTransport(BaseTransport):
    """基于 WebSocket 的媒体链路。

    Example::

        transport = WebSocketTransport(websocket, TransportConfig(session_id=sid))
        await transport.start()
        async for chunk in transport.recv_audio():
            ...
    """

    kind = TransportKind.WEBSOCKET

    def __init__(
        self,
        websocket: Any,
        config: TransportConfig | None = None,
        *,
        auto_pong: bool = True,
        forward_ping: bool = False,
    ) -> None:
        """
        Args:
            websocket: 底层 socket 对象或适配器。
            config: 链路配置。
            auto_pong: 是否在传输层直接应答客户端 ``ping``。
            forward_ping: 应答后是否仍把 ``ping`` 事件投递给上层。
        """
        super().__init__(config)
        self.adapter = adapt_websocket(websocket)
        self.auto_pong = bool(auto_pong)
        self.forward_ping = bool(forward_ping)
        self._reader_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._frames_read = 0

    # ------------------------------------------------------------------
    # 生命周期钩子
    # ------------------------------------------------------------------

    async def _open(self) -> None:
        """启动读循环与心跳。"""
        self._reader_task = asyncio.create_task(
            self._read_loop(), name=f"ws_reader:{self.session_id}"
        )
        if self.config.heartbeat_interval_ms > 0:
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(), name=f"ws_heartbeat:{self.session_id}"
            )
        self._log.info("websocket_opened", remote=self.adapter.remote)

    async def _close(self, reason: str) -> None:
        """停掉后台任务并关闭 socket。"""
        current = asyncio.current_task()
        for attr in ("_heartbeat_task", "_reader_task"):
            task: asyncio.Task[None] | None = getattr(self, attr)
            setattr(self, attr, None)
            if task is None or task is current:
                continue
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        code = 1000 if reason in ("normal", "peer_closed") else 1011
        await self.adapter.close(code)

    async def _send_binary(self, payload: bytes, *, media: str) -> None:
        """发送 OMS1 二进制帧。"""
        try:
            await self.adapter.send_bytes(payload)
        except Exception as exc:
            self.stats.on_error()
            try:
                transport_errors.labels(kind=str(self.kind), reason="send_failed").inc()
            except Exception:
                pass
            raise TransportClosed(
                "websocket binary send failed",
                detail={"session_id": self.session_id, "media": media},
                cause=exc,
            ) from exc

    async def _send_text(self, payload: bytes) -> None:
        """发送 Envelope 文本帧。"""
        try:
            await self.adapter.send_text(payload.decode("utf-8"))
        except Exception as exc:
            self.stats.on_error()
            try:
                transport_errors.labels(kind=str(self.kind), reason="send_failed").inc()
            except Exception:
                pass
            raise TransportClosed(
                "websocket text send failed",
                detail={"session_id": self.session_id},
                cause=exc,
            ) from exc

    # ------------------------------------------------------------------
    # 读循环
    # ------------------------------------------------------------------

    async def _read_loop(self) -> None:
        """持续读取上行帧并分发，直到对端关闭或出错。"""
        reason = "peer_closed"
        try:
            while True:
                frame = await self.adapter.receive()
                if frame is None:
                    break
                kind, payload = frame
                self._frames_read += 1
                if kind == FRAME_BINARY and isinstance(payload, (bytes, bytearray)):
                    await self._handle_binary(bytes(payload))
                elif kind == FRAME_TEXT:
                    await self._handle_ws_text(payload)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            reason = f"reader_error: {exc}"
            self.stats.on_error()
            try:
                transport_errors.labels(kind=str(self.kind), reason="reader_error").inc()
            except Exception:
                pass
            self._log.warning("websocket_read_failed", error=str(exc))
        finally:
            if not self.closed:
                asyncio.create_task(
                    self.stop(reason), name=f"ws_autostop:{self.session_id}"
                )

    async def _handle_ws_text(self, payload: bytes | str) -> None:
        """处理一条文本帧，必要时在传输层直接回 pong。"""
        data = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
        if self.auto_pong and self._maybe_pong(data):
            await self._reply_pong(data)
            if not self.forward_ping:
                self.stats.on_recv(len(data), media="event")
                return
        await self._handle_text(data)

    @staticmethod
    def _maybe_pong(data: bytes) -> bool:
        """快速判断一条文本是否是 ping（避免无谓的 JSON 解析）。"""
        return b'"' + str(ClientEventType.PING).encode("utf-8") + b'"' in data

    async def _reply_pong(self, data: bytes) -> None:
        """回一条 pong，回显客户端时间戳以便对端算 RTT。"""
        client_ts = 0
        try:
            from omnistream.protocol.envelope import Envelope

            envelope = Envelope.from_wire(data)
            if envelope.type != str(ClientEventType.PING):
                return
            raw_ts = envelope.payload.get("ts", 0)
            client_ts = int(raw_ts) if isinstance(raw_ts, (int, float, str)) else 0
        except Exception:
            client_ts = 0
        now = now_ms()
        if client_ts > 0:
            rtt = now - client_ts
            if 0 <= rtt < 60_000:
                self.stats.on_rtt(float(rtt))
        try:
            await self.send_event(
                ServerEventType.PONG, {"ts": client_ts, "server_ts": now}
            )
        except TransportClosed:
            return

    # ------------------------------------------------------------------
    # 心跳
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """周期心跳 + 空闲断链。"""
        period = max(1.0, self.config.heartbeat_interval_ms / 1000.0)
        while not self.closed:
            try:
                await asyncio.sleep(period)
            except asyncio.CancelledError:
                return
            if not self.connected:
                continue
            if (
                self.config.idle_timeout_ms > 0
                and self.stats.idle_ms >= self.config.idle_timeout_ms
            ):
                self._log.warning("websocket_idle_timeout", idle_ms=self.stats.idle_ms)
                asyncio.create_task(
                    self.stop("idle_timeout"), name=f"ws_idlestop:{self.session_id}"
                )
                return
            try:
                rtt = await self.adapter.heartbeat()
            except Exception:
                rtt = None
            if rtt is not None:
                self.stats.on_rtt(rtt)
                continue
            try:
                await self.send_event(
                    ServerEventType.PONG, {"heartbeat": True, "ts": now_ms()}
                )
            except TransportClosed:
                return
            except Exception as exc:
                self._log.warning("websocket_heartbeat_failed", error=str(exc))
                return

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """在基类快照上补充 WebSocket 特有信息。"""
        snapshot = super().describe()
        snapshot["websocket"] = {
            "remote": self.adapter.remote,
            "frames_read": self._frames_read,
            "auto_pong": self.auto_pong,
            "reader_alive": self._reader_task is not None and not self._reader_task.done(),
            "heartbeat_alive": (
                self._heartbeat_task is not None and not self._heartbeat_task.done()
            ),
        }
        return snapshot

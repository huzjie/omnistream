"""进程内回环链路：不碰网络也能跑通完整媒体管线。

三种拓扑
--------
* **自回声**（``echo=True``）—— 发出去的帧原路回到自己的上行流，
  一个进程即可验证 ingress → provider → egress 的闭环。
* **对等**（:meth:`LoopbackTransport.pair`）—— 两个端点互为对端，
  模拟"客户端 ↔ 服务端"，压测脚本与 Demo 用这个。
* **单向注入**（:meth:`inject_audio` / :meth:`inject_video`）—— 由信号源直接灌入上行流，
  用于回放 WAV / 合成语音。

回环链路走的是**真实的 OMS1 编解码路径**（``encode_audio`` → ``decode_frame``），
不是对象直传，因此协议层的 bug 在这里同样会暴露出来。

网络损伤通过 :class:`~omnistream.media.transport.base.LinkImpairment` 结构化协议注入，
实现见 :mod:`omnistream.media.simulate.network` —— 传输层对 ``simulate`` 没有反向依赖。
"""
from __future__ import annotations

import asyncio
from typing import Any

from omnistream.core.logging import get_logger
from omnistream.media.transport.base import (
    BaseTransport,
    LinkImpairment,
    TransportConfig,
    TransportKind,
)
from omnistream.schemas.audio import AudioChunk
from omnistream.schemas.video import VideoFrame

__all__ = ["LoopbackTransport"]

_log = get_logger(__name__)


class LoopbackTransport(BaseTransport):
    """进程内回环传输。

    Example::

        left, right = LoopbackTransport.pair(session_id="sess_demo")
        await left.start()
        await right.start()
        await left.send_audio(chunk)
        async for received in right.recv_audio():
            ...
    """

    kind = TransportKind.LOOPBACK

    def __init__(
        self,
        config: TransportConfig | None = None,
        *,
        echo: bool = False,
        impairment: LinkImpairment | None = None,
        name: str = "loopback",
    ) -> None:
        """
        Args:
            config: 链路配置。
            echo: 无对端时是否把下行帧回灌到自己的上行流。
            impairment: 可选的网络损伤注入器。
            name: 端点名，出现在日志里便于区分 pair 的两端。
        """
        super().__init__(config)
        self.echo = bool(echo)
        self.impairment = impairment
        self.name = name
        self._peer: LoopbackTransport | None = None
        self._pending: set[asyncio.Task[None]] = set()
        self._delivered = 0
        self._dropped = 0

    # ------------------------------------------------------------------
    # 拓扑
    # ------------------------------------------------------------------

    @classmethod
    def pair(
        cls,
        *,
        session_id: str = "",
        config: TransportConfig | None = None,
        impairment: LinkImpairment | None = None,
    ) -> tuple["LoopbackTransport", "LoopbackTransport"]:
        """创建一对互联的回环端点。

        Args:
            session_id: 会话 ID，两端共用。
            config: 模板配置，会被复制到两端（``None`` 时使用默认值）。
            impairment: 双向共用的损伤注入器。

        Returns:
            ``(client, server)`` 两个已互相 attach 的端点，尚未 ``start()``。
        """
        base = config or TransportConfig(session_id=session_id, kind=TransportKind.LOOPBACK)
        if session_id:
            base.session_id = session_id
        left_cfg = TransportConfig(
            session_id=base.session_id,
            kind=TransportKind.LOOPBACK,
            sample_rate=base.sample_rate,
            channels=base.channels,
            frame_ms=base.frame_ms,
            audio_encoding=base.audio_encoding,
            video_encoding=base.video_encoding,
            audio_maxsize=base.audio_maxsize,
            video_maxsize=base.video_maxsize,
            event_maxsize=base.event_maxsize,
            stall_timeout_ms=base.stall_timeout_ms,
            enable_video=base.enable_video,
            enable_stats=base.enable_stats,
            enable_watchdog=base.enable_watchdog,
        )
        right_cfg = TransportConfig(
            session_id=base.session_id,
            kind=TransportKind.LOOPBACK,
            sample_rate=base.sample_rate,
            channels=base.channels,
            frame_ms=base.frame_ms,
            audio_encoding=base.audio_encoding,
            video_encoding=base.video_encoding,
            audio_maxsize=base.audio_maxsize,
            video_maxsize=base.video_maxsize,
            event_maxsize=base.event_maxsize,
            stall_timeout_ms=base.stall_timeout_ms,
            enable_video=base.enable_video,
            enable_stats=base.enable_stats,
            enable_watchdog=base.enable_watchdog,
        )
        left = cls(left_cfg, impairment=impairment, name="client")
        right = cls(right_cfg, impairment=impairment, name="server")
        left.attach_peer(right)
        return left, right

    def attach_peer(self, peer: "LoopbackTransport", *, bidirectional: bool = True) -> None:
        """把另一个回环端点设为对端。

        Args:
            peer: 对端。
            bidirectional: 是否同时把自己设为对端的对端。
        """
        self._peer = peer
        if bidirectional and peer._peer is not self:
            peer.attach_peer(self, bidirectional=False)

    @property
    def peer(self) -> "LoopbackTransport | None":
        """当前对端。"""
        return self._peer

    @property
    def delivered(self) -> int:
        """已成功投递的分片数。"""
        return self._delivered

    @property
    def dropped(self) -> int:
        """被损伤注入器丢弃的分片数。"""
        return self._dropped

    # ------------------------------------------------------------------
    # 生命周期钩子
    # ------------------------------------------------------------------

    async def _open(self) -> None:
        """回环链路无需握手，只做拓扑校验。"""
        if self._peer is None and not self.echo:
            self._log.debug("loopback_open_without_peer", endpoint=self.name)

    async def _close(self, reason: str) -> None:
        """取消尚未投递的分片。"""
        pending = list(self._pending)
        self._pending.clear()
        for task in pending:
            task.cancel()
        for task in pending:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    # ------------------------------------------------------------------
    # 发送
    # ------------------------------------------------------------------

    async def _send_binary(self, payload: bytes, *, media: str) -> None:
        """按损伤策略投递一帧二进制数据。"""
        self._schedule(payload, media=media, text=False)

    async def _send_text(self, payload: bytes) -> None:
        """按损伤策略投递一条文本消息。"""
        self._schedule(payload, media="event", text=True)

    def _schedule(self, payload: bytes, *, media: str, text: bool) -> None:
        """规划一次投递。

        Args:
            payload: 待投递字节。
            media: ``audio`` / ``video`` / ``event``。
            text: 是否按文本消息处理。
        """
        target = self._target()
        if target is None:
            return
        delay_s = 0.0
        dropped = False
        if self.impairment is not None:
            try:
                delay_s, dropped = self.impairment.plan(payload)
            except Exception as exc:  # 损伤器异常不能影响主链路
                self._log.warning("impairment_plan_failed", error=str(exc))
                delay_s, dropped = 0.0, False
        if dropped:
            self._dropped += 1
            target.stats.on_loss(1, 1)
            return
        target.stats.on_loss(0, 1)
        task = asyncio.create_task(
            self._deliver(target, payload, media=media, text=text, delay_s=delay_s),
            name=f"loopback_deliver:{self.name}:{media}",
        )
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    def _target(self) -> "LoopbackTransport | None":
        """解析本次投递的接收端。"""
        if self._peer is not None:
            return self._peer
        return self if self.echo else None

    async def _deliver(
        self,
        target: "LoopbackTransport",
        payload: bytes,
        *,
        media: str,
        text: bool,
        delay_s: float,
    ) -> None:
        """执行一次（可能延迟的）投递。"""
        try:
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            if target.closed:
                return
            target.stats.on_transit(delay_s * 1000.0)
            if text:
                await target._handle_text(payload)
            else:
                await target._handle_binary(payload)
            self._delivered += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.stats.on_error()
            self._log.warning("loopback_deliver_failed", media=media, error=str(exc))

    async def flush(self, timeout_s: float = 2.0) -> int:
        """等待所有在途分片投递完成。

        Args:
            timeout_s: 最长等待时间。

        Returns:
            等待结束时仍未完成的分片数。
        """
        deadline = asyncio.get_running_loop().time() + max(0.0, timeout_s)
        while self._pending:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            pending = list(self._pending)
            done, _ = await asyncio.wait(pending, timeout=remaining)
            if not done:
                break
        return len(self._pending)

    # ------------------------------------------------------------------
    # 直接注入（信号源使用）
    # ------------------------------------------------------------------

    async def inject_audio(self, chunk: AudioChunk) -> None:
        """把一帧音频直接注入本端上行流（跳过链路）。"""
        self.stats.on_recv(len(chunk.data), media="audio")
        await self._emit_audio(chunk)

    async def inject_video(self, frame: VideoFrame) -> None:
        """把一帧视频直接注入本端上行流（跳过链路）。"""
        self.stats.on_recv(len(frame.data), media="video")
        await self._emit_video(frame)

    def describe(self) -> dict[str, Any]:
        """在基类快照上补充回环特有信息。"""
        snapshot = super().describe()
        snapshot["loopback"] = {
            "endpoint": self.name,
            "echo": self.echo,
            "has_peer": self._peer is not None,
            "pending": len(self._pending),
            "delivered": self._delivered,
            "dropped": self._dropped,
            "impaired": self.impairment is not None,
        }
        return snapshot

"""DataChannel 分片与重组。

为什么需要分片
--------------
SCTP 的单条消息在各家实现上限不一（Chrome 历史上 64KB，部分实现更低），
而一帧 720p JPEG 轻松超过 100KB。直接发大消息会在真实网络上随机失败，
所以这里统一切成 :data:`~omnistream.media.constants.DATACHANNEL_MTU` 大小的分片。

分片头（12 字节，小端）::

    offset  size  field
    0       4     magic = b"OMSC"
    4       4     msg_id  (uint32, 发送端自增)
    8       2     index   (uint16, 从 0 开始)
    10      2     total   (uint16, 分片总数)

重组端按 ``msg_id`` 聚合，超时（:data:`DATACHANNEL_REASSEMBLY_TIMEOUT_MS`）或
待重组消息数超过 :data:`DATACHANNEL_MAX_PENDING_MESSAGES` 时丢弃最旧的半成品，
永远不会无界增长。
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

from omnistream.core.clock import now_ms
from omnistream.core.logging import get_logger
from omnistream.media.constants import (
    DATACHANNEL_MAX_PENDING_MESSAGES,
    DATACHANNEL_MTU,
    DATACHANNEL_REASSEMBLY_TIMEOUT_MS,
)
from omnistream.media.errors import CodecError
from omnistream.media.metrics import frames_dropped

__all__ = [
    "CHUNK_HEADER_SIZE",
    "CHUNK_MAGIC",
    "ChunkHeader",
    "DataChannelCodec",
    "Reassembler",
    "ReassemblyStats",
    "fragment",
]

_log = get_logger(__name__)

CHUNK_MAGIC = b"OMSC"
_CHUNK_STRUCT = struct.Struct("<4sIHH")
CHUNK_HEADER_SIZE = _CHUNK_STRUCT.size
#: uint16 的分片序号上限。
MAX_CHUNKS = 0xFFFF


@dataclass(slots=True, frozen=True)
class ChunkHeader:
    """一个分片的头部信息。"""

    msg_id: int
    index: int
    total: int

    @property
    def is_single(self) -> bool:
        """是否为单分片消息。"""
        return self.total == 1

    def pack(self) -> bytes:
        """序列化为 12 字节头。"""
        return _CHUNK_STRUCT.pack(CHUNK_MAGIC, self.msg_id, self.index, self.total)

    @classmethod
    def unpack(cls, raw: bytes) -> "ChunkHeader":
        """从字节反序列化。

        Args:
            raw: 至少 12 字节的数据。

        Returns:
            解析出的头部。

        Raises:
            CodecError: 长度不足或 magic 非法。
        """
        if len(raw) < CHUNK_HEADER_SIZE:
            raise CodecError(
                "datachannel chunk too short", detail={"nbytes": len(raw)}
            )
        magic, msg_id, index, total = _CHUNK_STRUCT.unpack_from(raw, 0)
        if magic != CHUNK_MAGIC:
            raise CodecError("datachannel chunk magic mismatch")
        if total == 0 or index >= total:
            raise CodecError(
                "datachannel chunk index out of range",
                detail={"index": index, "total": total},
            )
        return cls(msg_id=msg_id, index=index, total=total)


def fragment(payload: bytes, *, msg_id: int, mtu: int = DATACHANNEL_MTU) -> list[bytes]:
    """把一条消息切成带头部的分片列表。

    Args:
        payload: 原始消息字节。
        msg_id: 消息 ID，重组端据此聚合。
        mtu: 单条 DataChannel 消息的字节上限（含头部）。

    Returns:
        分片列表，至少一个元素（空消息也会产生一个空负载分片）。

    Raises:
        CodecError: mtu 太小或消息大到超过分片序号上限。
    """
    body_size = max(1, int(mtu) - CHUNK_HEADER_SIZE)
    if body_size <= 0:
        raise CodecError("datachannel mtu too small", detail={"mtu": mtu})
    if not payload:
        return [ChunkHeader(msg_id=msg_id, index=0, total=1).pack()]
    total = (len(payload) + body_size - 1) // body_size
    if total > MAX_CHUNKS:
        raise CodecError(
            "message requires too many datachannel chunks",
            detail={"nbytes": len(payload), "total": total},
        )
    chunks: list[bytes] = []
    for index in range(total):
        start = index * body_size
        body = payload[start : start + body_size]
        chunks.append(ChunkHeader(msg_id=msg_id, index=index, total=total).pack() + body)
    return chunks


@dataclass(slots=True)
class ReassemblyStats:
    """重组统计。"""

    messages_completed: int = 0
    messages_expired: int = 0
    messages_evicted: int = 0
    chunks_received: int = 0
    chunks_duplicated: int = 0
    chunks_invalid: int = 0
    bytes_completed: int = 0

    def as_dict(self) -> dict[str, float]:
        """转为可序列化字典。"""
        return {
            "messages_completed": float(self.messages_completed),
            "messages_expired": float(self.messages_expired),
            "messages_evicted": float(self.messages_evicted),
            "chunks_received": float(self.chunks_received),
            "chunks_duplicated": float(self.chunks_duplicated),
            "chunks_invalid": float(self.chunks_invalid),
            "bytes_completed": float(self.bytes_completed),
        }


@dataclass(slots=True)
class _Pending:
    """一条待重组消息。"""

    total: int
    created_ms: int
    parts: dict[int, bytes] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """是否已集齐全部分片。"""
        return len(self.parts) >= self.total

    def assemble(self) -> bytes:
        """按序号拼接成完整消息。"""
        return b"".join(self.parts[i] for i in range(self.total))


class Reassembler:
    """分片重组器（单条链路独占，非线程安全）。"""

    def __init__(
        self,
        *,
        timeout_ms: int = DATACHANNEL_REASSEMBLY_TIMEOUT_MS,
        max_pending: int = DATACHANNEL_MAX_PENDING_MESSAGES,
    ) -> None:
        """
        Args:
            timeout_ms: 半成品消息的存活时间。
            max_pending: 同时在途的半成品消息上限。
        """
        self.timeout_ms = max(100, int(timeout_ms))
        self.max_pending = max(1, int(max_pending))
        self._pending: dict[int, _Pending] = {}
        self.stats = ReassemblyStats()

    @property
    def pending(self) -> int:
        """当前半成品消息数。"""
        return len(self._pending)

    def push(self, chunk: bytes) -> bytes | None:
        """喂入一个分片。

        Args:
            chunk: 带头部的分片字节。

        Returns:
            集齐时返回完整消息，否则返回 ``None``。
        """
        try:
            header = ChunkHeader.unpack(chunk)
        except CodecError:
            self.stats.chunks_invalid += 1
            return None
        self.stats.chunks_received += 1
        body = chunk[CHUNK_HEADER_SIZE:]

        if header.is_single:
            self.stats.messages_completed += 1
            self.stats.bytes_completed += len(body)
            self._pending.pop(header.msg_id, None)
            return body

        self.purge_expired()
        entry = self._pending.get(header.msg_id)
        if entry is None:
            self._evict_if_needed()
            entry = _Pending(total=header.total, created_ms=now_ms())
            self._pending[header.msg_id] = entry
        if entry.total != header.total:
            self.stats.chunks_invalid += 1
            return None
        if header.index in entry.parts:
            self.stats.chunks_duplicated += 1
            return None
        entry.parts[header.index] = body
        if not entry.complete:
            return None
        self._pending.pop(header.msg_id, None)
        message = entry.assemble()
        self.stats.messages_completed += 1
        self.stats.bytes_completed += len(message)
        return message

    def _evict_if_needed(self) -> None:
        """超过上限时淘汰最旧的半成品。"""
        while len(self._pending) >= self.max_pending:
            oldest = min(self._pending, key=lambda key: self._pending[key].created_ms)
            self._pending.pop(oldest, None)
            self.stats.messages_evicted += 1
            try:
                frames_dropped.labels(
                    media="datachannel", stage="reassembly", reason="evicted"
                ).inc()
            except Exception:
                pass

    def purge_expired(self) -> int:
        """清理超时的半成品消息。

        Returns:
            被清理的消息数。
        """
        if not self._pending:
            return 0
        deadline = now_ms() - self.timeout_ms
        expired = [key for key, item in self._pending.items() if item.created_ms < deadline]
        for key in expired:
            self._pending.pop(key, None)
            self.stats.messages_expired += 1
            try:
                frames_dropped.labels(
                    media="datachannel", stage="reassembly", reason="timeout"
                ).inc()
            except Exception:
                pass
        if expired:
            _log.debug("datachannel_reassembly_expired", count=len(expired))
        return len(expired)

    def clear(self) -> int:
        """丢弃全部半成品。

        Returns:
            被丢弃的消息数。
        """
        count = len(self._pending)
        self._pending.clear()
        return count

    def snapshot(self) -> dict[str, float]:
        """导出统计快照。"""
        data = self.stats.as_dict()
        data["pending"] = float(len(self._pending))
        return data

    def __repr__(self) -> str:
        """调试用文本表示。"""
        return f"Reassembler(pending={len(self._pending)}, completed={self.stats.messages_completed})"


class DataChannelCodec:
    """DataChannel 上的消息编解码器：发送端分片，接收端重组。"""

    def __init__(
        self,
        *,
        mtu: int = DATACHANNEL_MTU,
        timeout_ms: int = DATACHANNEL_REASSEMBLY_TIMEOUT_MS,
        max_pending: int = DATACHANNEL_MAX_PENDING_MESSAGES,
    ) -> None:
        """
        Args:
            mtu: 单条消息字节上限。
            timeout_ms: 重组超时。
            max_pending: 在途半成品上限。
        """
        self.mtu = max(CHUNK_HEADER_SIZE + 1, int(mtu))
        self.reassembler = Reassembler(timeout_ms=timeout_ms, max_pending=max_pending)
        self._next_msg_id = 0
        self.messages_sent = 0
        self.chunks_sent = 0

    def encode(self, payload: bytes) -> list[bytes]:
        """把一条消息编码为待发送的分片列表。

        Args:
            payload: 原始消息。

        Returns:
            分片列表。
        """
        msg_id = self._next_msg_id
        self._next_msg_id = (self._next_msg_id + 1) & 0xFFFFFFFF
        chunks = fragment(payload, msg_id=msg_id, mtu=self.mtu)
        self.messages_sent += 1
        self.chunks_sent += len(chunks)
        return chunks

    def decode(self, chunk: bytes) -> bytes | None:
        """喂入一个收到的分片，集齐时返回完整消息。"""
        return self.reassembler.push(chunk)

    def snapshot(self) -> dict[str, float]:
        """导出编解码统计。"""
        data = self.reassembler.snapshot()
        data["messages_sent"] = float(self.messages_sent)
        data["chunks_sent"] = float(self.chunks_sent)
        data["mtu"] = float(self.mtu)
        return data

    def __repr__(self) -> str:
        """调试用文本表示。"""
        return f"DataChannelCodec(mtu={self.mtu}, sent={self.messages_sent})"

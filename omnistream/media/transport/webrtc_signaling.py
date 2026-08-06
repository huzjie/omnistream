"""WebRTC 信令：SDP offer/answer 状态机、SDP 微调与 trickle ICE 暂存。

职责边界
--------
本模块**不碰 aiortc**，只做纯文本与状态管理，因此在没装 aiortc 的环境里
``POST /v1/rtc/offer`` 依然能给出结构化的 501 而不是 500。

SDP 微调（munging）为什么必要
-----------------------------
浏览器给出的 offer 往往把 Opus 的 ``maxaveragebitrate`` 留空、把 ``ptime`` 留在 20ms 之外，
也可能把视频排在音频前面。实时对话场景对**音频**最敏感，所以这里统一：

* 把 ``opus`` 提到音频 m 行的第一位；
* 补 ``useinbandfec=1``（丢包时用带内 FEC 而不是等重传）与 ``maxaveragebitrate``；
* 需要时整段剥掉视频 m 行（纯语音会话省一半协商开销）。
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from omnistream.core.clock import now_ms
from omnistream.core.ids import new_ulid
from omnistream.core.logging import get_logger
from omnistream.media.constants import AUDIO_BITRATE_DEFAULT_BPS, DEFAULT_FRAME_MS
from omnistream.media.errors import SignalingError
from omnistream.media.transport.ice import IceConfig, summarize_candidates

__all__ = [
    "SessionDescription",
    "SignalingRegistry",
    "SignalingSession",
    "SignalingState",
    "prefer_codec",
    "strip_video",
    "tune_opus",
    "validate_sdp",
]

_log = get_logger(__name__)

#: 信令会话默认存活时间（毫秒）。协商完成后转由传输层接管。
DEFAULT_SIGNALING_TTL_MS = 120_000
#: 单会话最多暂存多少条 trickle 候选，防止恶意刷爆内存。
MAX_PENDING_CANDIDATES = 256

_M_LINE_RE = re.compile(r"^m=(?P<media>\w+)\s+(?P<port>\d+)\s+(?P<proto>\S+)\s+(?P<fmts>.*)$")


class SignalingState(StrEnum):
    """信令状态机。"""

    NEW = "new"
    OFFER_RECEIVED = "offer_received"
    ANSWER_SENT = "answer_sent"
    CONNECTED = "connected"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(slots=True, frozen=True)
class SessionDescription:
    """一份 SDP（与 WebRTC ``RTCSessionDescription`` 同形）。"""

    type: str
    sdp: str

    @property
    def is_offer(self) -> bool:
        """是否是 offer。"""
        return self.type == "offer"

    @property
    def has_video(self) -> bool:
        """是否包含视频 m 行。"""
        return any(line.startswith("m=video") for line in self.sdp.splitlines())

    @property
    def has_audio(self) -> bool:
        """是否包含音频 m 行。"""
        return any(line.startswith("m=audio") for line in self.sdp.splitlines())

    def as_dict(self) -> dict[str, str]:
        """导出为 ``{"type": ..., "sdp": ...}``。"""
        return {"type": self.type, "sdp": self.sdp}


def validate_sdp(sdp: str, *, expect_type: str = "offer") -> None:
    """做基本的 SDP 合法性检查。

    Args:
        sdp: SDP 文本。
        expect_type: 期望的描述类型，仅用于错误信息。

    Raises:
        SignalingError: SDP 为空、缺 ``v=0`` 或没有任何 m 行。
    """
    if not sdp or not sdp.strip():
        raise SignalingError(f"{expect_type} sdp is empty")
    lines = sdp.splitlines()
    if not any(line.startswith("v=") for line in lines):
        raise SignalingError(f"{expect_type} sdp missing version line")
    if not any(line.startswith("m=") for line in lines):
        raise SignalingError(f"{expect_type} sdp has no media section")
    if len(sdp) > 64 * 1024:
        raise SignalingError("sdp too large", detail={"nbytes": len(sdp)})


def _payload_types_for(sdp_lines: list[str], codec: str) -> list[str]:
    """找出某个编解码器对应的所有 payload type。"""
    pattern = re.compile(rf"^a=rtpmap:(\d+)\s+{re.escape(codec)}/", re.IGNORECASE)
    found: list[str] = []
    for line in sdp_lines:
        match = pattern.match(line)
        if match is not None:
            found.append(match.group(1))
    return found


def prefer_codec(sdp: str, codec: str = "opus", *, media: str = "audio") -> str:
    """把指定编解码器提到对应 m 行的第一位。

    Args:
        sdp: 原始 SDP。
        codec: 编解码器名（如 ``opus`` / ``H264``）。
        media: 作用于哪个 m 行（``audio`` / ``video``）。

    Returns:
        调整后的 SDP；找不到对应 codec 时原样返回。
    """
    lines = sdp.splitlines()
    payloads = _payload_types_for(lines, codec)
    if not payloads:
        return sdp
    output: list[str] = []
    for line in lines:
        match = _M_LINE_RE.match(line)
        if match is None or match.group("media") != media:
            output.append(line)
            continue
        formats = match.group("fmts").split()
        ordered = [pt for pt in payloads if pt in formats]
        rest = [pt for pt in formats if pt not in ordered]
        output.append(
            f"m={match.group('media')} {match.group('port')} "
            f"{match.group('proto')} {' '.join(ordered + rest)}"
        )
    return "\r\n".join(output) + "\r\n"


def tune_opus(
    sdp: str,
    *,
    bitrate_bps: int = AUDIO_BITRATE_DEFAULT_BPS,
    frame_ms: int = DEFAULT_FRAME_MS,
    inband_fec: bool = True,
    dtx: bool = False,
) -> str:
    """为 Opus 补齐实时对话场景的 fmtp 参数。

    Args:
        sdp: 原始 SDP。
        bitrate_bps: 目标平均码率。
        frame_ms: 期望帧长，写入 ``ptime`` / ``maxptime``。
        inband_fec: 是否开启带内 FEC。
        dtx: 是否开启不连续传输（省流量但会削掉舒适噪声）。

    Returns:
        调整后的 SDP。
    """
    lines = sdp.splitlines()
    payloads = set(_payload_types_for(lines, "opus"))
    if not payloads:
        return sdp
    wanted = {
        "minptime": str(min(frame_ms, 10)),
        "useinbandfec": "1" if inband_fec else "0",
        "usedtx": "1" if dtx else "0",
        "maxaveragebitrate": str(int(bitrate_bps)),
        "stereo": "0",
    }
    output: list[str] = []
    patched: set[str] = set()
    for line in lines:
        fmtp = re.match(r"^a=fmtp:(\d+)\s+(.*)$", line)
        if fmtp is not None and fmtp.group(1) in payloads:
            params: dict[str, str] = {}
            for item in fmtp.group(2).split(";"):
                if "=" in item:
                    key, _, value = item.partition("=")
                    params[key.strip()] = value.strip()
            params.update(wanted)
            body = ";".join(f"{key}={value}" for key, value in params.items())
            output.append(f"a=fmtp:{fmtp.group(1)} {body}")
            patched.add(fmtp.group(1))
            continue
        output.append(line)
        rtpmap = re.match(r"^a=rtpmap:(\d+)\s+opus/", line, re.IGNORECASE)
        if rtpmap is not None:
            output.append(f"a=ptime:{frame_ms}")
            output.append(f"a=maxptime:{max(frame_ms, 60)}")
    for payload in sorted(payloads - patched):
        body = ";".join(f"{key}={value}" for key, value in wanted.items())
        output.append(f"a=fmtp:{payload} {body}")
    return "\r\n".join(output) + "\r\n"


def strip_video(sdp: str) -> str:
    """剥掉全部视频 m 行（纯语音会话用）。

    Args:
        sdp: 原始 SDP。

    Returns:
        只保留非视频段的 SDP。
    """
    output: list[str] = []
    skipping = False
    for line in sdp.splitlines():
        if line.startswith("m="):
            skipping = line.startswith("m=video")
        if skipping:
            continue
        output.append(line)
    return "\r\n".join(output) + "\r\n"


@dataclass(slots=True)
class SignalingSession:
    """一次 offer/answer 协商的全部状态。"""

    session_id: str
    handle: str = field(default_factory=new_ulid)
    state: SignalingState = SignalingState.NEW
    offer: SessionDescription | None = None
    answer: SessionDescription | None = None
    ice_config: IceConfig | None = None
    created_ms: int = field(default_factory=now_ms)
    updated_ms: int = field(default_factory=now_ms)
    ttl_ms: int = DEFAULT_SIGNALING_TTL_MS
    error: str = ""
    pending_candidates: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=MAX_PENDING_CANDIDATES)
    )

    @property
    def expired(self) -> bool:
        """是否已超过存活时间。"""
        return now_ms() - self.updated_ms > self.ttl_ms

    @property
    def age_ms(self) -> int:
        """创建至今的毫秒数。"""
        return now_ms() - self.created_ms

    def touch(self) -> None:
        """刷新最后活跃时间。"""
        self.updated_ms = now_ms()

    def set_offer(self, sdp: str, *, type_: str = "offer") -> SessionDescription:
        """记录远端 offer。

        Args:
            sdp: SDP 文本。
            type_: 描述类型。

        Returns:
            记录下来的描述。

        Raises:
            SignalingError: 状态不允许或 SDP 非法。
        """
        if self.state not in (SignalingState.NEW, SignalingState.OFFER_RECEIVED):
            raise SignalingError(
                f"cannot accept offer in state {self.state}",
                detail={"session_id": self.session_id, "state": str(self.state)},
            )
        validate_sdp(sdp, expect_type="offer")
        self.offer = SessionDescription(type=type_, sdp=sdp)
        self.state = SignalingState.OFFER_RECEIVED
        self.touch()
        return self.offer

    def set_answer(self, sdp: str, *, type_: str = "answer") -> SessionDescription:
        """记录本端 answer。

        Args:
            sdp: SDP 文本。
            type_: 描述类型。

        Returns:
            记录下来的描述。

        Raises:
            SignalingError: 尚未收到 offer 或 SDP 非法。
        """
        if self.state is not SignalingState.OFFER_RECEIVED:
            raise SignalingError(
                f"cannot send answer in state {self.state}",
                detail={"session_id": self.session_id, "state": str(self.state)},
            )
        validate_sdp(sdp, expect_type="answer")
        self.answer = SessionDescription(type=type_, sdp=sdp)
        self.state = SignalingState.ANSWER_SENT
        self.touch()
        return self.answer

    def add_candidate(self, candidate: dict[str, Any]) -> int:
        """暂存一条 trickle ICE 候选。

        Args:
            candidate: ``{"candidate": ..., "sdpMid": ..., "sdpMLineIndex": ...}``。

        Returns:
            暂存队列长度。
        """
        self.pending_candidates.append(dict(candidate))
        self.touch()
        return len(self.pending_candidates)

    def drain_candidates(self) -> list[dict[str, Any]]:
        """取走全部暂存候选。"""
        items = list(self.pending_candidates)
        self.pending_candidates.clear()
        return items

    def mark_connected(self) -> None:
        """标记协商成功。"""
        self.state = SignalingState.CONNECTED
        self.touch()

    def mark_failed(self, error: str) -> None:
        """标记协商失败。"""
        self.state = SignalingState.FAILED
        self.error = error
        self.touch()

    def close(self) -> None:
        """关闭信令会话。"""
        self.state = SignalingState.CLOSED
        self.pending_candidates.clear()
        self.touch()

    def as_dict(self) -> dict[str, Any]:
        """导出状态快照（不含完整 SDP，避免日志爆炸）。"""
        return {
            "session_id": self.session_id,
            "handle": self.handle,
            "state": str(self.state),
            "age_ms": self.age_ms,
            "has_offer": self.offer is not None,
            "has_answer": self.answer is not None,
            "pending_candidates": len(self.pending_candidates),
            "offer_candidates": (
                summarize_candidates(self.offer.sdp) if self.offer is not None else {}
            ),
            "error": self.error,
        }


class SignalingRegistry:
    """信令会话表：按会话 ID 索引，带 TTL 自动回收。"""

    def __init__(self, *, ttl_ms: int = DEFAULT_SIGNALING_TTL_MS, max_sessions: int = 4096) -> None:
        """
        Args:
            ttl_ms: 会话存活时间。
            max_sessions: 会话数上限，超出时淘汰最旧的。
        """
        self.ttl_ms = max(1_000, int(ttl_ms))
        self.max_sessions = max(1, int(max_sessions))
        self._sessions: dict[str, SignalingSession] = {}

    def __len__(self) -> int:
        """当前会话数。"""
        return len(self._sessions)

    def create(self, session_id: str, *, ice_config: IceConfig | None = None) -> SignalingSession:
        """创建（或重建）一个信令会话。

        Args:
            session_id: 会话 ID。
            ice_config: 本次下发的 ICE 配置。

        Returns:
            新建的信令会话。
        """
        self.purge_expired()
        self._evict_if_needed()
        session = SignalingSession(
            session_id=session_id, ice_config=ice_config, ttl_ms=self.ttl_ms
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> SignalingSession | None:
        """按会话 ID 查询，过期条目视为不存在。"""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.expired:
            self._sessions.pop(session_id, None)
            return None
        return session

    def require(self, session_id: str) -> SignalingSession:
        """按会话 ID 查询，不存在则抛异常。

        Raises:
            SignalingError: 会话不存在或已过期。
        """
        session = self.get(session_id)
        if session is None:
            raise SignalingError(
                "signaling session not found", detail={"session_id": session_id}
            )
        return session

    def remove(self, session_id: str) -> bool:
        """移除一个会话。

        Returns:
            是否确实移除了条目。
        """
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.close()
        return True

    def purge_expired(self) -> int:
        """清理过期会话。

        Returns:
            被清理的会话数。
        """
        expired = [key for key, item in self._sessions.items() if item.expired]
        for key in expired:
            self._sessions.pop(key, None)
        if expired:
            _log.debug("signaling_sessions_expired", count=len(expired))
        return len(expired)

    def _evict_if_needed(self) -> None:
        """超过上限时淘汰最旧的会话。"""
        while len(self._sessions) >= self.max_sessions:
            oldest = min(self._sessions, key=lambda key: self._sessions[key].updated_ms)
            self._sessions.pop(oldest, None)

    def snapshot(self) -> dict[str, Any]:
        """导出注册表快照。"""
        by_state: dict[str, int] = {}
        for session in self._sessions.values():
            key = str(session.state)
            by_state[key] = by_state.get(key, 0) + 1
        return {
            "sessions": len(self._sessions),
            "by_state": by_state,
            "ttl_ms": self.ttl_ms,
        }

    def __repr__(self) -> str:
        """调试用文本表示。"""
        return f"SignalingRegistry(sessions={len(self._sessions)}, ttl_ms={self.ttl_ms})"


#: 进程级默认注册表，路由层直接用它，避免每个请求都新建。
SIGNALING_REGISTRY = SignalingRegistry()

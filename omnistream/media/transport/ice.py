"""ICE / STUN / TURN 配置与凭据签发。

TURN 长期凭据（coturn ``use-auth-secret`` 模式）
-----------------------------------------------
不把静态账号密码发给浏览器，而是用共享密钥现算一次性凭据（REST API 约定）::

    username   = "<expiry_unix_ts>:<user_tag>"
    credential = base64(HMAC-SHA1(shared_secret, username))

coturn 侧只需配置 ``static-auth-secret``，无需维护用户表；凭据到期即失效，
泄露一份也只影响 :data:`~omnistream.media.constants.ICE_CREDENTIAL_TTL_S` 这段时间。

本模块**零第三方依赖**（只用标准库 hmac/hashlib/base64），
即便 aiortc 没装，服务端照样能给前端下发 ICE 配置。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import re
from dataclasses import dataclass, field
from typing import Any

from omnistream.core.clock import now_ms
from omnistream.core.logging import get_logger
from omnistream.media.constants import (
    DEFAULT_STUN_URLS,
    ICE_CREDENTIAL_TTL_S,
    ICE_GATHER_TIMEOUT_MS,
    media_setting,
)
from omnistream.media.errors import IceFailed

__all__ = [
    "CandidateInfo",
    "IceConfig",
    "IceServer",
    "build_ice_servers",
    "ice_config_from_settings",
    "parse_candidate",
    "summarize_candidates",
    "turn_credentials",
]

_log = get_logger(__name__)

#: ``a=candidate:...`` 行的解析正则（RFC 5245 §15.1 的必备字段）。
_CANDIDATE_RE = re.compile(
    r"candidate:(?P<foundation>\S+)\s+(?P<component>\d+)\s+(?P<protocol>\S+)\s+"
    r"(?P<priority>\d+)\s+(?P<ip>\S+)\s+(?P<port>\d+)\s+typ\s+(?P<type>\S+)"
)

#: 候选类型优先级排序，越靠前越"本地"。
_TYPE_ORDER: tuple[str, ...] = ("host", "srflx", "prflx", "relay")


@dataclass(slots=True, frozen=True)
class IceServer:
    """一个 ICE server 条目，形状与 WebRTC ``RTCIceServer`` 一致。"""

    urls: tuple[str, ...]
    username: str | None = None
    credential: str | None = None
    credential_type: str = "password"

    @property
    def is_turn(self) -> bool:
        """是否是 TURN 中继。"""
        return any(url.startswith(("turn:", "turns:")) for url in self.urls)

    def as_dict(self) -> dict[str, Any]:
        """导出为前端可直接使用的字典（``None`` 字段省略）。"""
        data: dict[str, Any] = {"urls": list(self.urls)}
        if self.username:
            data["username"] = self.username
        if self.credential:
            data["credential"] = self.credential
            data["credentialType"] = self.credential_type
        return data


@dataclass(slots=True)
class IceConfig:
    """下发给客户端的完整 ICE 配置。"""

    ice_servers: list[IceServer] = field(default_factory=list)
    ice_transport_policy: str = "all"
    bundle_policy: str = "balanced"
    gather_timeout_ms: int = ICE_GATHER_TIMEOUT_MS
    ttl_s: int = ICE_CREDENTIAL_TTL_S
    issued_ms: int = field(default_factory=now_ms)

    @property
    def has_turn(self) -> bool:
        """配置中是否包含 TURN。没有 TURN 时对称 NAT 下会连不通。"""
        return any(server.is_turn for server in self.ice_servers)

    @property
    def expires_ms(self) -> int:
        """凭据过期时刻（毫秒）。"""
        return self.issued_ms + self.ttl_s * 1000

    def as_dict(self) -> dict[str, Any]:
        """导出为 ``RTCConfiguration`` 形状的字典。"""
        return {
            "iceServers": [server.as_dict() for server in self.ice_servers],
            "iceTransportPolicy": self.ice_transport_policy,
            "bundlePolicy": self.bundle_policy,
            "gatherTimeoutMs": self.gather_timeout_ms,
            "expiresMs": self.expires_ms,
            "hasTurn": self.has_turn,
        }


def turn_credentials(
    shared_secret: str,
    user_tag: str,
    *,
    ttl_s: int = ICE_CREDENTIAL_TTL_S,
    now_s: int | None = None,
) -> tuple[str, str]:
    """按 coturn REST API 约定生成一次性 TURN 凭据。

    Args:
        shared_secret: coturn 的 ``static-auth-secret``。
        user_tag: 用户标识（建议用会话 ID 或租户 ID，会明文出现在 username 里）。
        ttl_s: 凭据有效期（秒）。
        now_s: 当前 UNIX 秒，``None`` 表示取系统时间（便于测试注入）。

    Returns:
        ``(username, credential)``。

    Raises:
        IceFailed: 共享密钥为空。
    """
    if not shared_secret:
        raise IceFailed("turn shared secret is empty")
    base_s = now_s if now_s is not None else now_ms() // 1000
    expiry = int(base_s) + max(60, int(ttl_s))
    username = f"{expiry}:{user_tag}" if user_tag else str(expiry)
    digest = hmac.new(
        shared_secret.encode("utf-8"), username.encode("utf-8"), hashlib.sha1
    ).digest()
    credential = base64.b64encode(digest).decode("ascii")
    return username, credential


def build_ice_servers(
    *,
    stun_urls: list[str] | tuple[str, ...] | None = None,
    turn_urls: list[str] | tuple[str, ...] | None = None,
    turn_username: str | None = None,
    turn_password: str | None = None,
    turn_shared_secret: str | None = None,
    user_tag: str = "",
    ttl_s: int = ICE_CREDENTIAL_TTL_S,
) -> list[IceServer]:
    """组装 ICE server 列表。

    TURN 凭据的优先级：``turn_shared_secret``（动态签发） > 静态账号密码。

    Args:
        stun_urls: STUN 地址列表，``None`` 时使用内置默认值。
        turn_urls: TURN 地址列表。
        turn_username: 静态 TURN 用户名。
        turn_password: 静态 TURN 密码。
        turn_shared_secret: coturn 共享密钥，存在时动态签发凭据。
        user_tag: 动态凭据里的用户标识。
        ttl_s: 动态凭据有效期。

    Returns:
        ICE server 列表，至少包含 STUN。
    """
    servers: list[IceServer] = []
    stun = tuple(stun_urls) if stun_urls else tuple(DEFAULT_STUN_URLS)
    if stun:
        servers.append(IceServer(urls=stun))
    turn = tuple(turn_urls) if turn_urls else ()
    if not turn:
        return servers
    if turn_shared_secret:
        username, credential = turn_credentials(
            turn_shared_secret, user_tag, ttl_s=ttl_s
        )
        servers.append(IceServer(urls=turn, username=username, credential=credential))
    elif turn_username and turn_password:
        servers.append(
            IceServer(urls=turn, username=turn_username, credential=turn_password)
        )
    else:
        _log.warning("turn_configured_without_credentials", urls=list(turn))
        servers.append(IceServer(urls=turn))
    return servers


def ice_config_from_settings(
    *,
    user_tag: str = "",
    force_relay: bool = False,
) -> IceConfig:
    """从应用配置构造 ICE 配置（配置缺失时安全退化为纯 STUN）。

    读取的配置项（全部走 :func:`~omnistream.media.constants.media_setting`，
    缺失即用默认值）：``stun_servers`` / ``turn_url`` / ``turn_username`` /
    ``turn_password`` / ``turn_shared_secret`` / ``turn_ttl_s``。

    Args:
        user_tag: 写入动态 TURN 凭据的用户标识。
        force_relay: 是否强制走中继（``iceTransportPolicy=relay``），排障用。

    Returns:
        可直接下发给前端的 ICE 配置。
    """
    raw_stun = media_setting("stun_servers", list(DEFAULT_STUN_URLS))
    stun_urls = [str(url) for url in raw_stun] if isinstance(raw_stun, (list, tuple)) else []
    turn_url = media_setting("turn_url", "")
    turn_urls = [str(turn_url)] if turn_url else []
    ttl_s = int(media_setting("turn_ttl_s", ICE_CREDENTIAL_TTL_S))
    servers = build_ice_servers(
        stun_urls=stun_urls,
        turn_urls=turn_urls,
        turn_username=str(media_setting("turn_username", "") or "") or None,
        turn_password=str(media_setting("turn_password", "") or "") or None,
        turn_shared_secret=str(media_setting("turn_shared_secret", "") or "") or None,
        user_tag=user_tag,
        ttl_s=ttl_s,
    )
    return IceConfig(
        ice_servers=servers,
        ice_transport_policy="relay" if force_relay else "all",
        gather_timeout_ms=int(media_setting("ice_gather_timeout_ms", ICE_GATHER_TIMEOUT_MS)),
        ttl_s=ttl_s,
    )


@dataclass(slots=True, frozen=True)
class CandidateInfo:
    """解析后的 ICE 候选。"""

    foundation: str
    component: int
    protocol: str
    priority: int
    ip: str
    port: int
    type: str

    @property
    def is_relay(self) -> bool:
        """是否是 TURN 中继候选。"""
        return self.type == "relay"

    @property
    def is_ipv6(self) -> bool:
        """是否是 IPv6 地址。"""
        return ":" in self.ip

    def as_dict(self) -> dict[str, Any]:
        """导出为字典。"""
        return {
            "foundation": self.foundation,
            "component": self.component,
            "protocol": self.protocol,
            "priority": self.priority,
            "ip": self.ip,
            "port": self.port,
            "type": self.type,
        }


def parse_candidate(line: str) -> CandidateInfo | None:
    """解析一行 ``a=candidate:``（或裸 ``candidate:``）。

    Args:
        line: SDP 候选行。

    Returns:
        解析结果；不是合法候选行时返回 ``None``。
    """
    text = line.strip()
    if text.startswith("a="):
        text = text[2:]
    match = _CANDIDATE_RE.search(text)
    if match is None:
        return None
    try:
        return CandidateInfo(
            foundation=match.group("foundation"),
            component=int(match.group("component")),
            protocol=match.group("protocol").lower(),
            priority=int(match.group("priority")),
            ip=match.group("ip"),
            port=int(match.group("port")),
            type=match.group("type").lower(),
        )
    except (TypeError, ValueError):
        return None


def summarize_candidates(sdp: str) -> dict[str, Any]:
    """统计一段 SDP 里的候选分布，用于连通性排障。

    Args:
        sdp: 完整 SDP 文本。

    Returns:
        含各类型计数、是否有中继候选、协议分布的字典。
    """
    counts: dict[str, int] = {name: 0 for name in _TYPE_ORDER}
    protocols: dict[str, int] = {}
    total = 0
    for line in sdp.splitlines():
        info = parse_candidate(line)
        if info is None:
            continue
        total += 1
        counts[info.type] = counts.get(info.type, 0) + 1
        protocols[info.protocol] = protocols.get(info.protocol, 0) + 1
    return {
        "total": total,
        "by_type": counts,
        "by_protocol": protocols,
        "has_relay": counts.get("relay", 0) > 0,
        "has_host": counts.get("host", 0) > 0,
    }


def ensure_connectable(sdp: str, *, require_relay: bool = False) -> None:
    """在协商前做一次"这条 SDP 有可能连通吗"的体检。

    Args:
        sdp: 远端 SDP。
        require_relay: 是否强制要求存在中继候选。

    Raises:
        IceFailed: 没有任何候选，或要求中继却没有中继候选。
    """
    summary = summarize_candidates(sdp)
    if summary["total"] == 0:
        # trickle ICE 下 offer 里可以没有候选，这里只在明确要求时才拦
        if require_relay:
            raise IceFailed("remote sdp carries no ice candidate", detail=summary)
        return
    if require_relay and not summary["has_relay"]:
        raise IceFailed("relay-only policy requires a relay candidate", detail=summary)

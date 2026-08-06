"""ID 生成与解析。

全部为 **ULID**（26 字符 Crockford Base32，字典序即时间序），带类型前缀，
数据库主键统一 ``String(32)`` 存放（含前缀）。前缀表见架构文档 §7.5。
"""
from __future__ import annotations

import secrets
from typing import Final

from ulid import ULID

from omnistream.core.constants import API_KEY_PLAINTEXT_PREFIX

__all__ = [
    "ID_PREFIXES",
    "ULID_LENGTH",
    "extract_ulid",
    "has_prefix",
    "new_api_key_id",
    "new_api_key_plaintext",
    "new_event_id",
    "new_id",
    "new_memory_id",
    "new_recording_id",
    "new_session_id",
    "new_tenant_id",
    "new_tool_call_id",
    "new_turn_id",
    "new_ulid",
    "new_user_id",
    "split_id",
    "timestamp_ms_of",
    "validate_id",
]

ULID_LENGTH: Final[int] = 26
_CROCKFORD: Final[str] = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

#: 类型 -> 前缀。新增实体请在此登记，并同步 docs/architecture.md。
ID_PREFIXES: Final[dict[str, str]] = {
    "session": "sess_",
    "turn": "turn_",
    "event": "evt_",
    "tenant": "ten_",
    "user": "usr_",
    "api_key": "ak_",
    "memory": "mem_",
    "tool_call": "call_",
    "recording": "rec_",
    "transcript": "tr_",
    "vision": "vis_",
    "usage": "use_",
    "audit": "aud_",
    "quality": "ql_",
    "webhook": "wh_",
    "request": "req_",
}


def new_ulid() -> str:
    """裸 ULID（26 字符，无前缀）。"""
    return str(ULID())


def new_id(prefix: str = "") -> str:
    """生成带前缀的 ULID。

    ``prefix`` 既可以传登记名（如 ``"session"``）也可以直接传前缀字面量
    （如 ``"sess_"``）；传空串则返回裸 ULID。
    """
    if not prefix:
        return new_ulid()
    resolved = ID_PREFIXES.get(prefix, prefix)
    if resolved and not resolved.endswith("_"):
        resolved = f"{resolved}_"
    return f"{resolved}{new_ulid()}"


def new_session_id() -> str:
    return new_id("session")


def new_turn_id() -> str:
    return new_id("turn")


def new_event_id() -> str:
    return new_id("event")


def new_tenant_id() -> str:
    return new_id("tenant")


def new_user_id() -> str:
    return new_id("user")


def new_api_key_id() -> str:
    return new_id("api_key")


def new_memory_id() -> str:
    return new_id("memory")


def new_tool_call_id() -> str:
    return new_id("tool_call")


def new_recording_id() -> str:
    return new_id("recording")


def new_api_key_plaintext(nbytes: int = 24) -> str:
    """签发给用户的明文 API Key：``oms_live_<43 字符 url-safe 随机>``。

    只在创建时返回一次，服务端仅保存 argon2/blake2b 哈希。
    """
    return f"{API_KEY_PLAINTEXT_PREFIX}{secrets.token_urlsafe(nbytes)}"


def split_id(value: str) -> tuple[str, str]:
    """拆成 ``(prefix, ulid)``。无前缀时 prefix 为空串。"""
    if "_" in value:
        head, _, tail = value.rpartition("_")
        return f"{head}_", tail
    return "", value


def extract_ulid(value: str) -> str:
    """取出 ID 中的 ULID 部分。"""
    return split_id(value)[1]


def has_prefix(value: str, kind: str) -> bool:
    """判断 ID 是否属于某个登记类型。"""
    expected = ID_PREFIXES.get(kind, kind)
    if expected and not expected.endswith("_"):
        expected = f"{expected}_"
    return value.startswith(expected)


def validate_id(value: str, kind: str | None = None) -> bool:
    """校验 ID 结构是否合法（前缀正确 + ULID 部分是合法 Crockford Base32）。"""
    if not value or not isinstance(value, str):
        return False
    if kind is not None and not has_prefix(value, kind):
        return False
    body = extract_ulid(value)
    if len(body) != ULID_LENGTH:
        return False
    return all(ch in _CROCKFORD for ch in body.upper())


def timestamp_ms_of(value: str) -> int:
    """从 ID 中还原生成时刻（UTC 毫秒）。非法 ID 抛 ``ValueError``。"""
    body = extract_ulid(value)
    return int(ULID.from_str(body).timestamp * 1000)

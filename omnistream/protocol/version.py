"""协议版本与兼容性检查。

版本形如 ``oms.v<major>``。同一 major 内**只允许新增可选字段与新增事件类型**，
不允许删除字段、改字段语义或改事件名 —— 客户端遇到未知事件类型必须忽略而非报错。
"""
from __future__ import annotations

from typing import Final

from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import ProtocolError
from omnistream.protocol.enums import PROTOCOL_VERSION

__all__ = [
    "PROTOCOL_MAJOR",
    "PROTOCOL_VERSION",
    "SUPPORTED_VERSIONS",
    "assert_compatible",
    "is_compatible",
    "negotiate",
    "parse_version",
]

#: 当前服务端实现的协议版本。
PROTOCOL_MAJOR: Final[int] = 1

#: 服务端可接受的全部协议版本（含历史版本）。
SUPPORTED_VERSIONS: Final[tuple[str, ...]] = (PROTOCOL_VERSION,)


def parse_version(value: str) -> int:
    """把 ``oms.v1`` 解析成 major 号 ``1``。非法格式抛 :class:`ProtocolError`。"""
    text = (value or "").strip().lower()
    if not text.startswith("oms.v"):
        raise ProtocolError(
            f"协议版本格式非法: {value!r}，应形如 'oms.v1'",
            code=ErrorCode.PROTOCOL_VERSION_MISMATCH,
            detail={"got": value, "expected_format": "oms.v<major>"},
        )
    suffix = text[len("oms.v"):]
    if not suffix.isdigit():
        raise ProtocolError(
            f"协议版本号不是数字: {value!r}",
            code=ErrorCode.PROTOCOL_VERSION_MISMATCH,
            detail={"got": value},
        )
    return int(suffix)


def is_compatible(client_version: str) -> bool:
    """客户端版本是否与本服务端兼容（同 major 即兼容）。"""
    try:
        return parse_version(client_version) == PROTOCOL_MAJOR
    except ProtocolError:
        return False


def assert_compatible(client_version: str) -> None:
    """不兼容时抛 :class:`ProtocolError`（``OMS-3000``）。"""
    if not is_compatible(client_version):
        raise ProtocolError(
            f"协议版本不匹配: 服务端 {PROTOCOL_VERSION}, 客户端 {client_version}",
            code=ErrorCode.PROTOCOL_VERSION_MISMATCH,
            detail={
                "server": PROTOCOL_VERSION,
                "client": client_version,
                "supported": list(SUPPORTED_VERSIONS),
            },
        )


def negotiate(client_versions: list[str] | tuple[str, ...] | None) -> str:
    """从客户端提供的候选版本里挑一个双方都支持的。

    客户端未声明时按当前版本处理（宽松兼容旧 SDK）。全不匹配则抛异常。
    """
    if not client_versions:
        return PROTOCOL_VERSION
    for candidate in client_versions:
        if candidate in SUPPORTED_VERSIONS:
            return candidate
    raise ProtocolError(
        f"没有共同支持的协议版本，客户端候选: {list(client_versions)}",
        code=ErrorCode.PROTOCOL_VERSION_MISMATCH,
        detail={"client_candidates": list(client_versions), "supported": list(SUPPORTED_VERSIONS)},
    )

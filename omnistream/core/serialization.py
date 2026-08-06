"""序列化：orjson 为主，msgpack 可选，Pydantic 走快路径。

统一在这里处理"orjson 不认识的类型"（Decimal、set、numpy 标量、Pydantic 模型、
dataclass），避免每个调用点各写一个 ``default=``。
"""
from __future__ import annotations

import dataclasses
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any, Final
from uuid import UUID

import orjson
from pydantic import BaseModel

from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import ValidationError

__all__ = [
    "MSGPACK_AVAILABLE",
    "dumps",
    "dumps_str",
    "loads",
    "msgpack_dumps",
    "msgpack_loads",
    "to_jsonable",
]

try:  # msgpack 属于可选加速路径，缺失时自动退回 JSON。
    import msgpack

    MSGPACK_AVAILABLE: Final[bool] = True
except ImportError:  # pragma: no cover - 取决于部署裁剪
    msgpack = None  # type: ignore[assignment]
    MSGPACK_AVAILABLE = False

_ORJSON_OPTIONS: Final[int] = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY


def _default(obj: Any) -> Any:
    """orjson 的兜底转换器。"""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json", exclude_none=True)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        import base64

        return base64.b64encode(bytes(obj)).decode("ascii")
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if hasattr(obj, "item") and callable(obj.item):  # numpy 标量
        return obj.item()
    if hasattr(obj, "tolist") and callable(obj.tolist):  # numpy 数组
        return obj.tolist()
    raise TypeError(f"无法序列化类型 {type(obj).__name__}")


def dumps(obj: Any, *, sort_keys: bool = False, indent: bool = False) -> bytes:
    """序列化为 UTF-8 JSON 字节。Pydantic 模型走 ``model_dump_json`` 快路径。"""
    if isinstance(obj, BaseModel) and not sort_keys and not indent:
        return obj.model_dump_json(exclude_none=True).encode("utf-8")
    option = _ORJSON_OPTIONS
    if sort_keys:
        option |= orjson.OPT_SORT_KEYS
    if indent:
        option |= orjson.OPT_INDENT_2
    return orjson.dumps(obj, default=_default, option=option)


def dumps_str(obj: Any, *, sort_keys: bool = False, indent: bool = False) -> str:
    """序列化为 ``str``（写文件/日志时用）。"""
    return dumps(obj, sort_keys=sort_keys, indent=indent).decode("utf-8")


def loads(raw: str | bytes | bytearray) -> Any:
    """反序列化 JSON。失败抛 :class:`ValidationError`（``OMS-1001``）。"""
    try:
        return orjson.loads(raw)
    except orjson.JSONDecodeError as exc:
        raise ValidationError(
            f"JSON 解析失败: {exc}", code=ErrorCode.INVALID_REQUEST, cause=exc
        ) from exc


def to_jsonable(obj: Any) -> Any:
    """把任意对象转成"纯 JSON 可表达"的 Python 结构（dict/list/str/int/float/bool/None）。"""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json", exclude_none=True)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    return loads(dumps(obj))


def msgpack_dumps(obj: Any) -> bytes:
    """msgpack 序列化（不可用时自动退回 JSON 字节）。"""
    if not MSGPACK_AVAILABLE or msgpack is None:
        return dumps(obj)
    return bytes(msgpack.packb(to_jsonable(obj), use_bin_type=True))


def msgpack_loads(raw: bytes) -> Any:
    """msgpack 反序列化（不可用时按 JSON 解析）。"""
    if not MSGPACK_AVAILABLE or msgpack is None:
        return loads(raw)
    try:
        return msgpack.unpackb(raw, raw=False)
    except Exception as exc:
        raise ValidationError(
            f"msgpack 解析失败: {exc}", code=ErrorCode.INVALID_REQUEST, cause=exc
        ) from exc

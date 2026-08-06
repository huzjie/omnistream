"""跨方言自定义列类型。

设计目标：同一份 ORM 模型既能跑 PostgreSQL(+pgvector) 也能跑 SQLite，
让 ``OMS_MOCK_MODE=true`` 时零外部依赖即可 ``alembic upgrade head``。
"""
from __future__ import annotations

import json
import math
import os
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import DateTime, LargeBinary, String, Text
from sqlalchemy import JSON as SA_JSON
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator

#: 记忆向量默认维度（与 hash_embedder / MiniLM 家族对齐）
DEFAULT_VECTOR_DIM = 384

#: ULID Crockford Base32 字母表
_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _encode_ulid(timestamp_ms: int, randomness: bytes) -> str:
    """把 48bit 时间 + 80bit 随机编码为 26 字符 Crockford Base32。"""
    value = (timestamp_ms & ((1 << 48) - 1)) << 80
    value |= int.from_bytes(randomness, "big") & ((1 << 80) - 1)
    chars = []
    for shift in range(125, -1, -5):
        chars.append(_ULID_ALPHABET[(value >> shift) & 0x1F])
    return "".join(chars)


def _local_ulid() -> str:
    return _encode_ulid(int(time.time() * 1000), secrets.token_bytes(10))


try:  # pragma: no cover - 取决于 WS1 是否已合入
    from omnistream.core.ids import new_id as _core_new_id  # type: ignore
except Exception:  # noqa: BLE001
    _core_new_id = None  # type: ignore[assignment]


def generate_id(prefix: str) -> str:
    """生成带前缀的 ULID（优先复用 ``omnistream.core.ids.new_id``）。

    §7.5 规定所有主键为 ``<prefix><26 字符 ULID>``，字典序即时间序。
    """
    if _core_new_id is not None:
        try:
            value = _core_new_id(prefix)
            if isinstance(value, str) and value:
                return value
        except TypeError:
            try:
                value = _core_new_id()  # type: ignore[call-arg]
                if isinstance(value, str) and value:
                    return f"{prefix}{value}"
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass
    return f"{prefix}{_local_ulid()}"


def utc_now() -> datetime:
    """tz-aware UTC 当前时间（优先复用 ``omnistream.core.clock.utcnow``）。"""
    try:
        from omnistream.core.clock import utcnow  # type: ignore

        value = utcnow()
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001
        pass
    return datetime.now(timezone.utc)


def ensure_utc(value: datetime | None) -> datetime | None:
    """把任意 datetime 归一化为 tz-aware UTC。"""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class ULIDType(TypeDecorator[str]):
    """ULID 主键列：固定 ``String(32)``，全大写不做转换。"""

    impl = String(32)
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Dialect) -> str | None:
        if value is None:
            return None
        return str(value)

    def process_result_value(self, value: Any, dialect: Dialect) -> str | None:
        if value is None:
            return None
        return str(value)


class JSONB(TypeDecorator[Any]):
    """PostgreSQL 用原生 JSONB，其余方言退化为通用 JSON（SQLite 存 TEXT）。"""

    impl = SA_JSON
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB

            return dialect.type_descriptor(PG_JSONB(astext_type=Text()))
        return dialect.type_descriptor(SA_JSON())


class UTCDateTime(TypeDecorator[datetime]):
    """强制 UTC 的 datetime 列。

    SQLite 不存时区，写入前转 UTC 并去掉 tzinfo，读出后补回 UTC，
    保证 §7.6 "后端只吐 UTC tz-aware" 的约定在两种方言下行为一致。
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise TypeError(f"UTCDateTime 只接受 datetime, 收到 {type(value)!r}")
        normalized = ensure_utc(value)
        assert normalized is not None
        if dialect.name == "sqlite":
            return normalized.replace(tzinfo=None)
        return normalized

    def process_result_value(self, value: Any, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return ensure_utc(value)


class VectorType(TypeDecorator[Any]):
    """嵌入向量列。

    * PostgreSQL 且安装了 ``pgvector`` → 使用原生 ``vector(dim)``，可建 ivfflat 索引；
    * 其余情况 → 以 JSON 数组文本存储，检索时在 Python 侧算余弦相似度。
    """

    impl = Text
    cache_ok = True

    def __init__(self, dim: int = DEFAULT_VECTOR_DIM) -> None:
        super().__init__()
        self.dim = int(dim)

    @property
    def python_type(self) -> type:
        return list

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if dialect.name == "postgresql":
            try:
                from pgvector.sqlalchemy import Vector as PGVector

                return dialect.type_descriptor(PGVector(self.dim))
            except Exception:  # noqa: BLE001 - 未装 pgvector 时降级为文本
                return dialect.type_descriptor(Text())
        return dialect.type_descriptor(Text())

    def _is_native(self, dialect: Dialect) -> bool:
        if dialect.name != "postgresql":
            return False
        try:
            import pgvector.sqlalchemy  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        vector = [float(x) for x in value]
        if len(vector) != self.dim:
            raise ValueError(f"向量维度不匹配: 期望 {self.dim}, 收到 {len(vector)}")
        if self._is_native(dialect):
            return vector
        return json.dumps(vector, separators=(",", ":"))

    def process_result_value(self, value: Any, dialect: Dialect) -> list[float] | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("[") and text.endswith("]"):
                try:
                    return [float(x) for x in json.loads(text)]
                except (ValueError, TypeError):
                    return None
            return None
        try:
            return [float(x) for x in value]
        except TypeError:
            return None


class EncryptedBytes(TypeDecorator[bytes]):
    """加密二进制列（明文由上层负责加解密，此处只保证方言兼容）。"""

    impl = LargeBinary
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Dialect) -> bytes | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value.encode("utf-8")
        return bytes(value)

    def process_result_value(self, value: Any, dialect: Dialect) -> bytes | None:
        if value is None:
            return None
        return bytes(value)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度，SQLite 路径下的向量检索由它兜底。"""
    if not a or not b:
        return 0.0
    size = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(size):
        x = float(a[i])
        y = float(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def env_flag(name: str, default: bool = False) -> bool:
    """基础设施级环境开关（仅用于 db/引导层，业务代码禁止使用）。"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


__all__ = [
    "DEFAULT_VECTOR_DIM",
    "EncryptedBytes",
    "JSONB",
    "ULIDType",
    "UTCDateTime",
    "VectorType",
    "cosine_similarity",
    "ensure_utc",
    "env_flag",
    "generate_id",
    "utc_now",
]

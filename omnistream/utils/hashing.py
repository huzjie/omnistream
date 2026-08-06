"""哈希与内容指纹。

全平台统一用 **BLAKE2b**（比 SHA-256 快、原生支持 keyed hashing 与可变摘要长度），
只有需要与外部系统对齐时才用 SHA-256。

三类用途严格区分，不要混用：

* :func:`content_id`   —— 内容寻址（录音片段、图片帧、工具结果缓存 key）；
* :func:`stable_hash`  —— 结构化对象的确定性指纹（记忆去重、幂等键）；
* :func:`keyed_digest` —— 带密钥的 MAC（API Key 校验、审计链），**绝不**用裸哈希。
"""
from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "DIGEST_SIZE",
    "blake2b_hex",
    "constant_time_equals",
    "content_id",
    "fingerprint_bits",
    "keyed_digest",
    "merkle_root",
    "sha256_hex",
    "short_hash",
    "stable_hash",
    "stable_json",
]

#: 默认摘要长度（字节）。32 字节 = 64 hex 字符，抗碰撞强度足够且不至于太长。
DIGEST_SIZE: int = 32
_SHORT_LEN: int = 12


def _as_bytes(value: str | bytes | bytearray | memoryview) -> bytes:
    """把常见输入统一成 ``bytes``（字符串按 UTF-8 编码）。"""
    if isinstance(value, str):
        return value.encode("utf-8")
    return bytes(value)


def blake2b_hex(data: str | bytes, *, digest_size: int = DIGEST_SIZE) -> str:
    """BLAKE2b 十六进制摘要。"""
    return hashlib.blake2b(_as_bytes(data), digest_size=digest_size).hexdigest()


def sha256_hex(data: str | bytes) -> str:
    """SHA-256 十六进制摘要（仅在需要与外部系统互操作时使用）。"""
    return hashlib.sha256(_as_bytes(data)).hexdigest()


def short_hash(data: str | bytes, length: int = _SHORT_LEN) -> str:
    """短指纹，用于日志与 UI 展示（默认 12 hex 字符 ≈ 48 bit）。"""
    size = max(4, (length + 1) // 2)
    return hashlib.blake2b(_as_bytes(data), digest_size=size).hexdigest()[:length]


def stable_json(obj: Any) -> str:
    """把任意可 JSON 化对象序列化成**确定性**字符串。

    与 ``json.dumps`` 的区别：键排序、无多余空格、非 ASCII 不转义、
    非 JSON 原生类型（datetime、Enum、set…）统一走 ``str()`` / ``sorted()``。
    这保证同一份语义数据在任何机器、任何 Python 版本下得到同一个字符串。
    """

    def _default(value: Any) -> Any:
        if isinstance(value, (set, frozenset)):
            return sorted(str(v) for v in value)
        if isinstance(value, bytes):
            return value.hex()
        if isinstance(value, Mapping):
            return dict(value)
        to_dict = getattr(value, "model_dump", None)
        if callable(to_dict):
            return to_dict(mode="json")
        return str(value)

    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=_default)


def stable_hash(obj: Any, *, digest_size: int = DIGEST_SIZE) -> str:
    """结构化对象的确定性指纹（先 :func:`stable_json` 再 BLAKE2b）。"""
    return blake2b_hex(stable_json(obj), digest_size=digest_size)


def content_id(data: bytes, *, prefix: str = "cnt") -> str:
    """内容寻址 ID，形如 ``cnt_9f2a...``。相同字节永远得到相同 ID。"""
    return f"{prefix}_{hashlib.blake2b(data, digest_size=16).hexdigest()}"


def keyed_digest(key: str | bytes, message: str | bytes, *,
                 digest_size: int = DIGEST_SIZE) -> str:
    """带密钥的 BLAKE2b MAC。用于 API Key 存储与审计哈希链。

    Args:
        key: 密钥（服务端 salt / secret），长度上限 64 字节，超出自动先摘要。
        message: 待认证的消息。
        digest_size: 输出摘要字节数。

    Returns:
        十六进制 MAC 字符串。
    """
    key_bytes = _as_bytes(key)
    if len(key_bytes) > 64:
        key_bytes = hashlib.blake2b(key_bytes, digest_size=64).digest()
    return hashlib.blake2b(_as_bytes(message), key=key_bytes,
                           digest_size=digest_size).hexdigest()


def constant_time_equals(a: str | bytes, b: str | bytes) -> bool:
    """恒定时间比较，防时序侧信道。校验任何密钥/令牌都必须用它。"""
    return hmac.compare_digest(_as_bytes(a), _as_bytes(b))


def fingerprint_bits(data: str | bytes, bits: int = 64) -> int:
    """把内容映射成一个 ``bits`` 位整数，用于 SimHash / 布隆过滤器等场景。"""
    if bits <= 0:
        raise ValueError("bits must be positive")
    n_bytes = (bits + 7) // 8
    digest = hashlib.blake2b(_as_bytes(data), digest_size=max(1, n_bytes)).digest()
    value = int.from_bytes(digest, "big")
    return value & ((1 << bits) - 1)


def merkle_root(leaves: Iterable[str | bytes]) -> str:
    """计算一组叶子的 Merkle 根（BLAKE2b，奇数节点复制末位）。

    审计日志按小时切段封存时，用根哈希做一次性防篡改证明。
    空输入返回全零摘要。
    """
    level: list[bytes] = [hashlib.blake2b(_as_bytes(x), digest_size=DIGEST_SIZE).digest()
                          for x in leaves]
    if not level:
        return "0" * (DIGEST_SIZE * 2)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.blake2b(level[i] + level[i + 1], digest_size=DIGEST_SIZE).digest()
            for i in range(0, len(level), 2)
        ]
    return level[0].hex()

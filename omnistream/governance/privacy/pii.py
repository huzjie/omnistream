"""PII 检测：手机号 / 邮箱 / 身份证 / 银行卡 / 地址。

基于正则 + 简单上下文规则，零第三方依赖即可运行。命中信息返回
:class:`~omnistream.schemas.audit.PIIMatch`，原文以 blake2b 短哈希记录（便于去重统计，
又不落明文）。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from omnistream.schemas.audit import PIIMatch

#: 默认启用的 PII 类型
DEFAULT_KINDS: frozenset[str] = frozenset(
    {"phone", "email", "id_card", "bank_card", "address"}
)

_PATTERNS: dict[str, re.Pattern[str]] = {
    "phone": re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)"),
    "email": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "id_card": re.compile(
        r"(?<!\d)([1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx])(?!\d)"
    ),
    "bank_card": re.compile(r"(?<!\d)([3-9]\d{3}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4})(?!\d)"),
    "address": re.compile(r"[一-龥]{2,}(?:省|市|区|县|路|街|号|栋|单元|室|大厦|广场)"),
}

_MASK = "***"


def _short_hash(value: str) -> str:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest()


@dataclass(slots=True)
class PIIDetector:
    """PII 检测器。"""

    kinds: set[str] = field(default_factory=lambda: set(DEFAULT_KINDS))

    def detect(self, text: str | None) -> list[PIIMatch]:
        """返回文本中所有 PII 命中（按出现位置排序）。"""
        if not text:
            return []
        out: list[PIIMatch] = []
        for kind in self.kinds:
            pattern = _PATTERNS.get(kind)
            if pattern is None:
                continue
            for m in pattern.finditer(text):
                original = m.group(0)
                out.append(
                    PIIMatch(
                        kind=kind,
                        start=m.start(),
                        end=m.end(),
                        original_hash=_short_hash(original),
                        replacement=_MASK,
                    )
                )
        out.sort(key=lambda item: item.start)
        return out

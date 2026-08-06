"""输入校验器。

Pydantic 负责「结构」校验，本模块负责「语义」校验：采样率是否受支持、
ID 前缀是否匹配、URL 是否可安全出网（SSRF 防护）等等。

所有函数遵循同一约定：**失败即抛** :class:`~omnistream.core.errors.ValidationError`
（带 ``OMS-1001``），成功则返回**归一化后**的值 —— 因此可以直接写
``rate = validate_sample_rate(raw)``。
"""
from __future__ import annotations

import ipaddress
import re
from typing import Any
from urllib.parse import urlparse

from omnistream.core.constants import (
    MAX_VIDEO_FPS,
    SUPPORTED_SAMPLE_RATES,
)
from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import ValidationError

__all__ = [
    "EMAIL_PATTERN",
    "SLUG_PATTERN",
    "ULID_PATTERN",
    "ensure_range",
    "is_private_host",
    "validate_channels",
    "validate_email",
    "validate_fps",
    "validate_frame_ms",
    "validate_id",
    "validate_language",
    "validate_sample_rate",
    "validate_slug",
    "validate_url",
    "validate_webhook_url",
]

EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9\-]{1,62}[a-z0-9]$")
ULID_PATTERN = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")
_LANG_PATTERN = re.compile(r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$")
#: 帧长必须能被采样率整除且落在实时可用区间内。
_ALLOWED_FRAME_MS: frozenset[int] = frozenset({10, 20, 30, 40, 60, 80, 100})
_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https", "ws", "wss"})


def _fail(message: str, **detail: Any) -> ValidationError:
    return ValidationError(message, code=ErrorCode.INVALID_REQUEST, detail=detail)


# --------------------------------------------------------------------------- 数值
def ensure_range(value: float, *, low: float, high: float, name: str = "value") -> float:
    """确保 ``low <= value <= high``，否则抛错。返回原值便于链式使用。"""
    if not low <= value <= high:
        raise _fail(f"{name} out of range", value=value, min=low, max=high)
    return value


def validate_sample_rate(rate: int) -> int:
    """校验采样率是否在 :data:`SUPPORTED_SAMPLE_RATES` 内。"""
    if rate not in SUPPORTED_SAMPLE_RATES:
        raise _fail("unsupported sample rate", sample_rate=rate,
                    supported=list(SUPPORTED_SAMPLE_RATES))
    return rate


def validate_channels(channels: int) -> int:
    """校验声道数。平台内部统一单声道，最多接受立体声输入后下混。"""
    if channels not in (1, 2):
        raise _fail("unsupported channel count", channels=channels, supported=[1, 2])
    return channels


def validate_frame_ms(frame_ms: int, sample_rate: int | None = None) -> int:
    """校验帧长（毫秒），并确保它能在给定采样率下整除成整数个采样点。"""
    if frame_ms not in _ALLOWED_FRAME_MS:
        raise _fail("unsupported frame duration", frame_ms=frame_ms,
                    supported=sorted(_ALLOWED_FRAME_MS))
    if sample_rate is not None and (sample_rate * frame_ms) % 1000:
        raise _fail("frame duration does not divide evenly at this sample rate",
                    frame_ms=frame_ms, sample_rate=sample_rate)
    return frame_ms


def validate_fps(fps: float) -> float:
    """校验视频采样帧率 ``(0, MAX_VIDEO_FPS]``。"""
    if not 0 < fps <= MAX_VIDEO_FPS:
        raise _fail("fps out of range", fps=fps, max=MAX_VIDEO_FPS)
    return float(fps)


# --------------------------------------------------------------------------- 标识
def validate_id(value: str, *, prefix: str = "", name: str = "id") -> str:
    """校验平台 ID：可选前缀 + ULID 主体（如 ``ses_01J...``）。"""
    raw = value.strip()
    if not raw:
        raise _fail(f"{name} must not be empty")
    body = raw
    if prefix:
        if not raw.startswith(prefix):
            raise _fail(f"{name} has wrong prefix", value=raw, expected_prefix=prefix)
        body = raw[len(prefix):]
    if not ULID_PATTERN.match(body.upper()):
        raise _fail(f"{name} is not a valid ULID", value=raw)
    return raw


def validate_slug(value: str, *, name: str = "slug") -> str:
    """校验租户/项目 slug：小写字母数字与连字符，3~64 字符，不以连字符收尾。"""
    normalized = value.strip().lower()
    if not SLUG_PATTERN.match(normalized):
        raise _fail(f"invalid {name}", value=value,
                    rule="^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
    if "--" in normalized:
        raise _fail(f"invalid {name}: consecutive hyphens", value=value)
    return normalized


def validate_email(value: str) -> str:
    """校验并归一化邮箱（域名部分小写）。"""
    raw = value.strip()
    if not EMAIL_PATTERN.match(raw) or len(raw) > 254:
        raise _fail("invalid email address", value=raw)
    local, _, domain = raw.partition("@")
    return f"{local}@{domain.lower()}"


def validate_language(value: str) -> str:
    """校验 BCP-47 语言标签（``zh``、``zh-CN``、``en-US``…），归一化为 ``xx-YY``。"""
    raw = value.strip()
    if not _LANG_PATTERN.match(raw):
        raise _fail("invalid language tag", value=value, example="zh-CN")
    parts = raw.split("-")
    head = parts[0].lower()
    tail = [p.upper() if len(p) == 2 else p.capitalize() for p in parts[1:]]
    return "-".join([head, *tail])


# --------------------------------------------------------------------------- 网络
def is_private_host(host: str) -> bool:
    """判断主机是否指向内网/环回/链路本地地址（SSRF 判定的核心）。"""
    lowered = host.strip().lower().strip("[]")
    if lowered in {"localhost", "metadata", "metadata.google.internal"} or lowered.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(lowered)
    except ValueError:
        return False
    return bool(
        address.is_private or address.is_loopback or address.is_link_local
        or address.is_reserved or address.is_multicast or address.is_unspecified
    )


def validate_url(value: str, *, schemes: frozenset[str] = _ALLOWED_SCHEMES) -> str:
    """校验 URL 的 scheme 与 host 是否合法，返回去空白后的原串。"""
    raw = value.strip()
    parsed = urlparse(raw)
    if parsed.scheme.lower() not in schemes:
        raise _fail("unsupported url scheme", url=raw, allowed=sorted(schemes))
    if not parsed.hostname:
        raise _fail("url has no host", url=raw)
    return raw


def validate_webhook_url(value: str, *, allow_private: bool = False) -> str:
    """校验对外回调 URL，默认拒绝内网地址以防 SSRF。

    Args:
        value: 待校验 URL。
        allow_private: 私有部署内网互调场景下显式放开。
    """
    raw = validate_url(value, schemes=frozenset({"http", "https"}))
    host = urlparse(raw).hostname or ""
    if not allow_private and is_private_host(host):
        raise _fail("webhook url must not point to a private address",
                    url=raw, host=host)
    return raw

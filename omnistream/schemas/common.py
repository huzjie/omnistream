"""通用响应封装：分页、成功包、错误体、时间范围、游标。"""
from __future__ import annotations

import base64
import binascii
from datetime import datetime, timedelta
from typing import Any, Generic, TypeVar

import orjson
from pydantic import Field, field_validator, model_validator

from omnistream.core.clock import utcnow
from omnistream.core.constants import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT
from omnistream.core.error_codes import ErrorCode
from omnistream.schemas.base import OmniModel

T = TypeVar("T")


class ErrorBody(OmniModel):
    """所有 4xx/5xx 响应的统一体，与 WS `error` 事件 payload 同构。"""

    code: ErrorCode = ErrorCode.INTERNAL
    message: str = "内部错误"
    detail: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None
    session_id: str | None = None
    retryable: bool = False
    ts: datetime = Field(default_factory=utcnow)


class Ok(OmniModel, Generic[T]):
    """成功响应包。REST 接口统一返回 ``{"ok": true, "data": ...}``。"""

    ok: bool = True
    data: T | None = None
    request_id: str | None = None

    @classmethod
    def of(cls, data: T, *, request_id: str | None = None) -> "Ok[T]":
        return cls(ok=True, data=data, request_id=request_id)


class Cursor(OmniModel):
    """不透明游标。对外是 base64 字符串，对内是 ``(created_at, id)`` 复合键。"""

    created_at: datetime
    id: str

    def encode(self) -> str:
        raw = orjson.dumps({"t": self.created_at.isoformat(), "i": self.id})
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    @classmethod
    def decode(cls, token: str) -> "Cursor":
        padded = token + "=" * (-len(token) % 4)
        try:
            payload = orjson.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
        except (binascii.Error, orjson.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"非法游标: {token!r}") from exc
        return cls(created_at=datetime.fromisoformat(payload["t"]), id=str(payload["i"]))


class PageRequest(OmniModel):
    """列表接口的通用查询参数。"""

    limit: int = Field(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT)
    cursor: str | None = None
    order: str = Field(default="desc", description="asc|desc")

    @field_validator("order")
    @classmethod
    def _order(cls, v: str) -> str:
        low = v.lower()
        if low not in ("asc", "desc"):
            raise ValueError("order 只能是 asc 或 desc")
        return low

    def decoded_cursor(self) -> Cursor | None:
        return Cursor.decode(self.cursor) if self.cursor else None


class Page(OmniModel, Generic[T]):
    """游标分页结果。"""

    items: list[T] = Field(default_factory=list)
    next_cursor: str | None = None
    has_more: bool = False
    total: int | None = Field(default=None, description="仅在明确请求计数时返回")

    @classmethod
    def of(cls, items: list[T], *, next_cursor: str | None = None,
           total: int | None = None) -> "Page[T]":
        return cls(items=items, next_cursor=next_cursor,
                   has_more=next_cursor is not None, total=total)

    @classmethod
    def empty(cls) -> "Page[T]":
        return cls(items=[], next_cursor=None, has_more=False, total=0)


class TimeRange(OmniModel):
    """闭开区间 ``[start, end)``，用于计量、审计与回放查询。"""

    start: datetime
    end: datetime

    @model_validator(mode="after")
    def _check(self) -> "TimeRange":
        if self.end <= self.start:
            raise ValueError("TimeRange.end 必须晚于 start")
        return self

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    @property
    def duration_ms(self) -> int:
        return int(self.duration.total_seconds() * 1000)

    def contains(self, moment: datetime) -> bool:
        return self.start <= moment < self.end

    @classmethod
    def last_hours(cls, hours: int = 24) -> "TimeRange":
        end = utcnow()
        return cls(start=end - timedelta(hours=hours), end=end)

    @classmethod
    def last_days(cls, days: int = 7) -> "TimeRange":
        end = utcnow()
        return cls(start=end - timedelta(days=days), end=end)


class HealthStatus(OmniModel):
    """``GET /healthz`` 与 ``/readyz`` 的统一返回体。"""

    status: str = Field(default="ok", description="ok|degraded|down")
    version: str = "1.0.0"
    env: str = "dev"
    uptime_s: float = 0.0
    checks: dict[str, bool] = Field(default_factory=dict)
    details: dict[str, str] = Field(default_factory=dict)

"""多维限流：令牌桶 + 滑动窗口。

* **Redis 后端**：用 Lua 脚本保证 `读-算-写` 原子性，多副本共享同一份计数。
* **内存后端**：进程内 ``asyncio.Lock`` 保护，零依赖启动时自动启用。

两种算法：

``token_bucket(key, rate, burst, cost)``
    平滑限速，允许突发。用于 API QPS、WS 消息速率。

``sliding_window(key, limit, window_seconds)``
    精确窗口计数。用于「每分钟最多登录 5 次」这类硬性次数限制。

限流维度由调用方拼 key，统一走 :func:`limit_key` 保证租户命名空间隔离。
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import QuotaError
from omnistream.platform.tenancy import get_redis, namespaced

__all__ = [
    "RateLimitResult",
    "RateLimiter",
    "default_limiter",
    "enforce_rate_limit",
    "limit_key",
]

#: 令牌桶 Lua：KEYS[1]=桶键 ARGV=[rate, burst, cost, now]
_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])
if tokens == nil then
  tokens = burst
  ts = now
end

local delta = now - ts
if delta < 0 then delta = 0 end
tokens = math.min(burst, tokens + delta * rate)

local allowed = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
end

local ttl = 60
if rate > 0 then
  ttl = math.ceil(burst / rate) + 60
end
redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, ttl)

local retry = 0
if allowed == 0 and rate > 0 then
  retry = (cost - tokens) / rate
end
return {allowed, tostring(tokens), tostring(retry)}
"""

#: 滑动窗口 Lua：KEYS[1]=zset 键 ARGV=[limit, window, now, member]
_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local member = ARGV[4]

redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)
if count < limit then
  redis.call('ZADD', key, now, member)
  redis.call('EXPIRE', key, math.ceil(window) + 1)
  return {1, count + 1, 0}
end

local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
local retry = window
if oldest[2] then
  retry = (tonumber(oldest[2]) + window) - now
  if retry < 0 then retry = 0 end
end
redis.call('EXPIRE', key, math.ceil(window) + 1)
return {0, count, tostring(retry)}
"""


def limit_key(tenant_id: str, dimension: str, identifier: str) -> str:
    """构造限流键：``oms:{env}:{tenant}:rl:{dimension}:{identifier}``。"""
    return namespaced(tenant_id or "ten_anonymous", f"rl:{dimension}", identifier or "-")


@dataclass(slots=True)
class RateLimitResult:
    """限流判定结果。"""

    allowed: bool = True
    remaining: float = 0.0
    retry_after: float = 0.0
    limit: float = 0.0
    dimension: str = ""
    key: str = ""

    def to_headers(self) -> dict[str, str]:
        """标准限流响应头。"""
        headers = {
            "X-RateLimit-Limit": str(int(self.limit)),
            "X-RateLimit-Remaining": str(max(0, int(self.remaining))),
        }
        if not self.allowed:
            headers["Retry-After"] = str(max(1, int(round(self.retry_after))))
        return headers

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "remaining": round(self.remaining, 3),
            "retry_after": round(self.retry_after, 3),
            "limit": self.limit,
            "dimension": self.dimension,
        }


class _MemoryBuckets:
    """进程内令牌桶 / 滑动窗口存储。"""

    __slots__ = ("_buckets", "_windows", "_lock")

    def __init__(self) -> None:
        self._buckets: dict[str, tuple[float, float]] = {}   # key -> (tokens, ts)
        self._windows: dict[str, list[float]] = {}           # key -> timestamps
        self._lock = asyncio.Lock()

    async def token_bucket(
        self, key: str, rate: float, burst: float, cost: float, now: float
    ) -> tuple[bool, float, float]:
        async with self._lock:
            tokens, ts = self._buckets.get(key, (burst, now))
            delta = max(0.0, now - ts)
            tokens = min(burst, tokens + delta * rate)
            if tokens >= cost:
                tokens -= cost
                self._buckets[key] = (tokens, now)
                return True, tokens, 0.0
            self._buckets[key] = (tokens, now)
            retry = (cost - tokens) / rate if rate > 0 else 60.0
            return False, tokens, retry

    async def sliding_window(
        self, key: str, limit: int, window: float, now: float
    ) -> tuple[bool, int, float]:
        async with self._lock:
            stamps = [t for t in self._windows.get(key, []) if t > now - window]
            if len(stamps) < limit:
                stamps.append(now)
                self._windows[key] = stamps
                return True, len(stamps), 0.0
            self._windows[key] = stamps
            retry = max(0.0, (stamps[0] + window) - now)
            return False, len(stamps), retry

    async def reset(self, key: str) -> None:
        async with self._lock:
            self._buckets.pop(key, None)
            self._windows.pop(key, None)

    async def peek(self, key: str, burst: float) -> float:
        async with self._lock:
            tokens, _ = self._buckets.get(key, (burst, 0.0))
            return tokens


class RateLimiter:
    """限流器。Redis 可用时自动使用，否则退化进程内。"""

    def __init__(self) -> None:
        self._memory = _MemoryBuckets()
        self._bucket_sha: str = ""
        self._window_sha: str = ""

    async def _script_sha(self, client: Any, source: str, cached: str) -> str:
        if cached:
            return cached
        return str(await client.script_load(source))

    async def token_bucket(
        self,
        key: str,
        *,
        rate: float,
        burst: float | int = 0,
        cost: float = 1.0,
        dimension: str = "",
    ) -> RateLimitResult:
        """令牌桶判定。``rate <= 0`` 表示不限流。"""
        if rate <= 0:
            return RateLimitResult(
                allowed=True, remaining=float("inf"), limit=0.0, dimension=dimension, key=key
            )
        capacity = float(burst) if burst and float(burst) > 0 else max(1.0, rate * 2.0)
        now = time.time()

        client = await get_redis()
        if client is not None:
            try:
                self._bucket_sha = await self._script_sha(
                    client, _TOKEN_BUCKET_LUA, self._bucket_sha
                )
                raw = await client.evalsha(
                    self._bucket_sha, 1, key, str(rate), str(capacity), str(cost), str(now)
                )
                allowed = bool(int(raw[0]))
                tokens = float(raw[1])
                retry = float(raw[2])
                return RateLimitResult(
                    allowed=allowed,
                    remaining=tokens,
                    retry_after=retry,
                    limit=capacity,
                    dimension=dimension,
                    key=key,
                )
            except Exception:  # noqa: BLE001 - Redis 异常时降级，不阻断业务
                self._bucket_sha = ""

        allowed, tokens, retry = await self._memory.token_bucket(
            key, rate, capacity, cost, now
        )
        return RateLimitResult(
            allowed=allowed,
            remaining=tokens,
            retry_after=retry,
            limit=capacity,
            dimension=dimension,
            key=key,
        )

    async def sliding_window(
        self,
        key: str,
        *,
        limit: int,
        window_seconds: float,
        dimension: str = "",
    ) -> RateLimitResult:
        """滑动窗口判定。``limit <= 0`` 表示不限流。"""
        if limit <= 0:
            return RateLimitResult(
                allowed=True, remaining=float("inf"), limit=0.0, dimension=dimension, key=key
            )
        now = time.time()
        member = f"{now:.6f}-{id(self)}"

        client = await get_redis()
        if client is not None:
            try:
                self._window_sha = await self._script_sha(
                    client, _SLIDING_WINDOW_LUA, self._window_sha
                )
                raw = await client.evalsha(
                    self._window_sha, 1, key, str(limit), str(window_seconds), str(now), member
                )
                allowed = bool(int(raw[0]))
                count = int(float(raw[1]))
                retry = float(raw[2])
                return RateLimitResult(
                    allowed=allowed,
                    remaining=max(0, limit - count),
                    retry_after=retry,
                    limit=float(limit),
                    dimension=dimension,
                    key=key,
                )
            except Exception:  # noqa: BLE001
                self._window_sha = ""

        allowed, count, retry = await self._memory.sliding_window(
            key, limit, float(window_seconds), now
        )
        return RateLimitResult(
            allowed=allowed,
            remaining=max(0, limit - count),
            retry_after=retry,
            limit=float(limit),
            dimension=dimension,
            key=key,
        )

    async def check_multi(
        self, checks: Sequence[tuple[str, str, float, float, float]]
    ) -> RateLimitResult:
        """多维度检查，任一维度拒绝即整体拒绝。

        Args:
            checks: ``(dimension, key, rate, burst, cost)`` 序列。

        Returns:
            第一个被拒绝的结果；全部通过时返回剩余额度最少的那个。
        """
        results: list[RateLimitResult] = []
        for dimension, key, rate, burst, cost in checks:
            result = await self.token_bucket(
                key, rate=rate, burst=burst, cost=cost, dimension=dimension
            )
            if not result.allowed:
                return result
            results.append(result)
        if not results:
            return RateLimitResult(allowed=True, remaining=float("inf"))
        return min(results, key=lambda item: item.remaining)

    async def reset(self, key: str) -> None:
        """清空某个键（测试与人工解封用）。"""
        client = await get_redis()
        if client is not None:
            try:
                await client.delete(key)
            except Exception:  # noqa: BLE001
                pass
        await self._memory.reset(key)


#: 全局限流器单例
default_limiter = RateLimiter()


async def enforce_rate_limit(
    tenant_id: str,
    *,
    identifiers: Iterable[tuple[str, str]],
    rate: float,
    burst: int = 0,
    cost: float = 1.0,
    limiter: RateLimiter | None = None,
) -> RateLimitResult:
    """多维限流并在超限时抛 ``QuotaError``。

    Args:
        tenant_id: 租户 ID，决定命名空间。
        identifiers: ``(dimension, identifier)`` 序列，例如
            ``[("tenant", tid), ("api_key", key_id), ("ip", client_ip)]``。
        rate: 每秒补充令牌数。
        burst: 桶容量，``0`` 时取 ``rate * 2``。
        cost: 本次消耗令牌数。
    """
    engine = limiter or default_limiter
    checks = [
        (dimension, limit_key(tenant_id, dimension, identifier), rate, float(burst), cost)
        for dimension, identifier in identifiers
        if identifier
    ]
    result = await engine.check_multi(checks)
    if not result.allowed:
        raise QuotaError(
            "请求过于频繁，请稍后重试",
            code=ErrorCode.RATE_LIMITED,
            detail={
                "dimension": result.dimension,
                "retry_after": round(result.retry_after, 3),
                "limit": result.limit,
            },
        )
    return result

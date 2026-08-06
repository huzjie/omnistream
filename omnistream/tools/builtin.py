"""内置工具集。

提供可直接注册到 ToolRegistry 的常用业务工具：
- 天气查询（mock 实现，可替换真实 API）
- 时间/时区
- HTTP 抓取（可选用）
"""
from __future__ import annotations

import asyncio
from typing import Any


async def tool_get_weather(args: dict[str, Any]) -> dict[str, Any]:
    """天气查询（内置 mock 数据，接入真实 API 时替换本函数）。"""
    city = args.get("city", "unknown")
    _mock_db = {
        "北京": {"temp": 28, "desc": "晴", "humidity": 45},
        "上海": {"temp": 31, "desc": "多云", "humidity": 70},
        "深圳": {"temp": 33, "desc": "雷阵雨", "humidity": 82},
        "广州": {"temp": 32, "desc": "多云", "humidity": 78},
    }
    data = _mock_db.get(city, {"temp": 25, "desc": "未知", "humidity": 50})
    return {"city": city, **data, "source": "mock"}


async def tool_get_time(args: dict[str, Any]) -> dict[str, Any]:
    """获取指定时区的当前时间。"""
    import datetime

    tz_name = args.get("timezone", "Asia/Shanghai")
    now = datetime.datetime.now(datetime.timezone.utc)
    return {"timezone": tz_name, "utc_iso": now.isoformat(),
            "epoch_ms": int(now.timestamp() * 1000)}


async def tool_http_fetch(args: dict[str, Any]) -> dict[str, Any]:
    """抓取 URL 内容（默认最多 32KB）。"""
    import httpx

    url = args.get("url", "")
    max_bytes = int(args.get("max_bytes", 32_768))
    if not url.startswith(("http://", "https://")):
        return {"error": "url must start with http(s)://"}
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url)
            body = resp.text[:max_bytes]
            return {"status": resp.status_code, "body": body}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


BUILTIN_TOOLS: dict[str, Any] = {
    "weather.get": tool_get_weather,
    "system.time": tool_get_time,
    "web.fetch": tool_http_fetch,
}

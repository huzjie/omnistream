"""基于配置的 CORS 装配。

浏览器端 SDK（sdk-ts）与控制台都跨域访问 API，这里统一从 ``Settings`` 取白名单，
并额外放行实时协议需要的自定义响应头。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from omnistream.config.settings import Settings
from omnistream.core.constants import API_KEY_HEADER, REQUEST_ID_HEADER

if TYPE_CHECKING:
    from fastapi import FastAPI

#: 需要暴露给浏览器 JS 读取的响应头
EXPOSE_HEADERS = [REQUEST_ID_HEADER, "X-RateLimit-Remaining", "X-RateLimit-Reset"]

#: 允许客户端携带的请求头
ALLOW_HEADERS = [
    "Authorization", "Content-Type", "Accept", "Origin",
    API_KEY_HEADER, REQUEST_ID_HEADER, "X-Tenant-Id", "Idempotency-Key",
]


def resolve_origins(settings: Settings) -> list[str]:
    """合并根配置与 security 段的白名单并去重保序。"""
    merged: list[str] = []
    for origin in [*settings.cors_origins, *settings.security.cors_origins]:
        cleaned = origin.strip().rstrip("/")
        if cleaned and cleaned not in merged:
            merged.append(cleaned)
    return merged or ["http://localhost:5173"]


def install_cors(app: "FastAPI", settings: Settings) -> None:
    """把 CORS 中间件装配到应用。

    ``*`` 与 ``allow_credentials=True`` 不能共存（浏览器规范），
    此时自动退化为 ``allow_credentials=False``，避免线上出现静默失败。
    """
    from fastapi.middleware.cors import CORSMiddleware

    origins = resolve_origins(settings)
    wildcard = "*" in origins
    allow_credentials = settings.security.cors_origins != ["*"] and not wildcard

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if wildcard else origins,
        allow_credentials=allow_credentials,
        allow_methods=settings.security.cors_allow_methods or ["*"],
        allow_headers=ALLOW_HEADERS if settings.security.cors_allow_headers == ["*"]
        else settings.security.cors_allow_headers,
        expose_headers=EXPOSE_HEADERS,
        max_age=600,
    )

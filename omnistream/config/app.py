"""应用级配置：环境、调试、监听地址、CORS、根路径。

说明：这些字段同时**平铺**在根 ``Settings`` 上（历史契约），
本模块提供一个聚合视图 ``AppSettings``，供需要整体传递的场景使用。
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OMS_APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    env: Literal["dev", "staging", "prod"] = "dev"
    debug: bool = False
    app_name: str = "omnistream"
    #: 展示名，控制台标题与文档站使用
    display_name: str = "OmniStream"
    host: str = "0.0.0.0"
    port: int = Field(default=8080, ge=1, le=65535)
    root_path: str = Field(default="", description="反代子路径部署时填写，如 /omnistream")
    workers: int = Field(default=1, ge=1, description="uvicorn worker 数；容器内建议 1")
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    cors_allow_credentials: bool = True
    docs_enabled: bool = True
    docs_url: str = "/docs"
    openapi_url: str = "/openapi.json"
    #: 优雅退出等待时间，需大于最长一轮对话的收尾时间
    graceful_shutdown_s: float = Field(default=20.0, ge=0.0)
    #: 单进程允许的最大并发会话，超过返回 429
    max_concurrent_sessions: int = Field(default=200, ge=1)
    #: 请求体大小上限（字节），默认 16MB，覆盖单帧图片上传
    max_request_bytes: int = Field(default=16 * 1024 * 1024, ge=1024)

    @property
    def is_prod(self) -> bool:
        return self.env == "prod"

    @property
    def is_dev(self) -> bool:
        return self.env == "dev"

    @property
    def base_url(self) -> str:
        host = "localhost" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}{self.root_path}"

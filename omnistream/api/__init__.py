"""HTTP / WebSocket 接入层。

对外只暴露三样东西：

* :func:`create_app` —— FastAPI 应用工厂；
* ``app``            —— 模块级单例，``uvicorn omnistream.api.app:app`` 用它；
* 依赖注入与错误渲染的公共件（``deps`` / ``errors`` / ``registry``）。

导入本包**不会**立刻构造应用：``create_app`` 与 ``app`` 都通过 :pep:`562`
的模块级 ``__getattr__`` 延迟解析，避免 CLI / worker 只想用 ``omnistream.api.deps``
时被迫拉起整个 FastAPI 实例。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omnistream.api.registry import discover_routers, mount_routers

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

    from omnistream.api.app import app, create_app

__all__ = [
    "API_PREFIX",
    "app",
    "create_app",
    "discover_routers",
    "mount_routers",
]

API_PREFIX = "/v1"

_LAZY_EXPORTS: dict[str, str] = {
    "create_app": "omnistream.api.app",
    "app": "omnistream.api.app",
}


def __getattr__(name: str) -> Any:
    """延迟解析 ``create_app`` / ``app``。"""
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:  # pragma: no cover - defensive
        raise AttributeError(f"module 'omnistream.api' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)

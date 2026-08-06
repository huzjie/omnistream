"""路由自动发现：各 WS 只需往 omnistream/api/routers/ 投放文件，无需修改任何公共文件。

约定：每个 router 模块导出名为 ``router`` 的 ``APIRouter``；
可选导出 ``ROUTER_ORDER: int``（默认 100，小的先挂载）与 ``ROUTER_TAGS: list[str]``。
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from fastapi import APIRouter, FastAPI

log = structlog.get_logger(__name__)
_PACKAGE = "omnistream.api.routers"


def discover_routers() -> list[tuple[int, str, "APIRouter"]]:
    from fastapi import APIRouter

    package = importlib.import_module(_PACKAGE)
    found: list[tuple[int, str, APIRouter]] = []
    for mod_info in pkgutil.iter_modules(package.__path__):
        if mod_info.name.startswith("_"):
            continue
        full = f"{_PACKAGE}.{mod_info.name}"
        try:
            module = importlib.import_module(full)
        except Exception as exc:  # 某个可选模块缺依赖时不应拖垮整个服务
            log.warning("router_import_failed", module=full, error=str(exc))
            continue
        router = getattr(module, "router", None)
        if isinstance(router, APIRouter):
            order = int(getattr(module, "ROUTER_ORDER", 100))
            found.append((order, mod_info.name, router))
        else:
            log.debug("router_missing", module=full)
    found.sort(key=lambda x: (x[0], x[1]))
    return found


def mount_routers(app: "FastAPI", prefix: str = "/v1") -> int:
    count = 0
    for order, name, router in discover_routers():
        app.include_router(router, prefix=prefix if not getattr(router, "_oms_no_prefix", False) else "")
        log.info("router_mounted", name=name, order=order, prefix=prefix)
        count += 1
    return count

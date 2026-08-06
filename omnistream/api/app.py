"""FastAPI 应用工厂。

启动方式（三种等价）::

    uvicorn omnistream.api.app:app --host 0.0.0.0 --port 8080
    python -m omnistream.api.app
    omnistream serve            # CLI（WS «Console & CLI» 提供）

本模块是整个 HTTP 面的组装点，职责边界非常克制：

* **只做装配，不写业务**。任何业务路由都放在 ``omnistream/api/routers/``，
  由 :func:`omnistream.api.registry.mount_routers` 自动发现并挂载 —— 这样六个并行
  工作包各自往 routers/ 投文件即可，永远不会在 ``app.py`` 上产生 merge 冲突；
* **零硬依赖启动**。数据库、Redis、Provider 密钥、本地模型全部缺席时，
  ``mock_mode`` 依然能把服务拉起来并跑通完整链路；
* **可观测的启动过程**。每个生命周期钩子的耗时都会打点，启动失败按 LIFO 回滚。
"""
from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, ORJSONResponse

from omnistream.__version__ import __version__
from omnistream.api.errors import install_error_handlers
from omnistream.api.middleware import install_middlewares
from omnistream.api.registry import mount_routers
from omnistream.config.settings import Settings, get_settings
from omnistream.core.clock import monotonic_s
from omnistream.core.lifecycle import LifecycleManager
from omnistream.core.logging import configure_logging, get_logger

__all__ = ["app", "create_app", "lifespan"]

log = get_logger(__name__)

API_PREFIX = "/v1"
_TITLE = "OmniStream"
_DESCRIPTION = (
    "Enterprise-grade realtime full-duplex omni-modal (audio + video + text) "
    "interaction platform. Self-hostable, multi-provider, observable.\n\n"
    "企业级实时全双工音视频多模态交互平台：可自托管、多 Provider 可插拔、可观测。"
)


# --------------------------------------------------------------------------- 生命周期
def _register_lifecycle_hooks(manager: LifecycleManager, settings: Settings) -> None:
    """把各子系统的启动/关闭钩子登记到生命周期管理器。

    钩子全部使用 **延迟导入**：某个工作包尚未落地或其可选依赖未安装时，
    只降级为 warning，不阻断服务启动（``critical=False``）。
    """

    # ---- 10 provider 注册表：把内置 mock provider 预热进注册表 ----
    async def _warm_providers() -> None:
        from omnistream.providers import _ensure_builtin_registered, available_providers

        _ensure_builtin_registered()
        log.info("providers_ready", available=available_providers(),
                 default=settings.providers.default)

    manager.on_startup("providers", _warm_providers, order=10, timeout_s=5.0, critical=True)

    # ---- 20 数据库：mock_mode / 缺依赖时跳过 ----
    async def _init_db() -> None:
        from omnistream.db.engine import init_engine

        await init_engine(settings.storage)
        log.info("database_ready", url=settings.storage.database_url.split("@")[-1],
                 backend="postgres" if settings.storage.is_postgres else "sqlite")

    async def _close_db() -> None:
        with contextlib.suppress(ImportError):
            from omnistream.db.engine import dispose_engine

            await dispose_engine()

    manager.on_startup("database", _init_db, order=20, timeout_s=15.0, critical=False)
    manager.on_shutdown("database", _close_db, order=20, timeout_s=10.0)

    # ---- 30 事件总线 ----
    async def _start_bus() -> None:
        from omnistream.core.eventbus import get_event_bus

        bus = get_event_bus()
        start = getattr(bus, "start", None)
        if callable(start):
            result = start()
            if hasattr(result, "__await__"):
                await result
        log.info("eventbus_ready", backend=settings.storage.event_bus)

    async def _stop_bus() -> None:
        with contextlib.suppress(ImportError):
            from omnistream.core.eventbus import get_event_bus

            bus = get_event_bus()
            close = getattr(bus, "close", None)
            if callable(close):
                result = close()
                if hasattr(result, "__await__"):
                    await result

    manager.on_startup("eventbus", _start_bus, order=30, timeout_s=5.0, critical=False)
    manager.on_shutdown("eventbus", _stop_bus, order=30, timeout_s=5.0)

    # ---- 40 可观测性（metrics / tracing 导出器） ----
    async def _start_observability() -> None:
        if not settings.observability.tracing_enabled:
            return
        from omnistream.observability.tracing import setup_tracing

        setup_tracing(settings.observability)
        log.info("tracing_ready", endpoint=settings.observability.otlp_endpoint)

    manager.on_startup("observability", _start_observability, order=40,
                       timeout_s=5.0, critical=False)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan：启动时按序拉起子系统，退出时逆序释放。"""
    settings: Settings = app.state.settings
    manager: LifecycleManager = app.state.lifecycle
    began = monotonic_s()

    _register_lifecycle_hooks(manager, settings)
    await manager.startup()

    mounted = mount_routers(app, prefix=API_PREFIX)
    app.state.mounted_routers = mounted
    app.state.ready = True

    log.info(
        "app_started",
        version=__version__,
        env=settings.env,
        mock_mode=settings.mock_mode,
        routers=mounted,
        elapsed_ms=round((monotonic_s() - began) * 1000, 2),
    )
    try:
        yield
    finally:
        app.state.ready = False
        await manager.shutdown()
        log.info("app_stopped", version=__version__)


# --------------------------------------------------------------------------- 内建路由
def _install_builtin_routes(app: FastAPI, settings: Settings) -> None:
    """挂载不属于任何业务域的基础端点：探针、版本、根路径。

    这些端点**不带** ``/v1`` 前缀，因为它们服务于 k8s / LB / 运维脚本，
    生命周期与 API 版本无关。
    """

    @app.get("/", include_in_schema=False)
    async def index() -> dict[str, Any]:
        return {
            "name": _TITLE,
            "version": __version__,
            "docs": "/docs",
            "openapi": "/openapi.json",
            "api_prefix": API_PREFIX,
            "mock_mode": settings.mock_mode,
        }

    @app.get("/healthz", tags=["ops"], summary="Liveness probe")
    async def healthz() -> dict[str, Any]:
        """存活探针：进程还在就返回 200，绝不访问任何外部依赖。"""
        return {"status": "ok", "version": __version__, "env": settings.env}

    @app.get("/readyz", tags=["ops"], summary="Readiness probe")
    async def readyz() -> JSONResponse:
        """就绪探针：生命周期全部启动完成后才返回 200。"""
        manager: LifecycleManager = app.state.lifecycle
        ready = bool(getattr(app.state, "ready", False)) and manager.running
        payload: dict[str, Any] = {
            "status": "ready" if ready else "starting",
            "lifecycle": manager.state,
            "routers": getattr(app.state, "mounted_routers", 0),
        }
        return JSONResponse(payload, status_code=200 if ready else 503)

    @app.get("/version", tags=["ops"], summary="Build/version info")
    async def version() -> dict[str, Any]:
        from omnistream.protocol.version import PROTOCOL_VERSION

        return {
            "version": __version__,
            "protocol_version": PROTOCOL_VERSION,
            "env": settings.env,
            "mock_mode": settings.mock_mode,
            "default_provider": settings.providers.default,
        }


# --------------------------------------------------------------------------- 工厂
def create_app(settings: Settings | None = None) -> FastAPI:
    """构造并返回配置完毕的 FastAPI 应用。

    Args:
        settings: 可选的显式配置。为 ``None`` 时读取进程级缓存配置
            （由环境变量 / ``.env`` 解析而来）。

    Returns:
        已装配中间件、异常处理器、内建探针与 lifespan 的 ``FastAPI`` 实例。
        业务路由在 lifespan 启动阶段自动发现挂载。
    """
    resolved = settings or get_settings()

    configure_logging(
        level=resolved.observability.log_level,
        fmt=resolved.observability.log_format,
        utc=resolved.observability.log_utc,
    )

    application = FastAPI(
        title=_TITLE,
        description=_DESCRIPTION,
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        root_path=resolved.root_path,
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
        contact={"name": "OmniStream", "url": "https://github.com/omnistream/omnistream"},
        license_info={"name": "Apache-2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
        openapi_tags=[
            {"name": "ops", "description": "Health, readiness and version endpoints."},
            {"name": "sessions", "description": "Realtime session lifecycle."},
            {"name": "providers", "description": "Provider catalog, health and routing."},
            {"name": "memory", "description": "Long/short term memory operations."},
            {"name": "governance", "description": "Recording, audit, quality and PII."},
            {"name": "admin", "description": "Tenants, users, API keys and quotas."},
        ],
    )

    application.state.settings = resolved
    application.state.lifecycle = LifecycleManager(
        shutdown_timeout_s=float(resolved.observability.provider_health_timeout_s * 5)
    )
    application.state.ready = False
    application.state.mounted_routers = 0

    order = install_middlewares(application, resolved)
    install_error_handlers(application)
    _install_builtin_routes(application, resolved)

    log.debug("app_created", middlewares=order, env=resolved.env,
              mock_mode=resolved.mock_mode)
    return application


#: 模块级单例，供 ``uvicorn omnistream.api.app:app`` 直接引用。
app: FastAPI = create_app()


def _main() -> None:
    """``python -m omnistream.api.app`` 入口：用当前配置起一个 uvicorn。"""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "omnistream.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_config=None,  # 日志已由 structlog 接管
        access_log=False,  # 由 AccessLogMiddleware 负责
    )


if __name__ == "__main__":  # pragma: no cover - 手工启动路径
    _main()

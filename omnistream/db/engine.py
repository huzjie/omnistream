"""异步引擎构建。

默认 SQLite（``sqlite+aiosqlite``，零配置可跑），生产切 PostgreSQL 只需改
``OMS_STORAGE__DATABASE_URL=postgresql+asyncpg://user:pass@host/db``。

本模块是**基础设施引导层**，允许直接读环境变量作为逃生舱口（§7.2 的例外），
业务代码仍必须走 ``omnistream.config.get_settings()``。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlsplit

import structlog
from sqlalchemy import event, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

log = structlog.get_logger(__name__)

T = TypeVar("T")

_MISSING = object()

#: 默认落盘位置：仓库根的 ./data/omnistream.db
DEFAULT_SQLITE_URL = "sqlite+aiosqlite:///./data/omnistream.db"

_engine: AsyncEngine | None = None
_engine_url: str = ""


def _coerce(raw: str, default: T) -> T:
    """按 default 的类型把环境变量字符串转成对应值。"""
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}  # type: ignore[return-value]
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(raw)  # type: ignore[return-value]
        except ValueError:
            return default
    if isinstance(default, float):
        try:
            return float(raw)  # type: ignore[return-value]
        except ValueError:
            return default
    if isinstance(default, list):
        return [item.strip() for item in raw.split(",") if item.strip()]  # type: ignore[return-value]
    return raw  # type: ignore[return-value]


def setting(path: str, default: T) -> T:
    """按点分路径读取全局配置，缺失时回退默认值。

    读取优先级：环境变量 ``OMS_<PATH_WITH_DOUBLE_UNDERSCORE>`` > Settings 对象 > default。

    这样即使某个配置字段尚未在 ``omnistream/config/*`` 中声明，WS2 的组件依然
    可以被环境变量配置并正常启动，避免跨工作包的时序耦合。
    """
    env_key = "OMS_" + path.replace(".", "__").upper()
    raw = os.environ.get(env_key)
    if raw is not None and raw != "":
        return _coerce(raw, default)

    node: Any
    try:
        from omnistream.config import get_settings  # type: ignore

        node = get_settings()
    except Exception:  # noqa: BLE001
        try:
            from omnistream.config.settings import get_settings  # type: ignore

            node = get_settings()
        except Exception:  # noqa: BLE001
            return default

    for part in path.split("."):
        node = getattr(node, part, _MISSING)
        if node is _MISSING:
            return default
    if node is None:
        return default
    return node  # type: ignore[return-value]


def normalize_async_url(url: str) -> str:
    """把常见的同步 DSN 归一化为 async 驱动 DSN。"""
    if not url:
        return DEFAULT_SQLITE_URL
    lowered = url.strip()
    if lowered.startswith("postgres://"):
        lowered = "postgresql://" + lowered[len("postgres://") :]
    if lowered.startswith("postgresql://"):
        return "postgresql+asyncpg://" + lowered[len("postgresql://") :]
    if lowered.startswith("postgresql+psycopg2://"):
        return "postgresql+asyncpg://" + lowered[len("postgresql+psycopg2://") :]
    if lowered.startswith("postgresql+psycopg://"):
        return "postgresql+asyncpg://" + lowered[len("postgresql+psycopg://") :]
    if lowered.startswith("sqlite://") and "+aiosqlite" not in lowered:
        return "sqlite+aiosqlite://" + lowered[len("sqlite://") :]
    return lowered


def to_sync_url(url: str) -> str:
    """把 async DSN 转成 alembic 离线/同步模式可用的 DSN。"""
    if url.startswith("postgresql+asyncpg://"):
        return "postgresql+psycopg://" + url[len("postgresql+asyncpg://") :]
    if url.startswith("sqlite+aiosqlite://"):
        return "sqlite://" + url[len("sqlite+aiosqlite://") :]
    return url


def resolve_database_url() -> str:
    """解析最终使用的数据库 URL。"""
    candidate = str(
        setting("storage.database_url", "")
        or setting("storage.db_url", "")
        or setting("storage.postgres_dsn", "")
        or os.environ.get("DATABASE_URL", "")
        or ""
    ).strip()
    if not candidate:
        candidate = DEFAULT_SQLITE_URL
    return normalize_async_url(candidate)


def is_sqlite(url: str | None = None) -> bool:
    return (url or resolve_database_url()).startswith("sqlite")


def is_postgres(url: str | None = None) -> bool:
    return (url or resolve_database_url()).startswith("postgresql")


def _ensure_sqlite_parent(url: str) -> None:
    """SQLite 文件所在目录不存在时自动创建，避免首启失败。"""
    split = urlsplit(url)
    raw_path = split.path or ""
    if not raw_path or raw_path == "/:memory:":
        return
    file_path = raw_path.lstrip("/")
    if file_path in {":memory:", ""}:
        return
    target = Path(file_path)
    if not target.is_absolute():
        target = Path.cwd() / target
    target.parent.mkdir(parents=True, exist_ok=True)


def _install_sqlite_pragmas(sync_engine: Engine) -> None:
    """SQLite 必备 PRAGMA：外键约束 + WAL + 合理的忙等待。"""

    @event.listens_for(sync_engine, "connect")
    def _set_pragmas(dbapi_connection: Any, _record: Any) -> None:  # noqa: ANN401
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=5000")
        finally:
            cursor.close()


def create_engine_from_url(url: str, *, echo: bool | None = None) -> AsyncEngine:
    """按 URL 构建 async engine，自动区分 SQLite / PostgreSQL 的连接池策略。"""
    echo_sql = bool(setting("storage.echo_sql", False)) if echo is None else echo
    sqlite = url.startswith("sqlite")

    kwargs: dict[str, Any] = {
        "echo": echo_sql,
        "future": True,
        "pool_pre_ping": True,
    }
    if sqlite:
        _ensure_sqlite_parent(url)
        kwargs["poolclass"] = NullPool
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        kwargs.pop("pool_pre_ping", None)
    else:
        kwargs["pool_size"] = int(setting("storage.pool_size", 10))
        kwargs["max_overflow"] = int(setting("storage.max_overflow", 20))
        kwargs["pool_recycle"] = int(setting("storage.pool_recycle_s", 1800))
        kwargs["pool_timeout"] = int(setting("storage.pool_timeout_s", 30))

    engine = create_async_engine(make_url(url), **kwargs)
    if sqlite:
        _install_sqlite_pragmas(engine.sync_engine)
    return engine


def get_engine() -> AsyncEngine:
    """进程内全局单例 engine。"""
    global _engine, _engine_url
    if _engine is None:
        _engine_url = resolve_database_url()
        _engine = create_engine_from_url(_engine_url)
        log.info(
            "db_engine_created",
            dialect=_engine.dialect.name,
            driver=_engine.dialect.driver,
            sqlite=_engine_url.startswith("sqlite"),
        )
    return _engine


def current_engine_url() -> str:
    """返回当前 engine 使用的 URL（未创建则解析一次）。"""
    return _engine_url or resolve_database_url()


async def dispose_engine() -> None:
    """关闭连接池（应用 shutdown 钩子调用）。"""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        log.info("db_engine_disposed")
        _engine = None


async def ensure_pgvector(engine: AsyncEngine | None = None) -> bool:
    """在 PostgreSQL 上启用 pgvector 扩展。返回是否成功启用。"""
    eng = engine or get_engine()
    if eng.dialect.name != "postgresql":
        return False
    try:
        async with eng.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        log.info("pgvector_enabled")
        return True
    except Exception as exc:  # noqa: BLE001 - 没有超级用户权限时降级为 JSON 存储
        log.warning("pgvector_enable_failed", error=str(exc))
        return False


async def create_all(engine: AsyncEngine | None = None) -> None:
    """直接按 metadata 建表（开发/Mock 模式的快捷路径，生产用 alembic）。"""
    from omnistream.db import models as _models  # noqa: F401  确保全部模型已注册
    from omnistream.db.base import Base

    eng = engine or get_engine()
    await ensure_pgvector(eng)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("db_schema_created", tables=len(Base.metadata.tables))


async def ping(engine: AsyncEngine | None = None) -> bool:
    """健康探测：执行 ``SELECT 1``。"""
    eng = engine or get_engine()
    try:
        async with eng.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("db_ping_failed", error=str(exc))
        return False


__all__ = [
    "DEFAULT_SQLITE_URL",
    "create_all",
    "create_engine_from_url",
    "current_engine_url",
    "dispose_engine",
    "ensure_pgvector",
    "get_engine",
    "is_postgres",
    "is_sqlite",
    "normalize_async_url",
    "ping",
    "resolve_database_url",
    "setting",
    "to_sync_url",
]

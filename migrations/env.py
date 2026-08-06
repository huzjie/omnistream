"""Alembic 环境。

同时支持：
* 离线模式 ``alembic upgrade head --sql``（生成 SQL 脚本，不连库）
* 在线模式（async engine，SQLite 与 PostgreSQL 通吃）

数据库 URL 解析顺序与运行时完全一致（``omnistream.db.engine.resolve_database_url``），
因此 ``OMS_STORAGE__DATABASE_URL`` 一处配置，服务与迁移同时生效。

用法::

    alembic -c migrations/alembic.ini upgrade head      # 仓库自带配置
    alembic upgrade head                                # 若根目录已有 alembic.ini
"""
from __future__ import annotations

import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# 保证以任意 CWD 调用 alembic 都能 import 到 omnistream 包
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omnistream.db.base import Base  # noqa: E402
from omnistream.db.engine import resolve_database_url, to_sync_url  # noqa: E402
import omnistream.db.models  # noqa: E402,F401  必须导入以注册全部表

config = context.config

if config.config_file_name is not None and os.path.exists(config.config_file_name):
    try:
        fileConfig(config.config_file_name)
    except Exception:  # noqa: BLE001 - 日志配置缺失不应阻断迁移
        pass

target_metadata = Base.metadata

_ASYNC_URL = resolve_database_url()
_IS_SQLITE = _ASYNC_URL.startswith("sqlite")

config.set_main_option("sqlalchemy.url", _ASYNC_URL.replace("%", "%%"))


def include_object(obj, name, type_, reflected, compare_to) -> bool:  # noqa: ANN001
    """跳过 pgvector 在 PG 上自动创建的内部对象。"""
    if type_ == "table" and name in {"spatial_ref_sys"}:
        return False
    return True


def _configure(connection: Connection | None = None, *, url: str | None = None) -> None:
    context.configure(
        connection=connection,
        url=url,
        target_metadata=target_metadata,
        literal_binds=connection is None,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
        # SQLite 不支持大部分 ALTER，统一走 batch 模式
        render_as_batch=_IS_SQLITE,
    )


def run_migrations_offline() -> None:
    """离线模式：只输出 SQL。"""
    _configure(url=to_sync_url(_ASYNC_URL))
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    _configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """在线模式：用 async engine 连库执行。"""
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _ASYNC_URL
    connectable = async_engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

"""数据持久层入口。

导出 engine / session / Base 三件套，业务代码统一从这里取。

    from omnistream.db import Base, get_db, session_scope
"""
from __future__ import annotations

from omnistream.db.base import Base, metadata_obj
from omnistream.db.engine import (
    create_all,
    current_engine_url,
    dispose_engine,
    ensure_pgvector,
    get_engine,
    is_postgres,
    is_sqlite,
    ping,
    resolve_database_url,
    setting,
)
from omnistream.db.session import (
    async_session_factory,
    get_db,
    healthcheck,
    readonly_scope,
    reset_session_factory,
    run_sync,
    run_with_session,
    session_scope,
)
from omnistream.db.types import (
    DEFAULT_VECTOR_DIM,
    JSONB,
    ULIDType,
    UTCDateTime,
    VectorType,
    generate_id,
    utc_now,
)

__all__ = [
    "DEFAULT_VECTOR_DIM",
    "Base",
    "JSONB",
    "ULIDType",
    "UTCDateTime",
    "VectorType",
    "async_session_factory",
    "create_all",
    "current_engine_url",
    "dispose_engine",
    "ensure_pgvector",
    "generate_id",
    "get_db",
    "get_engine",
    "healthcheck",
    "is_postgres",
    "is_sqlite",
    "metadata_obj",
    "ping",
    "readonly_scope",
    "reset_session_factory",
    "resolve_database_url",
    "run_sync",
    "run_with_session",
    "session_scope",
    "setting",
    "utc_now",
]

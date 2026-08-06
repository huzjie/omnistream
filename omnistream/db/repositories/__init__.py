"""仓储导出。

任何带 ``tenant_id`` 的表都必须经 ``TenantScopedRepository``（或 ``repo_for``）访问。
"""
from __future__ import annotations

from omnistream.db.repositories.base import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    GenericRepository,
    PageResult,
    TenantScopedRepository,
    UnscopedRepository,
    clamp_limit,
    repo_for,
)
from omnistream.db.repositories.memory_repo import MemoryRepository, create_memory_repo
from omnistream.db.repositories.session_repo import SessionRepository, create_session_repo

__all__ = [
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "GenericRepository",
    "MemoryRepository",
    "PageResult",
    "SessionRepository",
    "TenantScopedRepository",
    "UnscopedRepository",
    "clamp_limit",
    "create_memory_repo",
    "create_session_repo",
    "repo_for",
]

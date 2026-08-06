"""memory_items + pgvector extension + ivfflat index

在 PostgreSQL 上启用 ``vector`` 扩展并把 embedding 列建成原生 ``vector(384)``；
在 SQLite 上退化为 TEXT（JSON 数组），检索由 Python 侧余弦兜底。

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-06 09:10:00.000000
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from omnistream.db.types import DEFAULT_VECTOR_DIM, JSONB, ULIDType, UTCDateTime, VectorType

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _has_pgvector() -> bool:
    """尝试启用 pgvector；无权限或未安装时返回 False 并降级。"""
    if not _is_postgres():
        return False
    bind = op.get_bind()
    try:
        bind.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception:  # noqa: BLE001
        return False
    try:
        row = bind.execute(
            sa.text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        ).first()
    except Exception:  # noqa: BLE001
        return False
    return row is not None


def upgrade() -> None:
    native_vector = _has_pgvector()
    embedding_type: sa.types.TypeEngine = (
        VectorType(DEFAULT_VECTOR_DIM) if native_vector else sa.Text()
    )

    op.create_table(
        "memory_items",
        sa.Column("id", ULIDType(), nullable=False),
        sa.Column("tenant_id", ULIDType(), nullable=False),
        sa.Column("layer", sa.String(length=32), nullable=False, server_default="session"),
        sa.Column("key", sa.String(length=200), nullable=True),
        sa.Column("subject", sa.String(length=64), nullable=False, server_default="user"),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        sa.Column("embedding", embedding_type, nullable=True),
        sa.Column("embedding_model", sa.String(length=128), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.8"),
        sa.Column("importance", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("hit_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_hit_at", UTCDateTime(), nullable=True),
        sa.Column("expires_at", UTCDateTime(), nullable=True),
        sa.Column("source_session_id", ULIDType(), nullable=True),
        sa.Column("end_user_id", sa.String(length=200), nullable=True),
        sa.Column("tags", JSONB(), nullable=False),
        sa.Column("superseded_by", ULIDType(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_memory_items"),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name="fk_memory_items_tenant_id_tenants", ondelete="CASCADE"
        ),
    )
    op.create_index("ix_memory_items_tenant_id", "memory_items", ["tenant_id"])
    op.create_index("ix_memory_items_layer", "memory_items", ["layer"])
    op.create_index("ix_memory_items_expires_at", "memory_items", ["expires_at"])
    op.create_index("ix_memory_items_created_at", "memory_items", ["created_at"])
    op.create_index("ix_memory_items_deleted_at", "memory_items", ["deleted_at"])
    op.create_index("ix_memory_items_source_session_id", "memory_items", ["source_session_id"])
    op.create_index("ix_memory_items_end_user_id", "memory_items", ["end_user_id"])
    op.create_index("ix_memory_items_tenant_layer", "memory_items", ["tenant_id", "layer"])
    op.create_index("ix_memory_items_tenant_key", "memory_items", ["tenant_id", "key"])
    op.create_index("ix_memory_items_tenant_subject", "memory_items", ["tenant_id", "subject"])

    if native_vector:
        # ivfflat 需要在有数据后 ANALYZE 才最优；lists=100 适配十万级记忆规模。
        op.execute(
            sa.text(
                "CREATE INDEX IF NOT EXISTS ix_memory_items_embedding_ivfflat "
                "ON memory_items USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
            )
        )
        # private_state 层同键唯一（部分索引，只有 key 非空时生效）
        op.execute(
            sa.text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_items_tenant_key_subject "
                "ON memory_items (tenant_id, key, subject) WHERE key IS NOT NULL AND deleted_at IS NULL"
            )
        )


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute(sa.text("DROP INDEX IF EXISTS uq_memory_items_tenant_key_subject"))
        op.execute(sa.text("DROP INDEX IF EXISTS ix_memory_items_embedding_ivfflat"))
    op.drop_table("memory_items")

"""存储配置：Postgres / Redis / S3 兼容对象存储的连接串与池参数。"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OMS_STORAGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------ 关系库
    #: 默认 SQLite 让"克隆即可跑"，生产改为 postgresql+asyncpg://...
    database_url: str = "sqlite+aiosqlite:///./data/omnistream.db"
    db_pool_size: int = Field(default=10, ge=1)
    db_max_overflow: int = Field(default=20, ge=0)
    db_pool_timeout_s: float = Field(default=30.0, gt=0.0)
    db_pool_recycle_s: int = Field(default=1800, ge=60)
    db_echo: bool = False
    db_statement_timeout_ms: int = Field(default=15_000, ge=100)
    auto_migrate: bool = Field(default=True, description="启动时自动执行 alembic upgrade head")

    # ------------------------------------------------------------ Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = Field(default=50, ge=1)
    redis_socket_timeout_s: float = Field(default=5.0, gt=0.0)
    #: 事件总线实现：inproc 单机零依赖；redis 支持多实例水平扩展
    event_bus: str = Field(default="inproc", description="inproc|redis")
    event_stream_prefix: str = "oms:events:"
    event_stream_maxlen: int = Field(default=10_000, ge=100)
    #: Celery broker / backend，留空则复用 redis_url
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None

    # ------------------------------------------------------------ 对象存储
    object_store: str = Field(default="local", description="local|s3|minio|oss|cos")
    local_path: str = "./data/objects"
    s3_endpoint: str | None = None
    s3_region: str = "us-east-1"
    s3_bucket: str = "omnistream"
    s3_access_key: str | None = Field(default=None, repr=False)
    s3_secret_key: str | None = Field(default=None, repr=False)
    s3_use_ssl: bool = True
    s3_path_style: bool = Field(default=True, description="MinIO 需要 True")
    presign_expire_s: int = Field(default=3600, ge=60)
    #: 录制/导出文件的默认保留天数
    retention_days: int = Field(default=30, ge=0)

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith(("postgresql", "postgres"))

    @property
    def sync_database_url(self) -> str:
        """给 Alembic 用的同步驱动 URL。"""
        return (self.database_url
                .replace("+asyncpg", "+psycopg")
                .replace("+aiosqlite", ""))

    @property
    def effective_broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def effective_result_backend(self) -> str:
        return self.celery_result_backend or self.redis_url

    @property
    def supports_pgvector(self) -> bool:
        """SQLite 不支持向量索引，记忆层需退化为内存/暴力检索。"""
        return self.is_postgres

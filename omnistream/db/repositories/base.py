"""``TenantScopedRepository`` —— 强制租户隔离的仓储基类。

**所有**对带 ``tenant_id`` 的表的读写都必须经过本类。
它在每个语句上自动追加 ``WHERE tenant_id = :ctx``，从结构上杜绝越权。
直接 ``session.execute(select(Model))`` 绕过将在 code review 中被拒绝（ADR-0005）。
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import structlog
from sqlalchemy import Select, delete as sa_delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import NotFoundError, ValidationError
from omnistream.db.base import Base
from omnistream.db.types import utc_now

log = structlog.get_logger(__name__)

ModelT = TypeVar("ModelT", bound=Base)

#: 列表接口默认与最大分页大小（§7.9）
DEFAULT_LIMIT = 20
MAX_LIMIT = 200


@dataclass(slots=True)
class PageResult(Generic[ModelT]):
    """cursor 分页结果。ULID 主键天然有序，cursor 即上一页最后一条的 id。"""

    items: list[ModelT] = field(default_factory=list)
    total: int = 0
    cursor: str | None = None
    has_more: bool = False

    def to_payload(self, serializer: Any = None) -> dict[str, Any]:  # noqa: ANN401
        render = serializer or (lambda x: x)
        return {
            "items": [render(item) for item in self.items],
            "total": self.total,
            "cursor": self.cursor,
            "has_more": self.has_more,
        }


def clamp_limit(limit: int | None) -> int:
    if limit is None:
        return DEFAULT_LIMIT
    return max(1, min(int(limit), MAX_LIMIT))


class TenantScopedRepository(Generic[ModelT]):
    """租户作用域仓储。子类只需声明 ``model``。"""

    model: type[ModelT]

    def __init__(self, session: AsyncSession, tenant_id: str, *, include_deleted: bool = False) -> None:
        if not tenant_id:
            raise ValidationError("仓储必须绑定 tenant_id", code=ErrorCode.INVALID_REQUEST)
        self.session = session
        self.tenant_id = tenant_id
        self.include_deleted = include_deleted

    # ---------------- 语句构造 ----------------

    @property
    def _model(self) -> type[ModelT]:
        model = getattr(self, "model", None)
        if model is None:
            raise ValidationError("仓储未声明 model", code=ErrorCode.INTERNAL)
        return model

    def scoped(self, stmt: Select[Any] | None = None) -> Select[Any]:
        """给任意 select 语句注入租户过滤 + 软删除过滤。"""
        model = self._model
        statement = stmt if stmt is not None else select(model)
        if hasattr(model, "tenant_id"):
            statement = statement.where(model.tenant_id == self.tenant_id)  # type: ignore[attr-defined]
        if not self.include_deleted and hasattr(model, "deleted_at"):
            statement = statement.where(model.deleted_at.is_(None))  # type: ignore[attr-defined]
        return statement

    def _order_clause(self, descending: bool) -> ColumnElement[Any]:
        pk = self._model.id  # type: ignore[attr-defined]
        return pk.desc() if descending else pk.asc()

    # ---------------- 读 ----------------

    async def get(self, entity_id: str) -> ModelT | None:
        stmt = self.scoped().where(self._model.id == entity_id)  # type: ignore[attr-defined]
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def require(self, entity_id: str) -> ModelT:
        found = await self.get(entity_id)
        if found is None:
            raise NotFoundError(
                f"{self._model.__tablename__} not found: {entity_id}",
                code=ErrorCode.NOT_FOUND,
                detail={"resource": self._model.__tablename__, "id": entity_id},
            )
        return found

    async def find_one(self, *conditions: ColumnElement[bool]) -> ModelT | None:
        stmt = self.scoped()
        for condition in conditions:
            stmt = stmt.where(condition)
        result = await self.session.execute(stmt.limit(1))
        return result.scalar_one_or_none()

    async def find_all(
        self,
        *conditions: ColumnElement[bool],
        limit: int | None = None,
        order_desc: bool = True,
    ) -> list[ModelT]:
        stmt = self.scoped()
        for condition in conditions:
            stmt = stmt.where(condition)
        stmt = stmt.order_by(self._order_clause(order_desc))
        if limit:
            stmt = stmt.limit(clamp_limit(limit))
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count(self, *conditions: ColumnElement[bool]) -> int:
        stmt = self.scoped(select(func.count()).select_from(self._model))
        for condition in conditions:
            stmt = stmt.where(condition)
        result = await self.session.execute(stmt)
        return int(result.scalar_one() or 0)

    async def paginate(
        self,
        *conditions: ColumnElement[bool],
        limit: int | None = None,
        cursor: str | None = None,
        order_desc: bool = True,
        with_total: bool = True,
    ) -> PageResult[ModelT]:
        """cursor 分页。cursor 为上一页最后一条记录的主键。"""
        size = clamp_limit(limit)
        pk = self._model.id  # type: ignore[attr-defined]

        stmt = self.scoped()
        for condition in conditions:
            stmt = stmt.where(condition)
        if cursor:
            stmt = stmt.where(pk < cursor) if order_desc else stmt.where(pk > cursor)
        stmt = stmt.order_by(self._order_clause(order_desc)).limit(size + 1)

        result = await self.session.execute(stmt)
        rows = list(result.scalars().all())
        has_more = len(rows) > size
        items = rows[:size]

        total = 0
        if with_total:
            total = await self.count(*conditions)

        next_cursor = str(getattr(items[-1], "id")) if items and has_more else None
        return PageResult(items=items, total=total, cursor=next_cursor, has_more=has_more)

    async def exists(self, *conditions: ColumnElement[bool]) -> bool:
        return await self.count(*conditions) > 0

    # ---------------- 写 ----------------

    async def create(self, **values: Any) -> ModelT:
        model = self._model
        payload = dict(values)
        if hasattr(model, "tenant_id"):
            payload["tenant_id"] = self.tenant_id
        entity = model(**payload)
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def add(self, entity: ModelT) -> ModelT:
        if hasattr(entity, "tenant_id"):
            setattr(entity, "tenant_id", self.tenant_id)
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def update(self, entity: ModelT, patch: dict[str, Any]) -> ModelT:
        if getattr(entity, "tenant_id", self.tenant_id) != self.tenant_id:
            raise NotFoundError("跨租户更新被拒绝", code=ErrorCode.NOT_FOUND)
        for key, value in patch.items():
            if key in {"id", "tenant_id", "created_at"}:
                continue
            if hasattr(entity, key):
                setattr(entity, key, value)
        if hasattr(entity, "updated_at"):
            setattr(entity, "updated_at", utc_now())
        await self.session.flush()
        return entity

    async def update_by_id(self, entity_id: str, patch: dict[str, Any]) -> ModelT:
        entity = await self.require(entity_id)
        return await self.update(entity, patch)

    async def delete(self, entity: ModelT, *, hard: bool = False) -> None:
        if getattr(entity, "tenant_id", self.tenant_id) != self.tenant_id:
            raise NotFoundError("跨租户删除被拒绝", code=ErrorCode.NOT_FOUND)
        if not hard and hasattr(entity, "deleted_at"):
            setattr(entity, "deleted_at", utc_now())
            await self.session.flush()
            return
        await self.session.delete(entity)
        await self.session.flush()

    async def delete_by_id(self, entity_id: str, *, hard: bool = False) -> None:
        entity = await self.require(entity_id)
        await self.delete(entity, hard=hard)

    async def purge_where(self, *conditions: ColumnElement[bool]) -> int:
        """物理删除（清理任务用），仍强制租户过滤。"""
        model = self._model
        stmt = sa_delete(model)
        if hasattr(model, "tenant_id"):
            stmt = stmt.where(model.tenant_id == self.tenant_id)  # type: ignore[attr-defined]
        for condition in conditions:
            stmt = stmt.where(condition)
        result = await self.session.execute(stmt)
        return int(result.rowcount or 0)

    async def bulk_add(self, entities: Sequence[ModelT]) -> list[ModelT]:
        for entity in entities:
            if hasattr(entity, "tenant_id"):
                setattr(entity, "tenant_id", self.tenant_id)
        self.session.add_all(list(entities))
        await self.session.flush()
        return list(entities)


class GenericRepository(TenantScopedRepository[ModelT]):
    """运行时指定 model 的通用仓储，省去为每张表写一个子类。"""

    def __init__(
        self,
        session: AsyncSession,
        tenant_id: str,
        model: type[ModelT],
        *,
        include_deleted: bool = False,
    ) -> None:
        self.model = model
        super().__init__(session, tenant_id, include_deleted=include_deleted)


def repo_for(
    session: AsyncSession,
    tenant_id: str,
    model: type[ModelT],
    *,
    include_deleted: bool = False,
) -> GenericRepository[ModelT]:
    """快捷工厂：``repo_for(db, tid, ApiKey)``。"""
    return GenericRepository(session, tenant_id, model, include_deleted=include_deleted)


class UnscopedRepository(Generic[ModelT]):
    """**仅限**无租户维度的表（目前只有 ``tenants``）与平台管理任务使用。"""

    def __init__(self, session: AsyncSession, model: type[ModelT]) -> None:
        self.session = session
        self.model = model

    async def get(self, entity_id: str) -> ModelT | None:
        result = await self.session.execute(select(self.model).where(self.model.id == entity_id))  # type: ignore[attr-defined]
        return result.scalar_one_or_none()

    async def require(self, entity_id: str) -> ModelT:
        found = await self.get(entity_id)
        if found is None:
            raise NotFoundError(
                f"{self.model.__tablename__} not found: {entity_id}", code=ErrorCode.NOT_FOUND
            )
        return found

    async def find_one(self, *conditions: ColumnElement[bool]) -> ModelT | None:
        stmt = select(self.model)
        for condition in conditions:
            stmt = stmt.where(condition)
        result = await self.session.execute(stmt.limit(1))
        return result.scalar_one_or_none()

    async def find_all(self, *conditions: ColumnElement[bool], limit: int = 200) -> list[ModelT]:
        stmt = select(self.model)
        for condition in conditions:
            stmt = stmt.where(condition)
        result = await self.session.execute(stmt.order_by(self.model.id.desc()).limit(limit))  # type: ignore[attr-defined]
        return list(result.scalars().all())

    async def count(self, *conditions: ColumnElement[bool]) -> int:
        stmt = select(func.count()).select_from(self.model)
        for condition in conditions:
            stmt = stmt.where(condition)
        result = await self.session.execute(stmt)
        return int(result.scalar_one() or 0)

    async def add(self, entity: ModelT) -> ModelT:
        self.session.add(entity)
        await self.session.flush()
        return entity


__all__ = [
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "GenericRepository",
    "PageResult",
    "TenantScopedRepository",
    "UnscopedRepository",
    "clamp_limit",
    "repo_for",
]

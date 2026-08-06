"""记忆项仓储：CRUD + 向量相似度检索 + 衰减更新。

检索双路径：
* PostgreSQL + pgvector → 走 SQL 的 ``<=>`` 余弦距离算子，可用 ivfflat 索引；
* SQLite / 无 pgvector → 拉候选集在 Python 侧算余弦（``cosine_similarity``）。

两条路径返回同样的 ``(MemoryItem, score)`` 结构，上层（WS5 memory 层）无感知。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Sequence

import structlog
from sqlalchemy import Float, cast, func, literal_column, or_, select, text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import MemoryError_
from omnistream.db.models.memory_item import MemoryItem
from omnistream.db.repositories.base import PageResult, TenantScopedRepository
from omnistream.db.types import cosine_similarity, utc_now

log = structlog.get_logger(__name__)

#: SQLite 路径下一次最多拉多少候选进内存做余弦
PYTHON_SCAN_LIMIT = 2000


class MemoryRepository(TenantScopedRepository[MemoryItem]):
    """记忆主表仓储。"""

    model = MemoryItem

    def __init__(self, session: AsyncSession, tenant_id: str, *, include_deleted: bool = False) -> None:
        super().__init__(session, tenant_id, include_deleted=include_deleted)
        self._native_vector: bool | None = None

    # ---------------- 能力探测 ----------------

    def uses_native_vector(self) -> bool:
        """当前连接是否支持原生 pgvector 检索。"""
        if self._native_vector is None:
            dialect = self.session.bind.dialect.name if self.session.bind is not None else ""
            if dialect != "postgresql":
                self._native_vector = False
            else:
                try:
                    import pgvector.sqlalchemy  # noqa: F401

                    self._native_vector = True
                except Exception:  # noqa: BLE001
                    self._native_vector = False
        return bool(self._native_vector)

    # ---------------- 写 ----------------

    async def upsert(
        self,
        *,
        layer: str,
        content: str,
        key: str | None = None,
        subject: str = "user",
        embedding: Sequence[float] | None = None,
        embedding_model: str | None = None,
        confidence: float = 0.8,
        importance: float = 0.5,
        expires_at: datetime | None = None,
        source_session_id: str | None = None,
        end_user_id: str | None = None,
        tags: list[str] | None = None,
    ) -> MemoryItem:
        """写入记忆。``private_state`` 层按 ``(tenant_id, key, subject)`` 覆盖写。"""
        vector = [float(x) for x in embedding] if embedding is not None else None
        existing: MemoryItem | None = None
        if key:
            stmt = (
                self.scoped()
                .where(MemoryItem.key == key)
                .where(MemoryItem.layer == layer)
                .where(MemoryItem.subject == subject)
                .limit(1)
            )
            existing = (await self.session.execute(stmt)).scalar_one_or_none()

        if existing is not None:
            existing.content = content
            existing.confidence = confidence
            existing.importance = importance
            existing.expires_at = expires_at
            existing.tags = tags or existing.tags
            existing.updated_at = utc_now()
            if vector is not None:
                existing.embedding = vector
                existing.embedding_model = embedding_model
            if end_user_id:
                existing.end_user_id = end_user_id
            await self.session.flush()
            return existing

        item = MemoryItem(
            tenant_id=self.tenant_id,
            layer=layer,
            key=key,
            subject=subject,
            content=content,
            embedding=vector,
            embedding_model=embedding_model,
            confidence=confidence,
            importance=importance,
            expires_at=expires_at,
            source_session_id=source_session_id,
            end_user_id=end_user_id,
            tags=tags or [],
        )
        self.session.add(item)
        await self.session.flush()
        return item

    async def bulk_upsert(self, items: list[dict[str, Any]]) -> list[MemoryItem]:
        out: list[MemoryItem] = []
        for payload in items:
            out.append(await self.upsert(**payload))
        return out

    async def record_hit(self, item_id: str) -> MemoryItem:
        """命中计数 +1 并刷新最近命中时间。"""
        item = await self.require(item_id)
        item.hit_count = int(item.hit_count or 0) + 1
        item.last_hit_at = utc_now()
        await self.session.flush()
        return item

    async def record_hits(self, item_ids: Sequence[str]) -> int:
        count = 0
        for item_id in item_ids:
            found = await self.get(item_id)
            if found is not None:
                found.hit_count = int(found.hit_count or 0) + 1
                found.last_hit_at = utc_now()
                count += 1
        if count:
            await self.session.flush()
        return count

    async def decay(self, *, half_life_days: float = 30.0, floor: float = 0.05) -> int:
        """批量重要度衰减；低于 floor 的记忆标记软删除。返回处理条数。"""
        now = utc_now()
        stmt = self.scoped().where(MemoryItem.layer != "private_state").limit(PYTHON_SCAN_LIMIT)
        rows = list((await self.session.execute(stmt)).scalars().all())
        touched = 0
        for item in rows:
            new_importance = item.decayed_importance(now, half_life_days=half_life_days)
            if abs(new_importance - float(item.importance)) < 1e-6:
                continue
            item.importance = new_importance
            item.updated_at = now
            touched += 1
            if new_importance < floor and item.hit_count == 0:
                item.deleted_at = now
        if touched:
            await self.session.flush()
        log.info("memory_decayed", tenant_id=self.tenant_id, touched=touched)
        return touched

    async def forget_expired(self, *, hard: bool = False) -> int:
        """清理过期记忆。"""
        now = utc_now()
        stmt = self.scoped().where(MemoryItem.expires_at.is_not(None)).where(MemoryItem.expires_at <= now)
        rows = list((await self.session.execute(stmt)).scalars().all())
        for item in rows:
            if hard:
                await self.session.delete(item)
            else:
                item.deleted_at = now
        if rows:
            await self.session.flush()
        return len(rows)

    async def supersede(self, source_ids: Sequence[str], target_id: str) -> int:
        """把一组记忆标记为被 ``target_id`` 取代（压缩任务用）。"""
        now = utc_now()
        count = 0
        for source_id in source_ids:
            if source_id == target_id:
                continue
            found = await self.get(source_id)
            if found is None:
                continue
            found.superseded_by = target_id
            found.deleted_at = now
            count += 1
        if count:
            await self.session.flush()
        return count

    # ---------------- 读 ----------------

    async def by_key(self, key: str, *, subject: str = "user") -> MemoryItem | None:
        stmt = self.scoped().where(MemoryItem.key == key).where(MemoryItem.subject == subject).limit(1)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list_by_layer(
        self,
        layers: Sequence[str] | None = None,
        *,
        subject: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> PageResult[MemoryItem]:
        conditions = []
        if layers:
            conditions.append(MemoryItem.layer.in_(tuple(layers)))
        if subject:
            conditions.append(MemoryItem.subject == subject)
        return await self.paginate(*conditions, limit=limit, cursor=cursor)

    async def search(
        self,
        *,
        embedding: Sequence[float] | None = None,
        query_text: str = "",
        layers: Sequence[str] | None = None,
        subject: str | None = None,
        end_user_id: str | None = None,
        top_k: int = 8,
        min_score: float = 0.0,
        include_expired: bool = False,
    ) -> list[tuple[MemoryItem, float]]:
        """混合检索：向量相似度 + 关键词命中 + 时新度加成。"""
        top_k = max(1, min(int(top_k), 100))
        if embedding is None and not query_text.strip():
            raise MemoryError_("检索必须提供 embedding 或 query_text", code=ErrorCode.INVALID_REQUEST)

        if embedding is not None and self.uses_native_vector():
            candidates = await self._search_native(
                embedding=embedding,
                layers=layers,
                subject=subject,
                end_user_id=end_user_id,
                top_k=top_k * 4,
                include_expired=include_expired,
            )
        else:
            candidates = await self._search_python(
                embedding=embedding,
                layers=layers,
                subject=subject,
                end_user_id=end_user_id,
                include_expired=include_expired,
            )

        now = utc_now()
        keywords = [token for token in query_text.lower().split() if len(token) >= 2]
        scored: list[tuple[MemoryItem, float]] = []
        for item, vector_score in candidates:
            keyword_score = self._keyword_score(item.content, query_text, keywords)
            recency = self._recency_boost(item, now)
            total = 0.72 * vector_score + 0.18 * keyword_score + 0.10 * recency
            total *= 0.6 + 0.4 * float(item.confidence or 0.0)
            if total >= min_score:
                scored.append((item, round(min(1.0, total), 6)))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    async def _search_native(
        self,
        *,
        embedding: Sequence[float],
        layers: Sequence[str] | None,
        subject: str | None,
        end_user_id: str | None,
        top_k: int,
        include_expired: bool,
    ) -> list[tuple[MemoryItem, float]]:
        """pgvector 路径：``1 - (embedding <=> :query)`` 即余弦相似度。"""
        vector_literal = "[" + ",".join(f"{float(x):.6f}" for x in embedding) + "]"
        distance = literal_column(
            f"memory_items.embedding <=> '{vector_literal}'::vector"
        )
        stmt = self.scoped(select(MemoryItem, cast(distance, Float).label("distance")))
        stmt = stmt.where(MemoryItem.embedding.is_not(None))
        if layers:
            stmt = stmt.where(MemoryItem.layer.in_(tuple(layers)))
        if subject:
            stmt = stmt.where(MemoryItem.subject == subject)
        if end_user_id:
            stmt = stmt.where(
                or_(MemoryItem.end_user_id == end_user_id, MemoryItem.end_user_id.is_(None))
            )
        if not include_expired:
            stmt = stmt.where(
                or_(MemoryItem.expires_at.is_(None), MemoryItem.expires_at > utc_now())
            )
        stmt = stmt.order_by(sa_text("distance ASC")).limit(top_k)

        rows = (await self.session.execute(stmt)).all()
        out: list[tuple[MemoryItem, float]] = []
        for item, dist in rows:
            similarity = max(0.0, min(1.0, 1.0 - float(dist or 1.0)))
            out.append((item, similarity))
        return out

    async def _search_python(
        self,
        *,
        embedding: Sequence[float] | None,
        layers: Sequence[str] | None,
        subject: str | None,
        end_user_id: str | None,
        include_expired: bool,
    ) -> list[tuple[MemoryItem, float]]:
        """SQLite / 无 pgvector 路径：拉候选进内存算余弦。"""
        stmt = self.scoped()
        if layers:
            stmt = stmt.where(MemoryItem.layer.in_(tuple(layers)))
        if subject:
            stmt = stmt.where(MemoryItem.subject == subject)
        if end_user_id:
            stmt = stmt.where(
                or_(MemoryItem.end_user_id == end_user_id, MemoryItem.end_user_id.is_(None))
            )
        if not include_expired:
            stmt = stmt.where(
                or_(MemoryItem.expires_at.is_(None), MemoryItem.expires_at > utc_now())
            )
        stmt = stmt.order_by(MemoryItem.id.desc()).limit(PYTHON_SCAN_LIMIT)
        rows = list((await self.session.execute(stmt)).scalars().all())

        out: list[tuple[MemoryItem, float]] = []
        for item in rows:
            if embedding is not None and item.embedding:
                score = max(0.0, cosine_similarity(embedding, item.embedding))
            else:
                score = 0.0
            out.append((item, score))
        return out

    @staticmethod
    def _keyword_score(content: str, query_text: str, keywords: list[str]) -> float:
        """朴素关键词命中率：字符级子串 + 词级命中的加权。"""
        if not query_text:
            return 0.0
        lowered = content.lower()
        query_lower = query_text.lower().strip()
        if not query_lower:
            return 0.0
        if query_lower in lowered:
            return 1.0
        if not keywords:
            # 中文场景无空格分词，退化为 2-gram 覆盖率
            grams = {query_lower[i : i + 2] for i in range(max(1, len(query_lower) - 1))}
            if not grams:
                return 0.0
            hits = sum(1 for gram in grams if gram in lowered)
            return hits / len(grams)
        hits = sum(1 for token in keywords if token in lowered)
        return hits / len(keywords)

    @staticmethod
    def _recency_boost(item: MemoryItem, now: datetime) -> float:
        """7 天内线性衰减的时新度加成。"""
        reference = item.last_hit_at or item.updated_at or item.created_at
        if reference is None:
            return 0.0
        age_days = max(0.0, (now - reference).total_seconds() / 86400.0)
        return max(0.0, 1.0 - age_days / 7.0)

    # ---------------- 压缩支持 ----------------

    async def cluster_candidates(
        self, *, layer: str, threshold: float = 0.92, limit: int = 500
    ) -> list[list[MemoryItem]]:
        """按余弦相似度做简单单遍聚类，返回可合并的簇（size >= 2）。"""
        stmt = (
            self.scoped()
            .where(MemoryItem.layer == layer)
            .where(MemoryItem.embedding.is_not(None))
            .order_by(MemoryItem.created_at.asc())
            .limit(limit)
        )
        rows = list((await self.session.execute(stmt)).scalars().all())
        clusters: list[list[MemoryItem]] = []
        centroids: list[list[float]] = []
        for item in rows:
            vector = item.embedding or []
            if not vector:
                continue
            placed = False
            for idx, centroid in enumerate(centroids):
                if cosine_similarity(vector, centroid) >= threshold:
                    clusters[idx].append(item)
                    size = len(clusters[idx])
                    centroids[idx] = [
                        (centroid[i] * (size - 1) + vector[i]) / size for i in range(len(centroid))
                    ]
                    placed = True
                    break
            if not placed:
                clusters.append([item])
                centroids.append(list(vector))
        return [cluster for cluster in clusters if len(cluster) >= 2]

    async def stats(self) -> dict[str, Any]:
        stmt = self.scoped(
            select(MemoryItem.layer, func.count(MemoryItem.id), func.avg(MemoryItem.importance))
        ).group_by(MemoryItem.layer)
        rows = (await self.session.execute(stmt)).all()
        by_layer = {
            str(layer): {"count": int(count or 0), "avg_importance": float(avg or 0.0)}
            for layer, count, avg in rows
        }
        total = sum(entry["count"] for entry in by_layer.values())
        return {"total": total, "by_layer": by_layer, "native_vector": self.uses_native_vector()}

    async def expiring_soon(self, *, within_days: int = 3, limit: int = 200) -> list[MemoryItem]:
        deadline = utc_now() + timedelta(days=within_days)
        stmt = (
            self.scoped()
            .where(MemoryItem.expires_at.is_not(None))
            .where(MemoryItem.expires_at <= deadline)
            .order_by(MemoryItem.expires_at.asc())
            .limit(limit)
        )
        return list((await self.session.execute(stmt)).scalars().all())


def create_memory_repo(session: AsyncSession, tenant_id: str) -> MemoryRepository:
    return MemoryRepository(session, tenant_id)


__all__ = ["MemoryRepository", "PYTHON_SCAN_LIMIT", "create_memory_repo"]

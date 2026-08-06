"""向量检索。

- 首选 ChromaDB（真实向量库，支持持久化）
- 未安装时自动回退 numpy 内存向量（TF 哈希特征 + cosine 相似度），全功能降级
"""
from __future__ import annotations

import logging
import math
from typing import Any

from omnistream.memory.layers import MemoryItem, MemoryLayerKind

logger = logging.getLogger(__name__)


class VectorRetriever:
    """向量检索器（numpy / chroma 双后端）。"""

    def __init__(self, backend: str = "numpy", collection: str = "oms_memory",
                 persist_dir: str | None = None, dim: int = 512) -> None:
        self.backend = backend
        self._dim = dim
        self._items: dict[str, MemoryItem] = {}
        self._vectors: dict[str, list[float]] = {}
        self._chroma = None
        self._collection = None
        if backend == "chroma":
            self._init_chroma(collection, persist_dir)

    def _init_chroma(self, collection: str, persist_dir: str | None) -> None:
        try:
            import chromadb  # type: ignore

            client = chromadb.PersistentClient(path=persist_dir or "./data/chroma")
            self._collection = client.get_or_create_collection(collection)
            self._chroma = True
            logger.info("chroma backend ready: %s", collection)
        except Exception as exc:  # noqa: BLE001
            self._chroma = False
            self.backend = "numpy"
            logger.warning("chroma 不可用，回退 numpy: %s", exc)

    # ---------------- 写入 ----------------

    def add(self, item: MemoryItem) -> None:
        self._items[item.id] = item
        vec = self._hash_vector(item.text)
        self._vectors[item.id] = vec
        if self._chroma:
            try:
                self._collection.upsert(
                    ids=[item.id],
                    embeddings=[vec],
                    metadatas=[{"layer": item.layer.value,
                                "role": item.role}],
                    documents=[item.text],
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("chroma upsert failed: %s", exc)

    def remove(self, item_id: str) -> None:
        self._items.pop(item_id, None)
        self._vectors.pop(item_id, None)
        if self._chroma:
            try:
                self._collection.delete(ids=[item_id])
            except Exception:  # noqa: BLE001
                pass

    def clear(self) -> None:
        self._items.clear()
        self._vectors.clear()
        if self._chroma:
            try:
                self._collection.delete(where={})
            except Exception:  # noqa: BLE001
                pass

    # ---------------- 检索 ----------------

    async def search(self, query: str, k: int = 5,
                     layer: MemoryLayerKind | str | None = None) -> list[MemoryItem]:
        if self._chroma:
            return await self._search_chroma(query, k, layer)
        return await self._search_numpy(query, k, layer)

    async def _search_numpy(self, query: str, k: int,
                            layer: MemoryLayerKind | str | None) -> list[MemoryItem]:
        if not self._items:
            return []
        qv = self._hash_vector(query)
        scored: list[tuple[float, MemoryItem]] = []
        for iid, item in self._items.items():
            if layer is not None and item.layer != MemoryLayerKind(layer):
                continue
            sim = self._cosine(qv, self._vectors[iid])
            scored.append((sim * item.confidence, item))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [it for _, it in scored[:k]]

    async def _search_chroma(self, query: str, k: int,
                             layer: MemoryLayerKind | str | None) -> list[MemoryItem]:
        try:
            where = {"layer": MemoryLayerKind(layer).value} if layer else None
            res = self._collection.query(
                query_embeddings=[self._hash_vector(query)],
                n_results=k,
                where=where,
            )
            ids = res.get("ids", [[]])[0]
            return [self._items[i] for i in ids if i in self._items]
        except Exception as exc:  # noqa: BLE001
            logger.debug("chroma query failed, numpy fallback: %s", exc)
            return await self._search_numpy(query, k, layer)

    # ---------------- 向量化（TF 哈希特征） ----------------

    def _hash_vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        tokens = self._tokenize(text)
        for tok in tokens:
            idx = int(hashlib_sha256(tok) % self._dim)
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import re

        text = text.lower()
        cjk = re.findall(r"[\u4e00-\u9fff]", text)
        words = re.findall(r"[a-z0-9_]+", text)
        return cjk + words

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        return float(dot)  # 向量已归一化


def hashlib_sha256(text: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")

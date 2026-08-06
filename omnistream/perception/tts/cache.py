"""TTS 结果缓存：把重复短句的 TTFA 压到 0。

为什么值得做
------------
实时语音助手的回复里有大量**高频固定短句**：
"好的"、"稍等一下"、"我在听"、"抱歉，我没听清"、工具调用时的填充语……
这些句子每轮都可能出现，每次都真去跑一遍神经网络合成，
既浪费算力（云端还要花钱），又让首字发声（TTFA）多出 200~400ms。

缓存命中时 :meth:`CachedTTS.synthesize_text` 直接返回内存里的 PCM，
耗时是**微秒级**，TTFA 完全由播放链路决定。

三重容量约束
------------
1. ``max_entries``：条目数上限，LRU 淘汰；
2. ``max_seconds``：缓存音频总时长上限，防止几条长句吃掉全部内存；
3. ``ttl_s``：条目存活时间，避免换了模型/音色后长期返回旧音色的音频。

缓存键包含音色、语速、音高、音量、情绪、语言与文本全文
（见 :meth:`SynthesisRequest.cache_key`），任何一项变化都会失效，
**不会出现"换了音色还播老音色"的串味问题**。
"""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import structlog

from omnistream.perception.tts.base import BaseStreamingTTS, SynthesisRequest
from omnistream.schemas.audio import AudioChunk

log = structlog.get_logger(__name__)

__all__ = [
    "COMMON_PHRASES",
    "CacheEntry",
    "CacheStats",
    "CachedTTS",
    "TTSCache",
    "warmup_cache",
]

#: 默认条目上限
DEFAULT_MAX_ENTRIES: Final[int] = 512

#: 默认缓存音频总时长上限（秒）
DEFAULT_MAX_SECONDS: Final[float] = 600.0

#: 默认条目存活时间（秒），0 表示不过期
DEFAULT_TTL_S: Final[float] = 3600.0

#: 超过这个时长的句子不缓存（长句复用率低，占地方）
DEFAULT_MAX_ITEM_SECONDS: Final[float] = 12.0

#: 值得预热的高频短句（中英各一组）
COMMON_PHRASES: Final[tuple[str, ...]] = (
    "好的。",
    "好的，我明白了。",
    "稍等一下。",
    "我在听。",
    "嗯。",
    "收到。",
    "抱歉，我没有听清，可以再说一遍吗？",
    "让我想一想。",
    "正在为你查询。",
    "已经帮你处理好了。",
    "还有什么可以帮你的吗？",
    "Sure.",
    "One moment please.",
    "I am listening.",
    "Sorry, I did not catch that.",
    "Let me check that for you.",
)


@dataclass(slots=True)
class CacheEntry:
    """一条缓存记录。"""

    key: str
    samples: np.ndarray
    sample_rate: int
    created_at: float
    hits: int = 0

    @property
    def seconds(self) -> float:
        """音频时长（秒）。"""
        if self.sample_rate <= 0:
            return 0.0
        return self.samples.size / self.sample_rate

    @property
    def nbytes(self) -> int:
        """占用字节数。"""
        return int(self.samples.nbytes)

    def expired(self, ttl_s: float, now: float) -> bool:
        """是否已过期。"""
        return ttl_s > 0.0 and (now - self.created_at) > ttl_s


@dataclass(slots=True)
class CacheStats:
    """缓存运行统计。"""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    inserts: int = 0
    rejected: int = 0

    @property
    def total(self) -> int:
        """总查询次数。"""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """命中率 0~1。"""
        return self.hits / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, Any]:
        """序列化（指标上报用）。"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "inserts": self.inserts,
            "rejected": self.rejected,
            "hit_rate": round(self.hit_rate, 4),
        }


class TTSCache:
    """带三重容量约束的 LRU 音频缓存，线程/协程安全。

    Example::

        cache = TTSCache(max_entries=256)
        hit = cache.get(request.cache_key())
        if hit is None:
            samples = await engine.synthesize_text(request)
            cache.put(request.cache_key(), samples, sample_rate=24000)
    """

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_seconds: float = DEFAULT_MAX_SECONDS,
        ttl_s: float = DEFAULT_TTL_S,
        max_item_seconds: float = DEFAULT_MAX_ITEM_SECONDS,
    ) -> None:
        """初始化。

        Args:
            max_entries: 条目数上限，<=0 表示禁用缓存。
            max_seconds: 缓存音频总时长上限（秒）。
            ttl_s: 条目存活时间（秒），0 表示不过期。
            max_item_seconds: 单条最长时长，超过则不缓存。
        """
        self.max_entries = int(max_entries)
        self.max_seconds = max(0.0, float(max_seconds))
        self.ttl_s = max(0.0, float(ttl_s))
        self.max_item_seconds = max(0.0, float(max_item_seconds))

        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._seconds = 0.0
        self._lock = asyncio.Lock()
        self.stats = CacheStats()

    @property
    def enabled(self) -> bool:
        """缓存是否启用。"""
        return self.max_entries > 0

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    @property
    def seconds(self) -> float:
        """当前缓存的音频总时长（秒）。"""
        return round(self._seconds, 3)

    @property
    def nbytes(self) -> int:
        """当前占用字节数。"""
        return sum(e.nbytes for e in self._entries.values())

    def get(self, key: str) -> CacheEntry | None:
        """查缓存并把命中项移到 LRU 尾部。

        Args:
            key: 缓存键。

        Returns:
            命中的条目，未命中或已过期返回 ``None``。
        """
        if not self.enabled:
            return None
        entry = self._entries.get(key)
        if entry is None:
            self.stats.misses += 1
            return None
        if entry.expired(self.ttl_s, time.monotonic()):
            self._drop(key)
            self.stats.expirations += 1
            self.stats.misses += 1
            return None
        entry.hits += 1
        self._entries.move_to_end(key)
        self.stats.hits += 1
        return entry

    def put(self, key: str, samples: np.ndarray, *, sample_rate: int) -> bool:
        """写入缓存并按容量淘汰。

        Args:
            key: 缓存键。
            samples: 单声道 float32 PCM。
            sample_rate: 采样率。

        Returns:
            是否真的写入（太长或缓存禁用时返回 ``False``）。
        """
        if not self.enabled or sample_rate <= 0:
            return False
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return False
        seconds = arr.size / sample_rate
        if self.max_item_seconds > 0.0 and seconds > self.max_item_seconds:
            self.stats.rejected += 1
            return False

        if key in self._entries:
            self._drop(key)
        entry = CacheEntry(
            key=key,
            samples=arr.copy(),
            sample_rate=int(sample_rate),
            created_at=time.monotonic(),
        )
        self._entries[key] = entry
        self._seconds += seconds
        self.stats.inserts += 1
        self._evict()
        return True

    async def get_async(self, key: str) -> CacheEntry | None:
        """协程安全的查询。"""
        async with self._lock:
            return self.get(key)

    async def put_async(self, key: str, samples: np.ndarray, *,
                        sample_rate: int) -> bool:
        """协程安全的写入。"""
        async with self._lock:
            return self.put(key, samples, sample_rate=sample_rate)

    def clear(self) -> int:
        """清空缓存，返回清掉的条目数。"""
        count = len(self._entries)
        self._entries.clear()
        self._seconds = 0.0
        return count

    def prune(self) -> int:
        """清掉所有过期条目，返回清掉的条目数。"""
        now = time.monotonic()
        stale = [k for k, e in self._entries.items() if e.expired(self.ttl_s, now)]
        for key in stale:
            self._drop(key)
        self.stats.expirations += len(stale)
        return len(stale)

    def top_keys(self, limit: int = 10) -> list[tuple[str, int]]:
        """命中最多的若干键（运维排查用）。"""
        ranked = sorted(self._entries.values(), key=lambda e: e.hits, reverse=True)
        return [(e.key[:80], e.hits) for e in ranked[:limit]]

    def snapshot(self) -> dict[str, Any]:
        """状态快照。"""
        data = self.stats.as_dict()
        data.update({
            "entries": len(self._entries),
            "seconds": self.seconds,
            "bytes": self.nbytes,
            "max_entries": self.max_entries,
            "max_seconds": self.max_seconds,
        })
        return data

    def _drop(self, key: str) -> None:
        """移除单个条目并回收容量计数。"""
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._seconds = max(0.0, self._seconds - entry.seconds)

    def _evict(self) -> None:
        """按 LRU 淘汰，直到同时满足条目数与总时长约束。"""
        while self._entries and (
            len(self._entries) > self.max_entries
            or (self.max_seconds > 0.0 and self._seconds > self.max_seconds)
        ):
            oldest, _ = next(iter(self._entries.items()))
            self._drop(oldest)
            self.stats.evictions += 1

    def __repr__(self) -> str:
        return (f"<TTSCache entries={len(self._entries)} "
                f"seconds={self.seconds} hit_rate={self.stats.hit_rate:.2f}>")


class CachedTTS(BaseStreamingTTS):
    """给任意 TTS 引擎套一层缓存的装饰器。

    对上层完全透明：它自己就是一个 :class:`BaseStreamingTTS`，
    流式编排、取消、pts 连续性全部继承自基类，
    只是在 :meth:`synthesize_text` 这一层做了查表。

    Example::

        engine = CachedTTS(PiperTTS(voice="xiaoyan"))
        await warmup_cache(engine)
        async for chunk in engine.synthesize(text_stream):
            ...
    """

    def __init__(
        self,
        inner: BaseStreamingTTS,
        *,
        cache: TTSCache | None = None,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_seconds: float = DEFAULT_MAX_SECONDS,
        ttl_s: float = DEFAULT_TTL_S,
        first_chunk_chars: int = 8,
    ) -> None:
        """包装一个引擎。

        Args:
            inner: 被包装的真实引擎。
            cache: 复用已有缓存实例（多会话共享时传这个）。
            max_entries: 新建缓存时的条目上限。
            max_seconds: 新建缓存时的总时长上限。
            ttl_s: 新建缓存时的存活时间。
            first_chunk_chars: 首句最小字符数。
        """
        super().__init__(
            voice=inner.voice,
            language=inner.language,
            sample_rate=inner.sample_rate,
            speed=inner.speed,
            pitch=inner.pitch,
            volume_db=inner.volume_db,
            frame_ms=inner.frame_ms,
            session_id=inner.session_id,
        )
        self.inner = inner
        self.name = f"{inner.name}+cache"
        self.capabilities = inner.capabilities
        self.cache = cache if cache is not None else TTSCache(
            max_entries=max_entries, max_seconds=max_seconds, ttl_s=ttl_s
        )
        self.first_chunk_chars = max(1, int(first_chunk_chars))

    # ------------------------------------------------------------------
    # 透传给内层引擎
    # ------------------------------------------------------------------

    def available_voices(self) -> list[str]:
        """内层引擎的音色列表。"""
        return self.inner.available_voices()

    def set_voice(self, voice: str) -> None:
        """同步切换内外层音色。"""
        super().set_voice(voice)
        self.inner.set_voice(voice)

    async def cancel(self) -> None:
        """同时取消内外层。"""
        await super().cancel()
        await self.inner.cancel()

    def rearm(self) -> None:
        """同时重置内外层。"""
        super().rearm()
        self.inner.rearm()

    async def aclose(self) -> None:
        """关闭内层引擎（缓存保留，可能被别的会话共享）。"""
        await self.inner.aclose()

    # ------------------------------------------------------------------
    # 缓存逻辑
    # ------------------------------------------------------------------

    async def synthesize(self, text_stream: AsyncIterator[str]) -> AsyncIterator[AudioChunk]:
        """消费文本流并产出音频帧。"""
        async for chunk in self._stream_sentences(
            text_stream, first_chunk_chars=self.first_chunk_chars
        ):
            yield chunk

    async def synthesize_text(self, request: SynthesisRequest) -> np.ndarray:
        """先查缓存，未命中再落到内层引擎并回写。"""
        if not request.text.strip():
            return np.zeros(0, dtype=np.float32)
        key = request.cache_key()
        hit = self.cache.get(key)
        if hit is not None and hit.sample_rate == self.sample_rate:
            log.debug("tts.cache.hit", engine=self.inner.name,
                      chars=len(request.text), hits=hit.hits)
            return hit.samples

        samples = await self.inner.synthesize_text(request)
        if samples.size and not self.cancelled:
            self.cache.put(key, samples, sample_rate=self.sample_rate)
        return samples

    def stats(self) -> dict[str, Any]:
        """统计快照（含缓存指标）。"""
        base = super().stats()
        base["inner"] = self.inner.stats()
        base["cache"] = self.cache.snapshot()
        return base

    def __repr__(self) -> str:
        return f"<CachedTTS inner={self.inner.name} {self.cache!r}>"


async def warmup_cache(
    tts: CachedTTS,
    phrases: Sequence[str] | Iterable[str] = COMMON_PHRASES,
    *,
    concurrency: int = 2,
) -> int:
    """预合成高频短句，把它们提前放进缓存。

    通常在会话建立后立刻以后台任务的形式调用，
    这样用户第一次听到"好的"时 TTFA 就是 0。

    Args:
        tts: 已经套了缓存的引擎。
        phrases: 待预热的句子集合。
        concurrency: 并发度，避免把本地推理线程占满。

    Returns:
        成功写入缓存的条目数。
    """
    items = [p for p in phrases if p and p.strip()]
    if not items or not tts.cache.enabled:
        return 0

    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    warmed = 0

    async def _one(text: str, index: int) -> bool:
        async with semaphore:
            request = tts._build_request(text, index=index, pts_offset_ms=0)
            if tts.cache.get(request.cache_key()) is not None:
                return False
            try:
                samples = await tts.inner.synthesize_text(request)
            except Exception as exc:  # noqa: BLE001 - 预热失败不影响主链路
                log.debug("tts.warmup.failed", text=text[:24], error=str(exc))
                return False
            if samples.size == 0:
                return False
            return tts.cache.put(request.cache_key(), samples,
                                 sample_rate=tts.sample_rate)

    results = await asyncio.gather(
        *(_one(text, i) for i, text in enumerate(items)), return_exceptions=True
    )
    for result in results:
        if result is True:
            warmed += 1
    log.info("tts.warmup.done", engine=tts.inner.name,
             requested=len(items), warmed=warmed)
    return warmed

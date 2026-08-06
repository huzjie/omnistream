"""敏感词库加载与多模匹配。

优先使用 ``ahocorasick``（若存在）做 Aho-Corasick 一次扫描多模匹配；否则退化为顺序子串
扫描，保证零依赖也能跑。支持租户自定义词表。
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["SensitiveWordFilter", "WordHit"]

try:  # pragma: no cover - 取决于是否安装 pyahocorasick
    import ahocorasick

    HAVE_AC = True
except Exception:  # noqa: BLE001
    ahocorasick = None  # type: ignore[assignment]
    HAVE_AC = False


@dataclass(slots=True)
class WordHit:
    """一处敏感词命中。"""

    word: str
    start: int
    end: int


class SensitiveWordFilter:
    """敏感词过滤器。"""

    def __init__(self, words: list[str] | None = None) -> None:
        self._words: set[str] = set()
        self._ac: Any = None
        if words:
            self.load(words)

    def load(self, words: list[str] | None) -> None:
        """加载词表（去重、去空）。"""
        self._words = {w for w in (words or []) if w}
        self._build()

    def add(self, word: str) -> None:
        if word and word not in self._words:
            self._words.add(word)
            self._build()

    @property
    def size(self) -> int:
        return len(self._words)

    def _build(self) -> None:
        if HAVE_AC and ahocorasick is not None:
            automaton = ahocorasick.Automaton()
            for w in self._words:
                automaton.add_word(w, w)
            automaton.make_automaton()
            self._ac = automaton
        else:
            self._ac = None

    def scan(self, text: str | None) -> list[WordHit]:
        """扫描文本返回所有命中（可能重叠，由调用方决定去重）。"""
        if not text or not self._words:
            return []
        out: list[WordHit] = []
        if HAVE_AC and self._ac is not None:
            for end, word in self._ac.iter(text):
                start = end - len(word) + 1
                out.append(WordHit(word=word, start=start, end=end + 1))
        else:
            for w in self._words:
                start = text.find(w)
                while start != -1:
                    out.append(WordHit(word=w, start=start, end=start + len(w)))
                    start = text.find(w, start + 1)
        out.sort(key=lambda item: item.start)
        return out

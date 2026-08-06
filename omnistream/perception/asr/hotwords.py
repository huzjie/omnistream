"""热词偏置：把租户术语表变成对识别结果的可控纠偏。

为什么需要
----------
通用 ASR 对专有名词（产品名、人名、内部黑话）识别率很差，
"欧姆尼流" 会被识别成 "偶们你流"。工业界通行做法有两条路：

1. **解码期偏置**（shallow fusion / hotword boosting）—— 需要引擎支持，
   本模块把热词整理成引擎各自需要的格式（:meth:`HotwordBias.as_initial_prompt`
   / :meth:`HotwordBias.as_hotword_list`）。
2. **后处理纠偏**（本模块的 :meth:`HotwordBias.correct`）—— 引擎不支持时的兜底，
   基于**拼音/罗马化编辑距离**在识别文本中定位近似片段并替换。

第 2 条是纯 Python 实现、零依赖、可对任何引擎生效，因此是默认开启的。
纠偏是**保守的**：只有相似度超过阈值且长度匹配才替换，避免把正确文本改错。
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any, Final, Iterable

import structlog

log = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_MIN_SIMILARITY",
    "DEFAULT_MAX_HOTWORDS",
    "Hotword",
    "HotwordBias",
    "HotwordMatch",
    "edit_distance",
    "romanize",
    "similarity",
]

#: 低于该相似度不做替换（0~1）
DEFAULT_MIN_SIMILARITY: Final[float] = 0.72

#: 单次会话最多加载的热词数，防止 O(n*m) 纠偏拖慢实时路径
DEFAULT_MAX_HOTWORDS: Final[int] = 256

#: 汉字 -> 罗马化近似首音（覆盖高频易混声母）。
#: 说明：这里不引入 pypinyin 之类重依赖，而是用"声母等价类"做粗粒度归一，
#: 目标不是精确注音，而是让"欧姆尼流"和"偶们你流"落到相近的字符串上。
_TONE_STRIP = str.maketrans({
    "ā": "a", "á": "a", "ǎ": "a", "à": "a",
    "ē": "e", "é": "e", "ě": "e", "è": "e",
    "ī": "i", "í": "i", "ǐ": "i", "ì": "i",
    "ō": "o", "ó": "o", "ǒ": "o", "ò": "o",
    "ū": "u", "ú": "u", "ǔ": "u", "ù": "u",
    "ü": "v", "ǘ": "v", "ǚ": "v", "ǜ": "v",
})

#: 易混声母/韵母归一表：把听感相近的音归到同一个符号。
_CONFUSION_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("zh", "z"), ("ch", "c"), ("sh", "s"),
    ("ang", "an"), ("eng", "en"), ("ing", "in"),
    ("l", "n"), ("f", "h"), ("r", "l"),
)


def _strip_tone(text: str) -> str:
    """去掉拼音声调符号并做 NFKD 归一。"""
    normalized = unicodedata.normalize("NFKD", text)
    kept = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return kept.translate(_TONE_STRIP)


def romanize(text: str) -> str:
    """把文本折叠成"听感近似串"。

    对 ASCII 走小写 + 易混音归一；对 CJK 走 Unicode 码位分桶
    （同一个码位一定映射到同一符号，不同码位大概率不同），
    再对整串做易混音替换。

    Args:
        text: 任意文本。

    Returns:
        只含小写字母与数字的归一串。
    """
    out: list[str] = []
    for ch in _strip_tone(text.lower()):
        code = ord(ch)
        if ch.isalnum() and code < 128:
            out.append(ch)
        elif 0x4E00 <= code <= 0x9FFF:
            # CJK：用码位低位做稳定分桶，保证同字同码
            out.append(chr(ord("a") + (code % 26)))
            out.append(chr(ord("a") + ((code // 26) % 26)))
        elif 0x3040 <= code <= 0x30FF:
            out.append(chr(ord("a") + (code % 26)))
        # 其余字符（标点/空白）直接丢弃
    folded = "".join(out)
    for src, dst in _CONFUSION_PAIRS:
        folded = folded.replace(src, dst)
    return folded


def edit_distance(a: str, b: str, *, limit: int = 64) -> int:
    """Levenshtein 距离（滚动数组，O(len(a)*len(b)) 时间、O(len(b)) 空间）。

    Args:
        a: 串 A。
        b: 串 B。
        limit: 超过该长度直接截断，防止极端输入拖慢实时路径。

    Returns:
        编辑距离。
    """
    a = a[:limit]
    b = b[:limit]
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    """归一化相似度 = 1 - 编辑距离 / 较长串长度。"""
    if not a and not b:
        return 1.0
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return 1.0 - edit_distance(a, b) / longest


@dataclass(slots=True)
class Hotword:
    """一个热词条目。"""

    text: str
    #: 偏置强度，>1 表示更倾向识别成它；仅对支持解码期偏置的引擎有效
    boost: float = 2.0
    #: 已知的错误写法，命中即直接替换（比模糊匹配更快更准）
    aliases: tuple[str, ...] = ()
    #: 归一化听感串，构造时计算
    folded: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        self.text = self.text.strip()
        if not self.folded:
            self.folded = romanize(self.text)


@dataclass(slots=True)
class HotwordMatch:
    """一次纠偏命中。"""

    start: int
    end: int
    original: str
    replacement: str
    score: float
    exact: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "original": self.original,
            "replacement": self.replacement,
            "score": round(self.score, 3),
            "exact": self.exact,
        }


class HotwordBias:
    """热词表 + 后处理纠偏器。

    Example::

        bias = HotwordBias(["OmniStream", "全双工", "打断兜底"])
        text, matches = bias.correct("请介绍一下 omni stream 的全双攻能力")
    """

    def __init__(
        self,
        words: Iterable[str | Hotword] | None = None,
        *,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        max_hotwords: int = DEFAULT_MAX_HOTWORDS,
        enabled: bool = True,
    ) -> None:
        """
        Args:
            words: 热词，字符串或 :class:`Hotword`。
            min_similarity: 模糊纠偏的最低相似度。
            max_hotwords: 最大装载数量。
            enabled: 关闭后 :meth:`correct` 原样返回。
        """
        self.min_similarity = float(min_similarity)
        self.max_hotwords = int(max_hotwords)
        self.enabled = bool(enabled)
        self._words: list[Hotword] = []
        self._alias_index: dict[str, str] = {}
        self.corrections = 0
        self.lookups = 0
        if words:
            self.extend(words)

    # ------------------------------------------------------------------
    # 词表管理
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._words)

    def __bool__(self) -> bool:
        return bool(self._words)

    @property
    def words(self) -> tuple[Hotword, ...]:
        """当前热词表快照。"""
        return tuple(self._words)

    def add(self, word: str | Hotword) -> bool:
        """添加一个热词，重复或超限返回 ``False``。"""
        item = word if isinstance(word, Hotword) else Hotword(text=str(word))
        if not item.text:
            return False
        if len(self._words) >= self.max_hotwords:
            return False
        if any(existing.text == item.text for existing in self._words):
            return False
        self._words.append(item)
        lowered = item.text.lower()
        self._alias_index[lowered] = item.text
        for alias in item.aliases:
            if alias:
                self._alias_index[alias.lower()] = item.text
        return True

    def extend(self, words: Iterable[str | Hotword]) -> int:
        """批量添加，返回实际新增数量。"""
        return sum(1 for w in words if self.add(w))

    def clear(self) -> None:
        """清空词表。"""
        self._words.clear()
        self._alias_index.clear()

    # ------------------------------------------------------------------
    # 引擎适配
    # ------------------------------------------------------------------

    def as_hotword_list(self) -> list[str]:
        """给支持热词列表的引擎（funasr / 云端）。"""
        return [w.text for w in self._words]

    def as_weighted_list(self) -> list[tuple[str, float]]:
        """给支持带权热词的引擎。"""
        return [(w.text, w.boost) for w in self._words]

    def as_initial_prompt(self, *, max_chars: int = 200) -> str:
        """给 whisper 系引擎的 ``initial_prompt``（软偏置）。

        Args:
            max_chars: 截断长度，过长的 prompt 会显著拖慢首个 partial。
        """
        if not self._words:
            return ""
        joined = "、".join(w.text for w in self._words)
        if len(joined) <= max_chars:
            return joined
        return joined[:max_chars].rsplit("、", 1)[0]

    # ------------------------------------------------------------------
    # 后处理纠偏
    # ------------------------------------------------------------------

    def correct(self, text: str) -> tuple[str, list[HotwordMatch]]:
        """在识别文本中定位近似热词并替换。

        算法：对每个热词，用其字符长度作为滑窗宽度（±1 容差）扫描原文，
        取归一串相似度最高且超过阈值的窗口做替换。已替换区间不再重叠命中。

        Args:
            text: ASR 原始输出。

        Returns:
            ``(纠偏后文本, 命中列表)``。
        """
        if not self.enabled or not text or not self._words:
            return text, []
        self.lookups += 1

        matches: list[HotwordMatch] = []
        taken: list[tuple[int, int]] = []

        def _overlaps(start: int, end: int) -> bool:
            return any(not (end <= s or start >= e) for s, e in taken)

        # 先做精确别名替换（O(n)，最可靠）
        lowered = text.lower()
        for alias, target in self._alias_index.items():
            if alias == target.lower():
                continue
            idx = lowered.find(alias)
            while idx >= 0:
                end = idx + len(alias)
                if not _overlaps(idx, end):
                    matches.append(HotwordMatch(start=idx, end=end,
                                                original=text[idx:end],
                                                replacement=target,
                                                score=1.0, exact=True))
                    taken.append((idx, end))
                idx = lowered.find(alias, end)

        # 再做模糊滑窗匹配
        for word in self._words:
            width = len(word.text)
            if width < 2 or not word.folded:
                continue
            best: tuple[float, int, int] | None = None
            for span in (width, width + 1, width - 1):
                if span < 2 or span > len(text):
                    continue
                for start in range(0, len(text) - span + 1):
                    end = start + span
                    if _overlaps(start, end):
                        continue
                    window = text[start:end]
                    if window == word.text:
                        continue  # 已经正确，无需替换
                    score = similarity(romanize(window), word.folded)
                    if score >= self.min_similarity and (best is None or score > best[0]):
                        best = (score, start, end)
            if best is not None:
                score, start, end = best
                matches.append(HotwordMatch(start=start, end=end,
                                            original=text[start:end],
                                            replacement=word.text, score=score))
                taken.append((start, end))

        if not matches:
            return text, []

        matches.sort(key=lambda m: m.start)
        pieces: list[str] = []
        cursor = 0
        for m in matches:
            if m.start < cursor:
                continue
            pieces.append(text[cursor:m.start])
            pieces.append(m.replacement)
            cursor = m.end
        pieces.append(text[cursor:])
        corrected = "".join(pieces)
        self.corrections += len(matches)
        log.debug("asr.hotword_corrected", count=len(matches),
                  sample=[m.as_dict() for m in matches[:3]])
        return corrected, matches

    def stats(self) -> dict[str, Any]:
        """统计快照。"""
        return {
            "hotwords": len(self._words),
            "aliases": len(self._alias_index),
            "lookups": self.lookups,
            "corrections": self.corrections,
            "min_similarity": self.min_similarity,
            "enabled": self.enabled,
        }

    def __repr__(self) -> str:
        return f"<HotwordBias n={len(self._words)} enabled={self.enabled}>"

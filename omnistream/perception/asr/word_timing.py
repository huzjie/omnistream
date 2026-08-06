"""词级时间戳：对齐、插值、切分与导出。

用途
----
1. **打断点定位**：用户在助手说到第 N 个字时打断，需要把 ``played_ms``
   反查成"说到哪个词"，回填给模型（§4.1 ``truncate_context_on_barge_in``）。
2. **字幕对齐**：控制台高亮当前词。
3. **引擎补齐**：多数流式引擎不给词级时间戳，这里用
   :func:`interpolate_words` 按字符权重线性插值出一份可用的近似时间轴。

时间轴统一为**相对本段语音起点的毫秒**，与 :class:`WordTiming` 契约一致。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final, Iterable, Sequence

from omnistream.perception.asr.base import build_word_timing
from omnistream.schemas.transcript import WordTiming

__all__ = [
    "CJK_CHAR_WEIGHT",
    "LATIN_CHAR_WEIGHT",
    "WordSpan",
    "align_to_pts",
    "interpolate_words",
    "locate_word_at",
    "merge_word_lists",
    "split_words",
    "text_before_ms",
    "to_srt_cues",
    "total_span_ms",
]

#: 一个 CJK 字符的相对时长权重（中文单字信息量大、发音时长稳定）
CJK_CHAR_WEIGHT: Final[float] = 1.0

#: 一个拉丁字母的相对时长权重（英文按字母摊，约 3~4 个字母 ≈ 1 个汉字）
LATIN_CHAR_WEIGHT: Final[float] = 0.32

#: 标点几乎不占发音时长，但占一点停顿
_PUNCT_WEIGHT: Final[float] = 0.25

#: 分词正则：连续拉丁词/数字算一个词，单个 CJK 字算一个词，标点单独成词
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?"      # 英文单词（含缩写撇号）
    r"|\d+(?:[.,]\d+)*"              # 数字
    r"|[\u4e00-\u9fff]"              # 单个汉字
    r"|[\u3040-\u30ff]+"             # 日文假名串
    r"|[\uac00-\ud7af]+"             # 韩文
    r"|[^\s\w]"                      # 标点
)

_PUNCT_RE: Final[re.Pattern[str]] = re.compile(r"^[^\s\w]$")


@dataclass(slots=True)
class WordSpan:
    """内部使用的轻量词区间（避免频繁构造 pydantic 模型）。"""

    word: str
    start_ms: int
    end_ms: int
    confidence: float = 1.0

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    def to_model(self) -> WordTiming:
        """转成 WS1 契约模型。"""
        return build_word_timing(self.word, self.start_ms, self.end_ms, self.confidence)

    def as_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "confidence": round(self.confidence, 3),
        }


def _token_weight(token: str) -> float:
    """估算一个 token 的相对发音时长权重。"""
    if _PUNCT_RE.match(token):
        return _PUNCT_WEIGHT
    weight = 0.0
    for ch in token:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF or 0xAC00 <= code <= 0xD7AF:
            weight += CJK_CHAR_WEIGHT
        elif 0x3040 <= code <= 0x30FF:
            weight += CJK_CHAR_WEIGHT * 0.7
        else:
            weight += LATIN_CHAR_WEIGHT
    return max(weight, LATIN_CHAR_WEIGHT)


def split_words(text: str) -> list[str]:
    """把文本切成词单元（中文按字、英文按词、标点独立）。

    Args:
        text: 输入文本。

    Returns:
        token 列表；空文本返回空列表。
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text)


def interpolate_words(
    text: str,
    start_ms: int,
    end_ms: int,
    *,
    confidence: float = 0.6,
    skip_punct: bool = True,
) -> list[WordSpan]:
    """按字符权重把 ``[start_ms, end_ms]`` 线性摊到每个词上。

    这是引擎不提供词级时间戳时的近似方案。精度约 ±80ms，
    足以支撑字幕高亮与打断点定位，不足以做口型同步。

    Args:
        text: 该段文本。
        start_ms: 段起始时间（相对本段语音起点）。
        end_ms: 段结束时间。
        confidence: 统一置信度，插值结果本身不可靠，默认给 0.6。
        skip_punct: 是否在输出中剔除纯标点 token。

    Returns:
        词区间列表；文本为空或时长非正时返回空列表。
    """
    tokens = split_words(text)
    if not tokens:
        return []
    duration = max(0, int(end_ms) - int(start_ms))
    if duration <= 0:
        return [WordSpan(word=t, start_ms=int(start_ms), end_ms=int(start_ms),
                         confidence=confidence)
                for t in tokens if not (skip_punct and _PUNCT_RE.match(t))]

    weights = [_token_weight(t) for t in tokens]
    total = sum(weights) or 1.0
    spans: list[WordSpan] = []
    cursor = float(start_ms)
    for token, weight in zip(tokens, weights, strict=True):
        span_ms = duration * (weight / total)
        begin = int(round(cursor))
        cursor += span_ms
        finish = int(round(cursor))
        if skip_punct and _PUNCT_RE.match(token):
            continue
        spans.append(WordSpan(word=token, start_ms=begin,
                              end_ms=max(begin, finish), confidence=confidence))
    if spans:
        spans[-1].end_ms = max(spans[-1].end_ms, int(end_ms))
    return spans


def align_to_pts(words: Iterable[WordSpan], offset_ms: int) -> list[WordSpan]:
    """把相对时间轴平移到绝对 pts 时间轴。

    Args:
        words: 词区间。
        offset_ms: 本段语音起点对应的绝对 pts。

    Returns:
        平移后的新列表（不修改入参）。
    """
    shift = int(offset_ms)
    return [WordSpan(word=w.word, start_ms=w.start_ms + shift,
                     end_ms=w.end_ms + shift, confidence=w.confidence)
            for w in words]


def locate_word_at(words: Sequence[WordSpan], ms: int) -> int:
    """二分查找 ``ms`` 落在第几个词上。

    Args:
        words: **按时间升序**的词区间。
        ms: 查询时刻。

    Returns:
        词下标；早于首词返回 ``-1``，晚于末词返回 ``len(words)-1``。
    """
    if not words:
        return -1
    if ms < words[0].start_ms:
        return -1
    lo, hi = 0, len(words) - 1
    result = len(words) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        span = words[mid]
        if ms < span.start_ms:
            hi = mid - 1
            result = mid - 1
        elif ms > span.end_ms:
            lo = mid + 1
        else:
            return mid
    return max(-1, min(result, len(words) - 1))


def text_before_ms(words: Sequence[WordSpan], ms: int, *, separator: str = "") -> str:
    """取出 ``ms`` 之前已经说出口的文本（barge-in 上下文回填用）。

    Args:
        words: 按时间升序的词区间。
        ms: 截断时刻（通常是 ``playback.played_ms``）。
        separator: 词间分隔符；中文场景传空串，英文场景传空格。

    Returns:
        截断文本；未说出任何词时返回空串。
    """
    idx = locate_word_at(words, ms)
    if idx < 0:
        return ""
    spoken = [w.word for w in words[: idx + 1]]
    if separator:
        return separator.join(spoken)
    # 中英混排：英文词之间补空格，中文字之间不补
    out: list[str] = []
    for token in spoken:
        if out and token[:1].isascii() and token[:1].isalnum() \
                and out[-1][-1:].isascii() and out[-1][-1:].isalnum():
            out.append(" ")
        out.append(token)
    return "".join(out)


def merge_word_lists(*lists: Sequence[WordSpan]) -> list[WordSpan]:
    """合并多段词级时间戳并按起始时间排序、去重叠。

    重叠时保留先出现（时间更早）的那个，后者起点被推后到前者终点。

    Args:
        *lists: 多段词区间。

    Returns:
        合并后的单调列表。
    """
    merged: list[WordSpan] = []
    for chunk in lists:
        merged.extend(chunk)
    merged.sort(key=lambda w: (w.start_ms, w.end_ms))
    out: list[WordSpan] = []
    for span in merged:
        if out and span.start_ms < out[-1].end_ms:
            shifted_start = out[-1].end_ms
            if shifted_start >= span.end_ms:
                continue  # 完全被吞没，丢弃
            span = WordSpan(word=span.word, start_ms=shifted_start,
                            end_ms=span.end_ms, confidence=span.confidence)
        out.append(span)
    return out


def total_span_ms(words: Sequence[WordSpan]) -> int:
    """整段词级时间轴的跨度。"""
    if not words:
        return 0
    return max(0, words[-1].end_ms - words[0].start_ms)


def to_srt_cues(words: Sequence[WordSpan], *, max_chars: int = 18,
                max_gap_ms: int = 700) -> list[tuple[int, int, str]]:
    """把词级时间戳聚成字幕行。

    Args:
        words: 按时间升序的词区间。
        max_chars: 单行最大字符数。
        max_gap_ms: 相邻词间隔超过该值就断行。

    Returns:
        ``(start_ms, end_ms, text)`` 列表。
    """
    cues: list[tuple[int, int, str]] = []
    if not words:
        return cues
    buf: list[WordSpan] = []
    for span in words:
        if buf:
            gap = span.start_ms - buf[-1].end_ms
            chars = sum(len(w.word) for w in buf)
            if gap > max_gap_ms or chars + len(span.word) > max_chars:
                cues.append((buf[0].start_ms, buf[-1].end_ms,
                             "".join(w.word for w in buf)))
                buf = []
        buf.append(span)
    if buf:
        cues.append((buf[0].start_ms, buf[-1].end_ms, "".join(w.word for w in buf)))
    return cues

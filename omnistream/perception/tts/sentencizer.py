"""流式切句器 —— TTFA（首字发声延迟）的关键决定者。

问题
----
LLM 是一个字一个字吐出来的。如果等它说完整段再合成，TTFA 就等于
"LLM 生成完整段的时间"（往往 2~5 秒），体验完全不可接受。

策略
----
1. **首句激进**：只要攒够 ``first_chunk_chars`` 个字符**并且**遇到任意
   可断点（标点、连词、甚至空格），立刻切出去合成 —— 首句宁短勿长；
2. **后续保守**：按完整句末标点切，保证韵律自然；
3. **超长保护**：单句超过 ``max_chars`` 时在最近的次级断点强行切开，
   避免一句 500 字把播放队列撑爆；
4. **不切碎数字与小数点**：``3.14`` / ``2025.6`` 里的点不作为句末标点；
5. **不切碎缩写**：``Mr.`` ``Dr.`` ``etc.`` 后面的点不断句。

这些规则合起来能把 TTFA 从"整段生成时间"压到"首 8~12 字生成时间"，
通常 150~300ms。
"""
from __future__ import annotations

import re
from typing import Final, Iterator

__all__ = [
    "DEFAULT_FIRST_CHUNK_CHARS",
    "DEFAULT_MAX_CHARS",
    "DEFAULT_MIN_CHARS",
    "SECONDARY_BREAKS",
    "SENTENCE_ENDS",
    "StreamingSentencizer",
    "split_sentences",
]

#: 首句最小字符数
DEFAULT_FIRST_CHUNK_CHARS: Final[int] = 8

#: 非首句最小字符数（太短会导致韵律断裂）
DEFAULT_MIN_CHARS: Final[int] = 4

#: 单句最大字符数，超过强制切
DEFAULT_MAX_CHARS: Final[int] = 80

#: 句末标点
SENTENCE_ENDS: Final[frozenset[str]] = frozenset("。！？!?…；;\n")

#: 次级断点（逗号类），用于首句加速与超长强制切分
SECONDARY_BREAKS: Final[frozenset[str]] = frozenset("，,、：:—－-）)】」』")

#: 英文缩写白名单（其后的句点不断句）
_ABBREVIATIONS: Final[frozenset[str]] = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g",
    "i.e", "fig", "no", "inc", "ltd", "co", "corp", "dept", "univ", "approx",
})

#: 成对引号/括号，闭合前不断句
_OPENERS: Final[str] = "“‘「『（(【[{"
_CLOSERS: Final[str] = "”’」』）)】]}"
_PAIRS: Final[dict[str, str]] = dict(zip(_OPENERS, _CLOSERS, strict=True))

#: 数字上下文中的点（``3.14``）
_DECIMAL_RE: Final[re.Pattern[str]] = re.compile(r"\d\.\d")

#: 英文句点（需要额外判断缩写）
_LATIN_END: Final[str] = "."


class StreamingSentencizer:
    """增量切句器。

    Example::

        s = StreamingSentencizer(first_chunk_chars=8)
        for piece in ["你好", "，今天", "天气不错。", "要出门吗？"]:
            for sentence in s.feed(piece):
                print(sentence)
        for sentence in s.flush():
            print(sentence)
    """

    def __init__(
        self,
        *,
        language: str = "zh-CN",
        first_chunk_chars: int = DEFAULT_FIRST_CHUNK_CHARS,
        min_chars: int = DEFAULT_MIN_CHARS,
        max_chars: int = DEFAULT_MAX_CHARS,
        keep_punctuation: bool = True,
    ) -> None:
        """
        Args:
            language: 语言（英文会启用缩写与空格断点规则）。
            first_chunk_chars: 首句最小长度。
            min_chars: 后续句最小长度。
            max_chars: 单句最大长度。
            keep_punctuation: 切出的句子是否保留末尾标点
                （保留能让 TTS 读出正确语调）。
        """
        self.language = language
        self.first_chunk_chars = max(1, int(first_chunk_chars))
        self.min_chars = max(1, int(min_chars))
        self.max_chars = max(self.min_chars + 1, int(max_chars))
        self.keep_punctuation = bool(keep_punctuation)

        self._buf: list[str] = []
        self._depth: list[str] = []   # 未闭合的开括号/引号栈
        self._emitted = 0
        self.sentences_out = 0
        self.forced_cuts = 0

    # ------------------------------------------------------------------

    @property
    def is_latin(self) -> bool:
        """是否为以空格分词的语言。"""
        code = (self.language or "").split("-")[0].lower()
        return code in ("en", "de", "fr", "es", "it", "pt", "ru", "nl")

    @property
    def pending(self) -> str:
        """当前缓冲中尚未切出的文本。"""
        return "".join(self._buf)

    @property
    def emitted_sentences(self) -> int:
        """已切出的句子数。"""
        return self._emitted

    def _threshold(self) -> int:
        """当前句的最小长度门限。"""
        return self.first_chunk_chars if self._emitted == 0 else self.min_chars

    def _track_depth(self, ch: str) -> None:
        """维护成对符号栈。"""
        if ch in _PAIRS:
            self._depth.append(_PAIRS[ch])
        elif self._depth and ch == self._depth[-1]:
            self._depth.pop()

    def _is_abbrev_dot(self, buf: str) -> bool:
        """判断 ``buf`` 末尾的句点是否属于英文缩写。"""
        tail = buf[:-1].rstrip()
        if not tail:
            return False
        token = re.split(r"[\s(\[\"']", tail)[-1].lower()
        return token in _ABBREVIATIONS or len(token) == 1

    def _cut(self, size: int) -> str:
        """从缓冲头部切出 ``size`` 个字符。"""
        text = "".join(self._buf[:size])
        del self._buf[:size]
        self._emitted += 1
        self.sentences_out += 1
        if not self.keep_punctuation:
            text = text.rstrip("".join(SENTENCE_ENDS | SECONDARY_BREAKS))
        return text.strip()

    def _find_break(self) -> int:
        """在缓冲中找一个可切位置，返回切分长度（0 表示暂不切）。

        Returns:
            要切出的字符数；0 表示继续攒。
        """
        buf = self._buf
        n = len(buf)
        if n == 0:
            return 0
        threshold = self._threshold()

        # 1) 句末标点：满足最小长度即切
        for i in range(n - 1, -1, -1):
            ch = buf[i]
            if ch not in SENTENCE_ENDS and not (ch == _LATIN_END and self.is_latin):
                continue
            if self._depth:
                continue  # 引号/括号未闭合，不切
            if ch == _LATIN_END:
                joined = "".join(buf[: i + 1])
                if _DECIMAL_RE.search(joined[-3:]) or self._is_abbrev_dot(joined):
                    continue
            if i + 1 >= threshold:
                # 把紧随其后的收尾引号一并带上
                end = i + 1
                while end < n and buf[end] in _CLOSERS + "”’\"'":
                    end += 1
                return end

        # 2) 首句加速：攒够长度就在最近的次级断点切
        if self._emitted == 0 and n >= self.first_chunk_chars and not self._depth:
            for i in range(n - 1, threshold - 2, -1):
                if buf[i] in SECONDARY_BREAKS:
                    return i + 1
            if self.is_latin:
                for i in range(n - 1, threshold - 2, -1):
                    if buf[i].isspace():
                        return i + 1

        # 3) 超长强制切
        if n >= self.max_chars:
            self.forced_cuts += 1
            for i in range(n - 1, threshold - 2, -1):
                if buf[i] in SECONDARY_BREAKS or buf[i].isspace():
                    return i + 1
            return n
        return 0

    # ------------------------------------------------------------------

    def feed(self, text: str) -> list[str]:
        """喂入一段文本增量，返回本次切出的完整句子。

        Args:
            text: LLM 吐出的文本片段（可以是单个字）。

        Returns:
            切出的句子列表（可能为空）。
        """
        if not text:
            return []
        out: list[str] = []
        for ch in text:
            self._buf.append(ch)
            self._track_depth(ch)
            size = self._find_break()
            if size > 0:
                piece = self._cut(size)
                if piece:
                    out.append(piece)
        return out

    def flush(self) -> list[str]:
        """输出缓冲中剩余的全部内容（文本流结束时调用）。"""
        if not self._buf:
            return []
        remaining = "".join(self._buf).strip()
        self._buf.clear()
        self._depth.clear()
        if not remaining:
            return []
        self._emitted += 1
        self.sentences_out += 1
        return [remaining]

    def reset(self) -> None:
        """重置为新一轮（首句加速重新生效）。"""
        self._buf.clear()
        self._depth.clear()
        self._emitted = 0

    def stats(self) -> dict[str, int | str]:
        """统计快照。"""
        return {
            "language": self.language,
            "sentences_out": self.sentences_out,
            "forced_cuts": self.forced_cuts,
            "pending_chars": len(self._buf),
        }

    def __repr__(self) -> str:
        return (f"<StreamingSentencizer lang={self.language} "
                f"pending={len(self._buf)} out={self.sentences_out}>")


def split_sentences(
    text: str,
    *,
    language: str = "zh-CN",
    first_chunk_chars: int = DEFAULT_FIRST_CHUNK_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[str]:
    """一次性切句（非流式场景，如缓存预热与离线合成）。

    Args:
        text: 完整文本。
        language: 语言。
        first_chunk_chars: 首句最小长度。
        max_chars: 单句最大长度。

    Returns:
        句子列表。
    """
    sentencizer = StreamingSentencizer(language=language,
                                       first_chunk_chars=first_chunk_chars,
                                       max_chars=max_chars)
    out = sentencizer.feed(text)
    out.extend(sentencizer.flush())
    return [s for s in out if s]


def iter_sentences(text: str, **kwargs: object) -> Iterator[str]:
    """:func:`split_sentences` 的迭代器版本。"""
    yield from split_sentences(text, **kwargs)  # type: ignore[arg-type]

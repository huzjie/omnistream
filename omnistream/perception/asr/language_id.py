"""语种识别与中英混说处理。

真实业务里"纯中文"和"纯英文"都是少数，大量出现的是
「帮我把这个 pipeline 的 latency 压到 200 毫秒」这类**语码转换**（code-switching）。
本模块提供两级能力：

* :meth:`LanguageIdentifier.detect` —— 基于 Unicode 脚本分布 + 常用词命中的
  轻量语种判定，零依赖、微秒级，够用于"给 ASR/TTS 选引擎"这类决策；
* :meth:`LanguageIdentifier.segments` —— 把一句混说文本切成
  ``[zh 段, en 段, zh 段]``，供 TTS 分段选音色、供 ASR 分段送不同引擎。

不追求 176 语种的通用 LID；**只把中/英/日/韩/俄 + 数字符号这几类分清楚**，
这覆盖了实时语音场景 99% 的需求，且不引入 fasttext 之类的模型依赖。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Final

import structlog

log = structlog.get_logger(__name__)

#: 支持判定的语种码（BCP-47 主语言子标签）
SUPPORTED: Final[tuple[str, ...]] = ("zh", "en", "ja", "ko", "ru", "und")

#: 语种码到默认区域码的映射，供 TTS/ASR 使用
DEFAULT_LOCALE: Final[dict[str, str]] = {
    "zh": "zh-CN",
    "en": "en-US",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "ru": "ru-RU",
    "und": "zh-CN",
}

_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_KANA = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
_HANGUL = re.compile(r"[\uac00-\ud7af\u1100-\u11ff]")
_CYRILLIC = re.compile(r"[\u0400-\u04ff]")
_LATIN = re.compile(r"[A-Za-z]")
_DIGIT = re.compile(r"[0-9]")

#: 英文高频功能词，用于在"全拉丁字母"时区分英语与拼音/专名
_EN_STOPWORDS: Final[frozenset[str]] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "in",
    "on", "for", "with", "and", "or", "but", "if", "that", "this", "it", "you",
    "i", "we", "they", "he", "she", "not", "do", "does", "did", "can", "could",
    "will", "would", "should", "have", "has", "had", "what", "how", "why", "when",
    "where", "which", "who", "please", "thanks", "ok", "okay", "yes", "no",
})

#: 中文高频虚词，命中越多越确信是中文
_ZH_MARKERS: Final[frozenset[str]] = frozenset({
    "的", "了", "吗", "呢", "吧", "是", "在", "我", "你", "他", "她", "们", "这",
    "那", "个", "有", "和", "不", "也", "就", "都", "很", "把", "被", "给", "让",
})

_TOKEN_RE = re.compile(r"[A-Za-z']+|[0-9]+(?:\.[0-9]+)?|[\u4e00-\u9fff]|[^\sA-Za-z0-9\u4e00-\u9fff]")


@dataclass(slots=True, frozen=True)
class LanguageGuess:
    """一次语种判定结果。"""

    code: str = "und"
    confidence: float = 0.0
    #: 各脚本字符占比
    ratios: dict[str, float] = field(default_factory=dict)
    #: 是否存在明显的语码转换
    code_switched: bool = False

    @property
    def locale(self) -> str:
        """默认区域码，例如 ``zh-CN``。"""
        return DEFAULT_LOCALE.get(self.code, "zh-CN")

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "locale": self.locale,
            "confidence": round(self.confidence, 3),
            "code_switched": self.code_switched,
            "ratios": {k: round(v, 3) for k, v in self.ratios.items()},
        }


@dataclass(slots=True, frozen=True)
class LanguageSpan:
    """混说文本中的一个同语种片段。"""

    text: str
    code: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def normalize_code(value: str) -> str:
    """把 ``zh-CN`` / ``zh_Hans_CN`` / ``ZH`` 归一成 ``zh``。"""
    if not value:
        return "und"
    head = re.split(r"[-_]", value.strip())[0].lower()
    return head if head in SUPPORTED else ("zh" if head.startswith("zh") else head or "und")


class LanguageIdentifier:
    """轻量语种识别器。"""

    def __init__(self, *, default: str = "zh", min_chars: int = 2,
                 switch_threshold: float = 0.18) -> None:
        self.default = normalize_code(default)
        self.min_chars = int(min_chars)
        #: 次要语种占比超过该阈值即认定为混说
        self.switch_threshold = float(switch_threshold)

    # ------------------------------------------------------------------
    # 判定
    # ------------------------------------------------------------------

    def detect(self, text: str) -> LanguageGuess:
        """判定整段文本的主语种。"""
        content = (text or "").strip()
        if len(content) < self.min_chars:
            return LanguageGuess(code=self.default, confidence=0.2, ratios={})

        counts = {
            "cjk": len(_CJK.findall(content)),
            "kana": len(_KANA.findall(content)),
            "hangul": len(_HANGUL.findall(content)),
            "cyrillic": len(_CYRILLIC.findall(content)),
            "latin": len(_LATIN.findall(content)),
            "digit": len(_DIGIT.findall(content)),
        }
        letters = sum(counts.values())
        if letters == 0:
            return LanguageGuess(code=self.default, confidence=0.2, ratios={})

        ratios = {k: v / letters for k, v in counts.items()}
        scores: dict[str, float] = {
            "ja": ratios["kana"] * 1.6 + ratios["cjk"] * 0.35,
            "ko": ratios["hangul"] * 1.8,
            "ru": ratios["cyrillic"] * 1.8,
            "zh": ratios["cjk"] * 1.5 + self._marker_bonus(content),
            "en": ratios["latin"] * 1.1 + self._stopword_bonus(content),
        }
        # 假名存在时中文得分需要削弱（日文里也有大量汉字）
        if counts["kana"] > 0:
            scores["zh"] *= 0.35

        code = max(scores, key=lambda k: scores[k])
        best = scores[code]
        if best <= 0.05:
            code, best = self.default, 0.25

        ordered = sorted(scores.values(), reverse=True)
        margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
        confidence = float(max(0.0, min(1.0, 0.45 + 0.55 * margin)))

        minor = self._minor_ratio(ratios, code)
        return LanguageGuess(
            code=code,
            confidence=confidence,
            ratios=ratios,
            code_switched=minor >= self.switch_threshold,
        )

    def _marker_bonus(self, text: str) -> float:
        hits = sum(1 for ch in text if ch in _ZH_MARKERS)
        return min(0.4, hits * 0.06)

    def _stopword_bonus(self, text: str) -> float:
        tokens = [t.lower() for t in re.findall(r"[A-Za-z']+", text)]
        if not tokens:
            return 0.0
        hits = sum(1 for t in tokens if t in _EN_STOPWORDS)
        return min(0.45, 0.45 * hits / len(tokens))

    @staticmethod
    def _minor_ratio(ratios: dict[str, float], code: str) -> float:
        """主语种之外的"另一种文字"的占比。"""
        if code == "zh":
            return ratios.get("latin", 0.0)
        if code == "en":
            return ratios.get("cjk", 0.0)
        return max(ratios.get("latin", 0.0), ratios.get("cjk", 0.0))

    # ------------------------------------------------------------------
    # 混说切分
    # ------------------------------------------------------------------

    def segments(self, text: str, *, min_span: int = 2) -> list[LanguageSpan]:
        """把混说文本切成同语种片段。

        Args:
            text: 原始文本。
            min_span: 小于该长度的片段会被并入相邻片段，避免碎片化。

        Returns:
            按出现顺序排列的 :class:`LanguageSpan` 列表。
        """
        content = text or ""
        if not content:
            return []

        raw: list[tuple[int, int, str]] = []
        for match in _TOKEN_RE.finditer(content):
            token = match.group(0)
            code = self._token_language(token)
            if raw and (raw[-1][2] == code or code == "und"):
                start, _end, prev_code = raw[-1]
                raw[-1] = (start, match.end(), prev_code)
            else:
                raw.append((match.start(), match.end(), code))

        spans: list[LanguageSpan] = []
        for start, end, code in raw:
            resolved = code if code != "und" else (spans[-1].code if spans else self.default)
            piece = content[start:end]
            if spans and (spans[-1].code == resolved or len(piece.strip()) < min_span):
                prev = spans[-1]
                spans[-1] = LanguageSpan(text=content[prev.start:end], code=prev.code,
                                         start=prev.start, end=end)
            else:
                spans.append(LanguageSpan(text=piece, code=resolved, start=start, end=end))
        return spans

    def _token_language(self, token: str) -> str:
        if _CJK.search(token):
            return "zh"
        if _KANA.search(token):
            return "ja"
        if _HANGUL.search(token):
            return "ko"
        if _CYRILLIC.search(token):
            return "ru"
        if _LATIN.search(token):
            return "en"
        return "und"

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def is_code_switched(self, text: str) -> bool:
        """是否为中英（或其他）混说。"""
        return self.detect(text).code_switched

    @staticmethod
    def strip_accents(text: str) -> str:
        """去掉变音符号，便于做粗粒度匹配。"""
        normalized = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")

    def __repr__(self) -> str:
        return f"<LanguageIdentifier default={self.default}>"


#: 进程级共享实例（无状态，线程安全）
DEFAULT_IDENTIFIER: Final[LanguageIdentifier] = LanguageIdentifier()


def detect_language(text: str) -> LanguageGuess:
    """便捷函数：用默认识别器判定语种。"""
    return DEFAULT_IDENTIFIER.detect(text)


__all__ = [
    "DEFAULT_IDENTIFIER",
    "DEFAULT_LOCALE",
    "SUPPORTED",
    "LanguageGuess",
    "LanguageIdentifier",
    "LanguageSpan",
    "detect_language",
    "normalize_code",
]

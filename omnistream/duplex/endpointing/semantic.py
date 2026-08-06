"""语义端点：用 ASR 增量文本的"完整度"避免半句被截断。

纯规则实现（零模型、零延迟），覆盖中英文最常见的"话没说完"信号：

1. **悬挂连词/介词结尾**：「然后」「但是」「因为」「and」「but」「because」
2. **量词/数词悬挂**：「三个」「第」「一共」「about」
3. **未闭合括号/引号**
4. **无终止标点且长度不足**
5. **填充词结尾**：「嗯」「那个」「就是」「uh」「um」「like」

输出：

* :meth:`completeness` → [0,1]，1 表示很可能是完整句
* :meth:`extra_wait_ms` → 需要额外等待的毫秒数（0 ~ 600）
"""
from __future__ import annotations

from typing import Final

#: 中文悬挂结尾词（出现在末尾几乎必然还有后续）
_ZH_DANGLING: Final[tuple[str, ...]] = (
    "然后", "但是", "可是", "不过", "因为", "所以", "而且", "并且", "如果", "假如", "虽然",
    "于是", "接着", "另外", "还有", "以及", "或者", "要么", "既然", "除了", "关于", "对于",
    "由于", "为了", "的", "地", "得", "和", "跟", "与", "在", "把", "被", "让", "从", "到",
    "给", "对", "向", "往", "比", "第", "一共", "总共", "大概", "大约",
)

#: 中文填充/思考词
_ZH_FILLER: Final[tuple[str, ...]] = ("嗯", "呃", "啊", "那个", "这个", "就是", "然后呢", "怎么说")

#: 英文悬挂结尾词
_EN_DANGLING: Final[tuple[str, ...]] = (
    "and", "but", "or", "so", "because", "although", "though", "while", "if", "when",
    "the", "a", "an", "of", "to", "for", "with", "in", "on", "at", "by", "from", "about",
    "that", "which", "who", "as", "than", "into", "onto", "over", "under", "is", "are",
    "was", "were", "will", "would", "can", "could", "should", "my", "your", "his", "her",
)

_EN_FILLER: Final[tuple[str, ...]] = ("uh", "um", "er", "hmm", "like", "you know", "i mean")

#: 终止标点
_TERMINALS: Final[str] = "。！？.!?；;…"

_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("（", "）"), ("(", ")"), ("「", "」"), ("『", "』"), ("《", "》"),
    ("[", "]"), ("{", "}"),
)

_QUOTES: Final[tuple[str, ...]] = ('"', "'", "“", "”", "‘", "’")


class SemanticEndpointer:
    """基于规则的语义端点判定器。"""

    def __init__(
        self,
        *,
        min_complete_chars: int = 4,
        max_extra_wait_ms: int = 600,
        dangling_wait_ms: int = 420,
        filler_wait_ms: int = 300,
        unbalanced_wait_ms: int = 260,
        short_wait_ms: int = 180,
    ) -> None:
        self.min_complete_chars = int(min_complete_chars)
        self.max_extra_wait_ms = int(max_extra_wait_ms)
        self.dangling_wait_ms = int(dangling_wait_ms)
        self.filler_wait_ms = int(filler_wait_ms)
        self.unbalanced_wait_ms = int(unbalanced_wait_ms)
        self.short_wait_ms = int(short_wait_ms)

    # ------------------------------------------------------------------

    def completeness(self, text: str, *, language: str = "zh-CN") -> float:
        """估计文本作为完整语句的置信度 [0,1]。"""
        clean = (text or "").strip()
        if not clean:
            return 0.0

        score = 0.5
        if clean[-1] in _TERMINALS:
            score += 0.42
        if len(clean) >= self.min_complete_chars:
            score += 0.12
        else:
            score -= 0.22

        if self._ends_with_dangling(clean, language):
            score -= 0.45
        if self._ends_with_filler(clean, language):
            score -= 0.30
        if not self._is_balanced(clean):
            score -= 0.35
        if clean[-1] in ("，", ",", "、"):
            score -= 0.18

        return float(max(0.0, min(1.0, score)))

    def extra_wait_ms(self, text: str, *, language: str = "zh-CN") -> int:
        """需要额外等待的毫秒数。完整句返回 0。"""
        clean = (text or "").strip()
        if not clean:
            return 0
        if clean[-1] in _TERMINALS:
            return 0

        wait = 0
        if self._ends_with_dangling(clean, language):
            wait += self.dangling_wait_ms
        if self._ends_with_filler(clean, language):
            wait += self.filler_wait_ms
        if not self._is_balanced(clean):
            wait += self.unbalanced_wait_ms
        if len(clean) < self.min_complete_chars:
            wait += self.short_wait_ms
        if clean[-1] in ("，", ",", "、"):
            wait += 160

        return int(min(self.max_extra_wait_ms, wait))

    def is_complete(self, text: str, *, language: str = "zh-CN", threshold: float = 0.6) -> bool:
        """便捷判定：完整度是否达到阈值。"""
        return self.completeness(text, language=language) >= threshold

    # ------------------------------------------------------------------

    @staticmethod
    def _ends_with_dangling(text: str, language: str) -> bool:
        if language.lower().startswith("en"):
            words = text.lower().replace(",", " ").split()
            return bool(words) and words[-1].strip(".,;:") in _EN_DANGLING
        tail = text[-3:]
        if any(tail.endswith(tok) for tok in _ZH_DANGLING):
            return True
        # 中英混说：末尾是英文悬挂词也算
        words = text.lower().split()
        return bool(words) and words[-1].strip(".,;:") in _EN_DANGLING

    @staticmethod
    def _ends_with_filler(text: str, language: str) -> bool:
        lowered = text.lower()
        if language.lower().startswith("en"):
            return any(lowered.endswith(tok) for tok in _EN_FILLER)
        return any(text.endswith(tok) for tok in _ZH_FILLER) or any(
            lowered.endswith(tok) for tok in _EN_FILLER
        )

    @staticmethod
    def _is_balanced(text: str) -> bool:
        for left, right in _PAIRS:
            if text.count(left) != text.count(right):
                return False
        for quote in _QUOTES:
            if quote in ('"', "'") and text.count(quote) % 2 == 1:
                return False
        if text.count("“") != text.count("”"):
            return False
        return text.count("‘") == text.count("’")

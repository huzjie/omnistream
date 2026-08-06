"""自适应尾静音阈值。

固定 500ms 尾静音是全双工体验的最大杀手之一：语速快的人觉得慢，
边想边说的人被半句截断。本模块按四个信号在 [min_ms, max_ms] 区间内动态调整：

============  ================================================  ==========
信号           判断                                              调整
============  ================================================  ==========
语速           字/秒越高 → 说话人连贯 → 可以更快收尾              -120 ~ +160ms
话语长度       极短（< 700ms）多为"嗯/好的"或思考停顿 → 多等       +0 ~ +200ms
疑问句         以疑问词/问号结尾 → 用户在等答复 → 尽快收尾        -140ms
历史被打断率   上一轮刚被用户"补话"→ 说明收尾过早 → 加惩罚         +0 ~ +250ms
============  ================================================  ==========

所有阈值最终裁剪到 [300, 1200] ms（设计文档 §2.19 约定）。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

_ZH_QUESTION_TOKENS: tuple[str, ...] = (
    "吗", "呢", "吧", "么", "?", "？", "如何", "怎么", "怎样", "为什么", "多少", "哪个", "哪些",
    "是不是", "有没有", "能不能", "可不可以",
)
_EN_QUESTION_TOKENS: tuple[str, ...] = (
    "?", "what", "why", "how", "when", "where", "who", "which", "can you", "could you",
    "is it", "are you", "do you", "does it",
)


@dataclass(slots=True)
class TurnFeedback:
    """一轮结束后的反馈，用于在线调参。"""

    speech_duration_ms: float
    text_chars: int
    premature_cut: bool = False
    interrupted: bool = False


class AdaptiveSilence:
    """自适应尾静音阈值计算器。"""

    def __init__(
        self,
        *,
        base_ms: int = 480,
        min_ms: int = 300,
        max_ms: int = 1200,
        history_size: int = 12,
    ) -> None:
        self.base_ms = int(base_ms)
        self.min_ms = int(min_ms)
        self.max_ms = int(max_ms)
        self._history: deque[TurnFeedback] = deque(maxlen=max(1, int(history_size)))
        self._premature_penalty_ms = 0.0

    # ------------------------------------------------------------------

    def observe_turn(self, feedback: TurnFeedback) -> None:
        """记录一轮反馈。检测到过早截断时逐步加惩罚，平稳时逐步衰减。"""
        self._history.append(feedback)
        if feedback.premature_cut:
            self._premature_penalty_ms = min(250.0, self._premature_penalty_ms + 90.0)
        else:
            self._premature_penalty_ms = max(0.0, self._premature_penalty_ms - 30.0)

    def compute(self, *, speech_duration_ms: float, partial_text: str, language: str = "zh-CN") -> int:
        """计算当前话语应使用的尾静音阈值（毫秒）。"""
        value = float(self.base_ms)

        value += self._speech_rate_adjust(speech_duration_ms, partial_text)
        value += self._length_adjust(speech_duration_ms)
        value += self._question_adjust(partial_text, language)
        value += self._premature_penalty_ms
        value += self._history_adjust()

        return int(max(self.min_ms, min(self.max_ms, round(value))))

    def reset(self) -> None:
        self._history.clear()
        self._premature_penalty_ms = 0.0

    # ------------------------------------------------------------------

    @staticmethod
    def _speech_rate_adjust(speech_duration_ms: float, partial_text: str) -> float:
        """语速快 → 减少等待；语速慢（边想边说）→ 增加等待。"""
        chars = len(partial_text.strip())
        if chars < 2 or speech_duration_ms < 300.0:
            return 0.0
        cps = chars / (speech_duration_ms / 1000.0)
        # 中文自然语速约 4~6 字/秒；以 4.5 为中枢
        delta = (cps - 4.5) * -60.0
        return float(max(-120.0, min(160.0, delta)))

    @staticmethod
    def _length_adjust(speech_duration_ms: float) -> float:
        if speech_duration_ms < 400.0:
            return 200.0
        if speech_duration_ms < 700.0:
            return 110.0
        if speech_duration_ms > 4000.0:
            return -60.0
        return 0.0

    @staticmethod
    def _question_adjust(partial_text: str, language: str) -> float:
        text = partial_text.strip().lower()
        if not text:
            return 0.0
        tokens = _EN_QUESTION_TOKENS if language.lower().startswith("en") else _ZH_QUESTION_TOKENS
        tail = text[-6:]
        if any(tok in tail for tok in tokens) or any(text.startswith(tok) for tok in tokens):
            return -140.0
        return 0.0

    def _history_adjust(self) -> float:
        """连续多轮被打断说明助手话太长/收尾太慢，略微收紧尾静音。"""
        if len(self._history) < 3:
            return 0.0
        interrupted = sum(1 for f in self._history if f.interrupted)
        ratio = interrupted / len(self._history)
        if ratio > 0.5:
            return -80.0
        if ratio > 0.3:
            return -40.0
        return 0.0

    @property
    def premature_penalty_ms(self) -> float:
        return self._premature_penalty_ms

    @property
    def history_size(self) -> int:
        return len(self._history)

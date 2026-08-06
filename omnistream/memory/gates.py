"""三闸门选择性注入策略（Injection Gates）。

对齐 Meta 主动记忆 Agent 的设计：
1. relevance gate    —— 记忆与当前上下文的相关性是否超过阈值
2. confidence gate   —— 记忆置信度是否足够（避免过度自信的推测注入）
3. timing gate       —— 是否处于可注入时机（会话注入配额未耗尽、非打断时机）

三闸门全过 → 注入"一条定向提醒"；否则保持沉默。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from omnistream.memory.layers import MemoryItem

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class InjectionDecision:
    """注入决策结果。"""
    inject: bool
    reason: str = ""
    reminder: str | None = None
    matched_items: list[MemoryItem] = field(default_factory=list)
    relevance: float = 0.0
    confidence: float = 0.0
    timing_ok: bool = False


class InjectionGate:
    """三闸门组合闸门。"""

    def __init__(
        self,
        relevance_threshold: float = 0.55,
        confidence_threshold: float = 0.60,
        max_injections_per_turn: int = 1,
        max_injections_per_session: int = 6,
        min_relevance_delta: float = 0.05,
    ) -> None:
        self.relevance_threshold = relevance_threshold
        self.confidence_threshold = confidence_threshold
        self.max_injections_per_turn = max_injections_per_turn
        self.max_injections_per_session = max_injections_per_session
        self.min_relevance_delta = min_relevance_delta

    async def evaluate(
        self,
        *,
        context: str,
        candidates: list[MemoryItem],
        injected_this_session: int,
        injected_this_turn: int = 0,
    ) -> InjectionDecision:
        """对候选记忆做三闸门评估。"""
        # ---- 0. 候选为空 → 沉默 ----
        if not candidates:
            return InjectionDecision(inject=False, reason="no_candidates")

        # ---- 1. Timing gate：会话配额 ----
        if injected_this_session >= self.max_injections_per_session:
            return InjectionDecision(inject=False, reason="session_quota_exhausted")
        if injected_this_turn >= self.max_injections_per_turn:
            return InjectionDecision(inject=False, reason="turn_quota_exhausted")

        # ---- 2. Relevance + Confidence gate：择优 ----
        # 用排序后的相似度（已在 retriever 中乘以 confidence 打分，这里再独立校验）
        top = candidates[0]
        relevance = self._relevance_score(context, top)
        if relevance < self.relevance_threshold:
            return InjectionDecision(
                inject=False, reason="relevance_below_threshold",
                relevance=relevance, confidence=top.confidence)
        if top.confidence < self.confidence_threshold:
            return InjectionDecision(
                inject=False, reason="confidence_below_threshold",
                relevance=relevance, confidence=top.confidence)

        # ---- 3. 相对优势：top1 与 top2 差距过小 → 记忆不稳定，宁沉默 ----
        if len(candidates) >= 2:
            second_relevance = self._relevance_score(context, candidates[1])
            if relevance - second_relevance < self.min_relevance_delta:
                return InjectionDecision(
                    inject=False, reason="ambiguous_top_results",
                    relevance=relevance, confidence=top.confidence)

        # ---- 通过：构造定向提醒 ----
        reminder = self._build_reminder(top)
        return InjectionDecision(
            inject=True,
            reason="gates_passed",
            reminder=reminder,
            matched_items=[top],
            relevance=relevance,
            confidence=top.confidence,
            timing_ok=True,
        )

    @staticmethod
    def _relevance_score(context: str, item: MemoryItem) -> float:
        """轻量相关性评估：上下文与记忆文本的词重叠（Jaccard 变体）。"""
        ctx_tokens = set(_tokenize(context))
        mem_tokens = set(_tokenize(item.text))
        if not ctx_tokens or not mem_tokens:
            return 0.0
        inter = len(ctx_tokens & mem_tokens)
        union = len(ctx_tokens | mem_tokens)
        return inter / union if union else 0.0

    @staticmethod
    def _build_reminder(item: MemoryItem) -> str:
        """构造一条不超过 1 句的定向提醒。"""
        prefix = {
            "session": "会话提醒",
            "private_state": "重要偏好",
            "stable_knowledge": "知识回顾",
            "procedural": "执行要点",
        }.get(item.layer.value, "记忆")
        return f"[{prefix}] {item.text}"


def _tokenize(text: str) -> list[str]:
    import re

    text = text.lower()
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9_]+", text)

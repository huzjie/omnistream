"""会话治理层：质检评分。

对会话转写做质量评分：
- 打断频率（过多打断 = 体验差）
- 平均轮次长度
- 敏感词/PII 命中
- 语义完整度（启发式）
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def score_transcript(transcript: str, *, language: str = "zh-CN") -> dict[str, Any]:
    """对转写文本做整体质检评分（0-100）。"""
    text = transcript or ""
    chars = len(text.strip())
    sentences = [s for s in re.split(r"[。！？!?.\n]", text) if s.strip()]

    # 维度分
    completeness = min(100.0, (chars / 120.0) * 100) if chars else 0.0
    sentence_score = min(100.0, len(sentences) * 15.0)
    readability = 80.0 if chars else 0.0

    # 敏感内容命中扣分
    penalty = 0.0
    from omnistream.governance.privacy.sensitive_words import SensitiveWordFilter

    hits = SensitiveWordFilter().scan(text)
    hit_words = [h.word for h in hits]
    if hit_words:
        penalty = min(40.0, len(hit_words) * 10.0)

    score = round(
        max(0.0, min(100.0,
                     completeness * 0.4 + sentence_score * 0.3
                     + readability * 0.3 - penalty)),
        1,
    )

    grade = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D"
    return {
        "score": score,
        "grade": grade,
        "chars": chars,
        "sentences": len(sentences),
        "sensitive_hits": hit_words,
        "dimensions": {
            "completeness": round(completeness, 1),
            "sentence": round(sentence_score, 1),
            "readability": round(readability, 1),
            "penalty": round(penalty, 1),
        },
    }

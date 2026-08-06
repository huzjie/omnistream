"""增量结果稳定器：LocalAgreement-2。

流式 ASR 的原始输出会不停"反悔"——上一帧还是"今天天气"，下一帧变成
"金天天汽"，再下一帧又改回来。直接上屏会让用户看到文字疯狂跳动。

**LocalAgreement-n**（Liu et al., 2020；whisper-streaming 采用的策略）：
只有**连续 n 次假设都同意**的前缀才被"提交"（committed），
提交过的文本永不回退；未提交的部分作为低稳定度的"草稿"（draft）展示，
前端可以用灰色/斜体渲染。

本实现是 n=2 的版本，并做了三点工程增强：

1. **按字符对齐**（中文没有空格，按词对齐不适用），
   同时保留词边界信息用于 word timing 对齐；
2. **回退保护**：新假设比已提交前缀还短时不回退，只是暂不增长；
3. **强制提交**：``max_pending_chars`` / ``max_pending_ms`` 超限时，
   即便没达成一致也提交最长公共前缀的一部分，避免"永远在草稿区"。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import structlog

from omnistream.core.clock import now_ns

log = structlog.get_logger(__name__)

#: 默认一致性窗口（LocalAgreement-n 的 n）
DEFAULT_AGREEMENT: Final[int] = 2

#: 草稿区超过该字符数时强制提交
DEFAULT_MAX_PENDING_CHARS: Final[int] = 24

#: 草稿区停留超过该时长（毫秒）时强制提交
DEFAULT_MAX_PENDING_MS: Final[int] = 2500


@dataclass(slots=True)
class StabilizedText:
    """稳定化后的一次输出。"""

    #: 已提交、永不回退的前缀
    committed: str = ""
    #: 尚未达成一致的草稿
    draft: str = ""
    #: 本次新提交的增量（可能为空）
    newly_committed: str = ""
    #: 稳定度 [0,1]：committed 占全文的比例
    stability: float = 1.0
    #: 是否发生了强制提交
    forced: bool = False

    @property
    def text(self) -> str:
        """完整文本 = 已提交 + 草稿。"""
        return self.committed + self.draft

    def as_dict(self) -> dict[str, object]:
        return {
            "committed": self.committed,
            "draft": self.draft,
            "newly_committed": self.newly_committed,
            "stability": round(self.stability, 3),
            "forced": self.forced,
        }


def common_prefix(a: str, b: str) -> str:
    """两个字符串的最长公共前缀。"""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]


def common_prefix_of(texts: list[str]) -> str:
    """一组字符串的最长公共前缀。"""
    if not texts:
        return ""
    prefix = texts[0]
    for text in texts[1:]:
        prefix = common_prefix(prefix, text)
        if not prefix:
            break
    return prefix


class LocalAgreementStabilizer:
    """LocalAgreement-n 增量稳定器。"""

    def __init__(
        self,
        *,
        agreement: int = DEFAULT_AGREEMENT,
        max_pending_chars: int = DEFAULT_MAX_PENDING_CHARS,
        max_pending_ms: int = DEFAULT_MAX_PENDING_MS,
        force_ratio: float = 0.6,
    ) -> None:
        self.agreement = max(1, int(agreement))
        self.max_pending_chars = int(max_pending_chars)
        self.max_pending_ms = int(max_pending_ms)
        #: 强制提交时提交草稿的比例
        self.force_ratio = float(min(1.0, max(0.1, force_ratio)))

        self._committed = ""
        self._history: list[str] = []
        self._pending_since_ns = 0
        self._forced_count = 0
        self._updates = 0

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def committed(self) -> str:
        """已提交前缀。"""
        return self._committed

    @property
    def forced_count(self) -> int:
        return self._forced_count

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def update(self, hypothesis: str) -> StabilizedText:
        """喂入一条新假设，返回稳定化结果。

        Args:
            hypothesis: 引擎给出的**完整** utterance 假设（不是增量）。

        Returns:
            :class:`StabilizedText`
        """
        self._updates += 1
        text = hypothesis or ""

        # 回退保护：新假设短于已提交前缀时，保留已提交内容。
        if not text.startswith(self._committed):
            merged = common_prefix(text, self._committed)
            if len(merged) < len(self._committed):
                log.debug("asr.stabilizer_rollback_blocked",
                          committed_len=len(self._committed), new_len=len(text))
                text = self._committed + text[len(merged):]

        self._history.append(text)
        while len(self._history) > self.agreement:
            self._history.pop(0)

        agreed = common_prefix_of(self._history) if len(self._history) >= self.agreement else ""
        newly = ""
        forced = False

        if len(agreed) > len(self._committed):
            newly = agreed[len(self._committed):]
            self._committed = agreed
            self._pending_since_ns = now_ns()

        draft = text[len(self._committed):] if text.startswith(self._committed) else ""

        if draft and self._should_force(draft):
            take = max(1, int(len(draft) * self.force_ratio))
            forced_part = draft[:take]
            self._committed += forced_part
            newly += forced_part
            draft = draft[take:]
            forced = True
            self._forced_count += 1
            self._pending_since_ns = now_ns()
            log.debug("asr.stabilizer_forced_commit", chars=take,
                      committed_len=len(self._committed))

        if self._pending_since_ns == 0:
            self._pending_since_ns = now_ns()

        total = len(self._committed) + len(draft)
        stability = 1.0 if total == 0 else len(self._committed) / total
        return StabilizedText(
            committed=self._committed,
            draft=draft,
            newly_committed=newly,
            stability=stability,
            forced=forced,
        )

    def _should_force(self, draft: str) -> bool:
        if len(draft) >= self.max_pending_chars:
            return True
        if self._pending_since_ns == 0:
            return False
        elapsed_ms = (now_ns() - self._pending_since_ns) / 1e6
        return elapsed_ms >= self.max_pending_ms

    def finalize(self, hypothesis: str | None = None) -> StabilizedText:
        """收尾：把剩余草稿全部提交。"""
        text = hypothesis if hypothesis is not None else (
            self._history[-1] if self._history else self._committed)
        if not text.startswith(self._committed):
            text = self._committed + text
        newly = text[len(self._committed):]
        self._committed = text
        return StabilizedText(
            committed=self._committed,
            draft="",
            newly_committed=newly,
            stability=1.0,
            forced=False,
        )

    def reset(self) -> None:
        """新一轮开始时复位。"""
        self._committed = ""
        self._history.clear()
        self._pending_since_ns = 0

    def stats(self) -> dict[str, object]:
        return {
            "updates": self._updates,
            "forced_commits": self._forced_count,
            "committed_chars": len(self._committed),
            "agreement": self.agreement,
        }

    def __repr__(self) -> str:
        return (f"<LocalAgreementStabilizer n={self.agreement} "
                f"committed={len(self._committed)}ch forced={self._forced_count}>")


@dataclass(slots=True)
class SegmentStabilizer:
    """多段落场景的稳定器容器：每个 utterance 一个独立稳定器。"""

    agreement: int = DEFAULT_AGREEMENT
    _by_key: dict[str, LocalAgreementStabilizer] = field(default_factory=dict)

    def get(self, key: str) -> LocalAgreementStabilizer:
        """按 key（通常是 turn_id）取稳定器，不存在则创建。"""
        stabilizer = self._by_key.get(key)
        if stabilizer is None:
            stabilizer = LocalAgreementStabilizer(agreement=self.agreement)
            self._by_key[key] = stabilizer
        return stabilizer

    def drop(self, key: str) -> None:
        """轮次结束后释放。"""
        self._by_key.pop(key, None)

    def clear(self) -> None:
        self._by_key.clear()


__all__ = [
    "DEFAULT_AGREEMENT",
    "DEFAULT_MAX_PENDING_CHARS",
    "DEFAULT_MAX_PENDING_MS",
    "LocalAgreementStabilizer",
    "SegmentStabilizer",
    "StabilizedText",
    "common_prefix",
    "common_prefix_of",
]

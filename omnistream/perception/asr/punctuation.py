"""轻量标点恢复（规则 + 可选小模型）。

多数流式 ASR 引擎（尤其是本地小模型）输出的是"裸文本"：
``今天天气怎么样明天呢`` —— 既影响可读性，也影响下游 LLM 的断句理解。

两级策略：

* **规则层（默认，零依赖）** —— 综合三类信号：
  1. **停顿时长**：VAD/词级时间戳给出的静音间隔，> ``comma_pause_ms`` 插逗号，
     > ``period_pause_ms`` 插句号；
  2. **疑问线索**：中文句末语气词（吗/呢/吧）、疑问代词（什么/怎么/为什么/哪/谁/几/多少），
     英文疑问词开头或助动词倒装；
  3. **连接词**：但是/然后/所以/因为/如果…… 前面补逗号（长句才补，短句不补）。
* **模型层（可选）** —— 若安装了 ``funasr`` 且配置了 ct-punc 模型，
  则调用模型；失败自动回落规则层，绝不影响主链路。

规则层是**幂等**的：已经带标点的文本再跑一遍不会重复加标点。
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Final

import structlog

log = structlog.get_logger(__name__)

#: 中文标点
ZH_COMMA: Final[str] = "，"
ZH_PERIOD: Final[str] = "。"
ZH_QUESTION: Final[str] = "？"
ZH_EXCLAIM: Final[str] = "！"

#: 所有被视为"终结"的标点
TERMINALS: Final[frozenset[str]] = frozenset("。！？.!?…")

#: 所有被视为"停顿"的标点
PAUSES: Final[frozenset[str]] = frozenset("，、,;；:：")

ALL_PUNCT: Final[frozenset[str]] = TERMINALS | PAUSES | frozenset("\"'“”‘’()（）《》")

#: 停顿超过该毫秒数插逗号
DEFAULT_COMMA_PAUSE_MS: Final[int] = 260

#: 停顿超过该毫秒数插句号
DEFAULT_PERIOD_PAUSE_MS: Final[int] = 700

#: 中文句末疑问语气词
_ZH_QUESTION_TAILS: Final[tuple[str, ...]] = ("吗", "呢", "吧", "么", "没有", "不")

#: 中文疑问代词
_ZH_QUESTION_WORDS: Final[tuple[str, ...]] = (
    "什么", "怎么", "为什么", "为何", "哪里", "哪儿", "哪个", "哪些", "谁",
    "多少", "几点", "几个", "如何", "是否", "能不能", "可不可以", "行不行",
)

#: 英文疑问词与助动词
_EN_QUESTION_HEADS: Final[tuple[str, ...]] = (
    "what", "why", "how", "when", "where", "which", "who", "whom", "whose",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "will",
    "would", "should", "shall", "may", "might", "have", "has", "had",
)

#: 中文连接词（其前应有逗号）
_ZH_CONNECTIVES: Final[tuple[str, ...]] = (
    "但是", "不过", "然后", "所以", "因为", "如果", "虽然", "而且", "并且",
    "否则", "另外", "同时", "接着", "于是", "因此", "可是",
)

#: 感叹线索
_ZH_EXCLAIM_CUES: Final[tuple[str, ...]] = ("太好了", "真棒", "厉害", "糟糕", "天哪", "不会吧")

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_TRAILING_SPACE_RE = re.compile(r"\s+([，。？！、；：])")


@dataclass(slots=True)
class PauseMark:
    """一个停顿标记：``char_index`` 之后有 ``pause_ms`` 的静音。"""

    char_index: int
    pause_ms: float


class PunctuationRestorer:
    """标点恢复器。"""

    def __init__(
        self,
        *,
        language: str = "zh",
        comma_pause_ms: int = DEFAULT_COMMA_PAUSE_MS,
        period_pause_ms: int = DEFAULT_PERIOD_PAUSE_MS,
        min_clause_chars: int = 4,
        use_model: bool = False,
        model_dir: str = "",
    ) -> None:
        self.language = language
        self.comma_pause_ms = int(comma_pause_ms)
        self.period_pause_ms = int(period_pause_ms)
        self.min_clause_chars = int(min_clause_chars)
        self.use_model = bool(use_model)
        self.model_dir = model_dir

        self._model: Any = None
        self._model_failed = False

    # ------------------------------------------------------------------
    # 规则层
    # ------------------------------------------------------------------

    def restore(self, text: str, *, pauses: list[PauseMark] | None = None,
                is_final: bool = True) -> str:
        """恢复标点（纯规则，同步、微秒级）。

        Args:
            text: 裸文本。
            pauses: 可选的停顿标记列表（来自词级时间戳）。
            is_final: 是否为最终结果；非最终结果不补句末标点，
                避免用户看到"句号又被删掉"的抖动。

        Returns:
            带标点的文本。
        """
        content = (text or "").strip()
        if not content:
            return ""

        content = self._insert_pause_punct(content, pauses or [])
        content = self._insert_connective_commas(content)
        content = self._collapse(content)
        if is_final:
            content = self._append_terminal(content)
        return content

    def _insert_pause_punct(self, text: str, pauses: list[PauseMark]) -> str:
        if not pauses:
            return text
        chars = list(text)
        # 从后往前插入，避免索引偏移
        for mark in sorted(pauses, key=lambda m: m.char_index, reverse=True):
            idx = int(mark.char_index)
            if idx <= 0 or idx >= len(chars):
                continue
            if chars[idx - 1] in ALL_PUNCT:
                continue
            if mark.pause_ms >= self.period_pause_ms:
                chars.insert(idx, ZH_PERIOD)
            elif mark.pause_ms >= self.comma_pause_ms:
                left = "".join(chars[:idx])
                if self._last_clause_len(left) >= self.min_clause_chars:
                    chars.insert(idx, ZH_COMMA)
        return "".join(chars)

    def _insert_connective_commas(self, text: str) -> str:
        result = text
        for word in _ZH_CONNECTIVES:
            start = 0
            while True:
                idx = result.find(word, start)
                if idx < 0:
                    break
                start = idx + len(word)
                if idx == 0:
                    continue
                if result[idx - 1] in ALL_PUNCT:
                    continue
                if self._last_clause_len(result[:idx]) < self.min_clause_chars:
                    continue
                result = result[:idx] + ZH_COMMA + result[idx:]
                start += 1
        return result

    @staticmethod
    def _last_clause_len(text: str) -> int:
        """最后一个标点之后的字符数。"""
        for i in range(len(text) - 1, -1, -1):
            if text[i] in ALL_PUNCT:
                return len(text) - 1 - i
        return len(text)

    def _append_terminal(self, text: str) -> str:
        if not text:
            return text
        if text[-1] in TERMINALS:
            return text
        stripped = text.rstrip("".join(PAUSES))
        if not stripped:
            return text
        if self.is_question(stripped):
            return stripped + ZH_QUESTION
        if self._is_exclamation(stripped):
            return stripped + ZH_EXCLAIM
        return stripped + ZH_PERIOD

    def _collapse(self, text: str) -> str:
        """清理重复标点与标点前的空格。"""
        result = _TRAILING_SPACE_RE.sub(r"\1", text)
        out: list[str] = []
        for ch in result:
            if ch in ALL_PUNCT and out and out[-1] in ALL_PUNCT:
                # 后来的终结标点优先级更高
                if ch in TERMINALS and out[-1] in PAUSES:
                    out[-1] = ch
                continue
            out.append(ch)
        return "".join(out).strip()

    # ------------------------------------------------------------------
    # 句型判断
    # ------------------------------------------------------------------

    def is_question(self, text: str) -> bool:
        """是否为疑问句。"""
        content = text.strip()
        if not content:
            return False
        if content[-1] in "?？":
            return True
        if _CJK_RE.search(content):
            tail = content[-2:]
            if any(tail.endswith(t) for t in _ZH_QUESTION_TAILS):
                return True
            return any(word in content for word in _ZH_QUESTION_WORDS)
        tokens = re.findall(r"[A-Za-z']+", content.lower())
        if not tokens:
            return False
        return tokens[0] in _EN_QUESTION_HEADS

    def _is_exclamation(self, text: str) -> bool:
        return any(cue in text for cue in _ZH_EXCLAIM_CUES)

    # ------------------------------------------------------------------
    # 可选模型层
    # ------------------------------------------------------------------

    def _load_model(self) -> Any:
        """惰性加载 funasr ct-punc；失败后永久回落规则层。"""
        if self._model is not None or self._model_failed:
            return self._model
        try:
            from funasr import AutoModel  # type: ignore[import-not-found]

            self._model = AutoModel(model=self.model_dir or "ct-punc", disable_update=True)
            log.info("asr.punctuation_model_loaded", model=self.model_dir or "ct-punc")
        except Exception as exc:  # noqa: BLE001 - 缺依赖是预期路径
            self._model_failed = True
            self._model = None
            log.info("asr.punctuation_model_unavailable", error=str(exc))
        return self._model

    async def restore_async(self, text: str, *, pauses: list[PauseMark] | None = None,
                            is_final: bool = True) -> str:
        """异步恢复标点：可用模型时走模型（线程池），否则走规则。"""
        rule_result = self.restore(text, pauses=pauses, is_final=is_final)
        if not self.use_model or not is_final or not text.strip():
            return rule_result

        model = await asyncio.to_thread(self._load_model)
        if model is None:
            return rule_result
        try:
            raw = await asyncio.to_thread(model.generate, input=text)
            punctuated = _extract_funasr_text(raw)
            return punctuated or rule_result
        except Exception as exc:  # noqa: BLE001
            log.warning("asr.punctuation_model_failed", error=str(exc))
            return rule_result

    def reset(self) -> None:
        """重置一次性状态。

        规则层本身是无状态的（每次 :meth:`restore` 都从裸文本重算），
        这里只清掉"模型加载失败"的记忆，让下一轮有机会重新尝试加载
        —— 模型目录可能在会话之间被补齐。
        """
        self._model_failed = False

    def __repr__(self) -> str:
        return f"<PunctuationRestorer lang={self.language} model={self.use_model}>"


def _extract_funasr_text(raw: Any) -> str:
    """从 funasr 返回结构里取出文本。"""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, dict):
            return str(first.get("text", "")).strip()
        return str(first).strip()
    if isinstance(raw, dict):
        return str(raw.get("text", "")).strip()
    return ""


def strip_punctuation(text: str) -> str:
    """去掉全部标点（用于文本比对与去重）。"""
    return "".join(ch for ch in text if ch not in ALL_PUNCT).strip()


def has_terminal(text: str) -> bool:
    """是否以终结标点结尾。"""
    content = text.strip()
    return bool(content) and content[-1] in TERMINALS


__all__ = [
    "ALL_PUNCT",
    "DEFAULT_COMMA_PAUSE_MS",
    "DEFAULT_PERIOD_PAUSE_MS",
    "PAUSES",
    "TERMINALS",
    "ZH_COMMA",
    "ZH_EXCLAIM",
    "ZH_PERIOD",
    "ZH_QUESTION",
    "PauseMark",
    "PunctuationRestorer",
    "has_terminal",
    "strip_punctuation",
]

"""文本归一化：ASR 后处理、TTS 前处理、记忆去重共用的一套规则。

三个使用场景的诉求不同，因此拆成三个入口而不是一个「万能 normalize」：

* :func:`normalize_for_display` —— 给人看：保留标点与大小写，只修空白与全半角；
* :func:`normalize_for_tts`     —— 给 TTS 读：展开数字/符号，去掉 Markdown 与 emoji；
* :func:`normalize_for_match`   —— 给机器比：全部小写、去标点、去空白，用于去重/命中判断。
"""
from __future__ import annotations

import re
import unicodedata

__all__ = [
    "CJK_PATTERN",
    "collapse_whitespace",
    "count_tokens_approx",
    "detect_script",
    "fullwidth_to_halfwidth",
    "is_probably_complete_sentence",
    "normalize_for_display",
    "normalize_for_match",
    "normalize_for_tts",
    "split_sentences",
    "strip_emoji",
    "strip_markdown",
    "truncate",
]

CJK_PATTERN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
_LATIN_PATTERN = re.compile(r"[A-Za-z]")
_WHITESPACE = re.compile(r"\s+")
_MD_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_MD_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_EMPHASIS = re.compile(r"(\*{1,3}|_{1,3}|~~)(.+?)\1", re.DOTALL)
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_QUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_MD_LIST = re.compile(r"^\s{0,3}([-*+]|\d+[.)])\s+", re.MULTILINE)
_MD_RULE = re.compile(r"^\s{0,3}([-*_])\s*(\1\s*){2,}$", re.MULTILINE)
_EMOJI_RANGES = re.compile(
    "["
    "\U0001f300-\U0001f9ff"  # 符号与象形文字
    "\U0001fa00-\U0001faff"  # 扩展 A/B
    "\U00002600-\U000027bf"  # 杂项符号 + Dingbats
    "\U0001f000-\U0001f0ff"  # 麻将/扑克
    "\U0000fe00-\U0000fe0f"  # 变体选择符
    "\U0001f1e6-\U0001f1ff"  # 区域指示符（国旗）
    "\U00002190-\U000021ff"  # 箭头
    "\U00002b00-\U00002bff"
    "\U0000200d"             # 零宽连接符
    "]+"
)
#: 中英文句末标点。半角句点单独处理（要避开 "3.14" 与 "e.g."）。
_SENT_END = re.compile(r"(?<=[。！？!?；;])\s*|(?<=[.])\s+(?=[A-Z\u4e00-\u9fff])")
_PUNCT_CLASS = re.compile(r"[^\w\s]", re.UNICODE)
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "fig", "no", "inc", "ltd", "co", "approx",
})
#: TTS 前处理需要口语化的常见符号。
_SYMBOL_SPEECH: dict[str, str] = {
    "%": " percent ", "&": " and ", "@": " at ", "#": " number ",
    "+": " plus ", "=": " equals ", "/": " slash ", "~": " to ",
    "×": " times ", "÷": " divided by ", "°": " degrees ",
}


def fullwidth_to_halfwidth(text: str) -> str:
    """全角 ASCII 与全角空格 → 半角。CJK 标点（，。！？）保持不变。"""
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def collapse_whitespace(text: str) -> str:
    """把任意连续空白（含换行/制表/零宽空格）压成单个半角空格并去首尾。"""
    cleaned = text.replace("\u200b", "").replace("\ufeff", "")
    return _WHITESPACE.sub(" ", cleaned).strip()


def strip_emoji(text: str) -> str:
    """移除 emoji 与变体选择符。TTS 前必做，否则会读出「笑脸」之类的字面词。"""
    return _EMOJI_RANGES.sub("", text)


def strip_markdown(text: str) -> str:
    """把 Markdown 降级成纯文本，保留可读的内容主体。"""
    out = _MD_CODE_FENCE.sub(" ", text)
    out = _MD_IMAGE.sub(r"\1", out)
    out = _MD_LINK.sub(r"\1", out)
    out = _MD_INLINE_CODE.sub(r"\1", out)
    out = _MD_RULE.sub("", out)
    out = _MD_HEADING.sub("", out)
    out = _MD_QUOTE.sub("", out)
    out = _MD_LIST.sub("", out)
    out = _MD_EMPHASIS.sub(r"\2", out)
    return out


def detect_script(text: str) -> str:
    """粗判文本主体书写系统：``zh`` / ``en`` / ``mixed`` / ``unknown``。

    只看字符构成，不做语言模型判别——足够驱动「选中文音色还是英文音色」这类决策，
    真正的语种识别由 :mod:`omnistream.perception.asr.language_id` 负责。
    """
    cjk = len(CJK_PATTERN.findall(text))
    latin = len(_LATIN_PATTERN.findall(text))
    total = cjk + latin
    if total == 0:
        return "unknown"
    cjk_ratio = cjk / total
    if cjk_ratio >= 0.7:
        return "zh"
    if cjk_ratio <= 0.1:
        return "en"
    return "mixed"


def normalize_for_display(text: str) -> str:
    """展示用归一化：NFKC + 全角转半角 + 空白压缩。语义完全保留。"""
    if not text:
        return ""
    return collapse_whitespace(fullwidth_to_halfwidth(unicodedata.normalize("NFKC", text)))


def normalize_for_tts(text: str) -> str:
    """TTS 前处理：去 Markdown、去 emoji、符号口语化、压缩空白。"""
    if not text:
        return ""
    out = strip_markdown(text)
    out = strip_emoji(out)
    out = unicodedata.normalize("NFKC", out)
    for symbol, spoken in _SYMBOL_SPEECH.items():
        if symbol in out:
            out = out.replace(symbol, spoken)
    out = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", out)
    return collapse_whitespace(out)


def normalize_for_match(text: str) -> str:
    """匹配/去重用归一化：小写 + 去标点 + 去全部空白。不可逆，仅用于比较。"""
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text).lower()
    out = strip_emoji(out)
    out = _PUNCT_CLASS.sub("", out)
    return _WHITESPACE.sub("", out)


def split_sentences(text: str, *, min_length: int = 1) -> list[str]:
    """按中英文句末标点切句，过滤掉长度不足 ``min_length`` 的碎片。

    语义端点检测（semantic endpointing）用它判断「用户这句话说完了没有」。
    """
    if not text.strip():
        return []
    parts = [p.strip() for p in _SENT_END.split(text) if p and p.strip()]
    return [p for p in parts if len(p) >= min_length]


def is_probably_complete_sentence(text: str) -> bool:
    """启发式判断一句话是否说完：句末标点 + 非缩写 + 长度下限。"""
    stripped = text.strip()
    if len(stripped) < 2:
        return False
    if stripped[-1] in "。！？!?；;":
        return True
    if stripped.endswith("."):
        tail = stripped[:-1].rsplit(" ", 1)[-1].lower()
        return tail not in _ABBREVIATIONS and not tail.isdigit()
    return False


def count_tokens_approx(text: str) -> int:
    """粗估 token 数：CJK 每字 1 token，其余按 4 字符 1 token。

    只用于配额预估与记忆预算裁剪，不追求与任何具体 tokenizer 精确一致。
    """
    if not text:
        return 0
    cjk = len(CJK_PATTERN.findall(text))
    rest = len(text) - cjk
    return cjk + max(0, rest) // 4 + (1 if rest % 4 else 0)


def truncate(text: str, max_chars: int, *, suffix: str = "…") -> str:
    """按字符截断并追加省略号，尽量在词/句边界处断开。"""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    budget = max(1, max_chars - len(suffix))
    head = text[:budget]
    cut = max(head.rfind(" "), head.rfind("，"), head.rfind("。"), head.rfind(","))
    if cut > budget * 0.6:
        head = head[:cut]
    return head.rstrip() + suffix

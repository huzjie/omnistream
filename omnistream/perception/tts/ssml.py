"""SSML 构建与解析：让"说话方式"成为可编程的一等公民。

平台内部统一用**纯文本 + 韵律参数**表达合成意图，
到引擎边界再翻译成各家 SSML 方言（Azure / 阿里 / 火山语法差异很大）。

本模块提供三件事：

1. :class:`Prosody` —— 韵律参数（语速/音高/音量/停顿/情绪）的结构化表达；
2. :func:`build_ssml` —— 把文本 + 韵律渲染成标准 SSML；
3. :func:`strip_ssml` —— 反向剥离标签取纯文本（引擎不支持 SSML 时降级用），
   并把 ``<break>`` 转换成可被 mock/piper 消费的静音标记。

安全：所有用户文本都经过 :func:`escape_text` 转义，
避免 LLM 输出里的 ``<`` ``&`` 破坏 SSML 结构（这是真实存在的注入面）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final
from xml.sax.saxutils import escape as _xml_escape

__all__ = [
    "EMOTION_STYLES",
    "Prosody",
    "SSMLDialect",
    "build_ssml",
    "escape_text",
    "extract_breaks",
    "is_ssml",
    "strip_ssml",
]

#: 支持的情绪风格（对齐 Azure ``mstts:express-as`` 的常用集合）
EMOTION_STYLES: Final[frozenset[str]] = frozenset({
    "neutral", "cheerful", "sad", "angry", "excited", "friendly",
    "hopeful", "shouting", "whispering", "terrified", "unfriendly",
    "empathetic", "calm", "serious",
})

_TAG_RE: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")
_BREAK_RE: Final[re.Pattern[str]] = re.compile(
    r"<break\s+time\s*=\s*[\"'](\d+)(ms|s)[\"']\s*/?>", re.IGNORECASE)
_SPEAK_RE: Final[re.Pattern[str]] = re.compile(r"^\s*<speak[\s>]", re.IGNORECASE)


def escape_text(text: str) -> str:
    """XML 转义（``&`` ``<`` ``>`` ``"``）。"""
    return _xml_escape(text, {'"': "&quot;"})


def is_ssml(text: str) -> bool:
    """判断文本是否已经是 SSML。"""
    return bool(_SPEAK_RE.match(text or ""))


@dataclass(slots=True)
class Prosody:
    """韵律参数。

    所有取值都用**相对量**表达，便于跨引擎映射：
    ``rate=1.2`` 表示比默认快 20%，``pitch_st=2`` 表示升高 2 个半音。
    """

    rate: float = 1.0
    #: 音高偏移，单位半音（semitone），范围 [-12, 12]
    pitch_st: float = 0.0
    volume_db: float = 0.0
    #: 句首停顿（毫秒）
    lead_break_ms: int = 0
    #: 句尾停顿（毫秒）
    trail_break_ms: int = 0
    emotion: str = ""
    #: 情绪强度 0~2
    emotion_degree: float = 1.0

    def normalized(self) -> "Prosody":
        """夹到合法区间。"""
        return Prosody(
            rate=max(0.5, min(3.0, self.rate)),
            pitch_st=max(-12.0, min(12.0, self.pitch_st)),
            volume_db=max(-30.0, min(12.0, self.volume_db)),
            lead_break_ms=max(0, min(5000, self.lead_break_ms)),
            trail_break_ms=max(0, min(5000, self.trail_break_ms)),
            emotion=self.emotion if self.emotion in EMOTION_STYLES else "",
            emotion_degree=max(0.01, min(2.0, self.emotion_degree)),
        )

    @property
    def rate_attr(self) -> str:
        """SSML ``rate`` 属性值（百分比形式）。"""
        return f"{(self.rate - 1.0) * 100:+.0f}%"

    @property
    def pitch_attr(self) -> str:
        """SSML ``pitch`` 属性值（半音形式）。"""
        return f"{self.pitch_st:+.1f}st"

    @property
    def volume_attr(self) -> str:
        """SSML ``volume`` 属性值（dB 形式）。"""
        return f"{self.volume_db:+.1f}dB"

    @property
    def is_default(self) -> bool:
        """是否全为默认值（可省略 prosody 标签）。"""
        return (abs(self.rate - 1.0) < 1e-3 and abs(self.pitch_st) < 1e-3
                and abs(self.volume_db) < 1e-3 and not self.emotion
                and not self.lead_break_ms and not self.trail_break_ms)


@dataclass(slots=True)
class SSMLDialect:
    """一家引擎的 SSML 方言差异。"""

    name: str = "standard"
    #: 命名空间声明
    namespace: str = "http://www.w3.org/2001/10/synthesis"
    #: 是否支持 ``mstts:express-as`` 情绪标签
    supports_emotion: bool = False
    #: 情绪标签名
    emotion_tag: str = "mstts:express-as"
    #: 额外命名空间
    extra_ns: str = ""
    #: 是否支持 ``<voice>`` 标签
    supports_voice: bool = True


#: 标准 W3C 方言
STANDARD_DIALECT: Final[SSMLDialect] = SSMLDialect()

#: Azure 方言（支持情绪）
AZURE_DIALECT: Final[SSMLDialect] = SSMLDialect(
    name="azure",
    supports_emotion=True,
    extra_ns='xmlns:mstts="https://www.w3.org/2001/mstts"',
)


def build_ssml(
    text: str,
    *,
    voice: str = "",
    language: str = "zh-CN",
    prosody: Prosody | None = None,
    dialect: SSMLDialect = STANDARD_DIALECT,
    escape: bool = True,
) -> str:
    """把文本与韵律渲染成 SSML。

    Args:
        text: 纯文本（会被转义）。若已是 SSML 则原样返回。
        voice: 音色名，留空则不生成 ``<voice>`` 标签。
        language: BCP-47 语言标记。
        prosody: 韵律参数，``None`` 表示全默认。
        dialect: 目标方言。
        escape: 是否对 ``text`` 做 XML 转义（内部已生成的片段传 False）。

    Returns:
        完整 SSML 文档字符串。
    """
    if is_ssml(text):
        return text
    params = (prosody or Prosody()).normalized()
    body = escape_text(text) if escape else text

    if params.lead_break_ms:
        body = f'<break time="{params.lead_break_ms}ms"/>{body}'
    if params.trail_break_ms:
        body = f'{body}<break time="{params.trail_break_ms}ms"/>'

    if not params.is_default:
        body = (f'<prosody rate="{params.rate_attr}" pitch="{params.pitch_attr}" '
                f'volume="{params.volume_attr}">{body}</prosody>')

    if params.emotion and dialect.supports_emotion:
        body = (f'<{dialect.emotion_tag} style="{params.emotion}" '
                f'styledegree="{params.emotion_degree:.2f}">{body}'
                f'</{dialect.emotion_tag}>')

    if voice and dialect.supports_voice:
        body = f'<voice name="{escape_text(voice)}">{body}</voice>'

    ns = f'xmlns="{dialect.namespace}"'
    extra = f" {dialect.extra_ns}" if dialect.extra_ns else ""
    return (f'<speak version="1.0" {ns}{extra} xml:lang="{escape_text(language)}">'
            f"{body}</speak>")


def extract_breaks(ssml: str) -> list[tuple[int, int]]:
    """提取 ``<break>`` 标签在**纯文本坐标系**中的位置与时长。

    Args:
        ssml: SSML 文本。

    Returns:
        ``(纯文本字符偏移, 停顿毫秒)`` 列表，按偏移升序。
    """
    breaks: list[tuple[int, int]] = []
    plain_len = 0
    pos = 0
    for match in _TAG_RE.finditer(ssml):
        plain_len += len(ssml[pos:match.start()])
        pos = match.end()
        tag = match.group(0)
        bm = _BREAK_RE.match(tag)
        if bm:
            value = int(bm.group(1))
            unit = bm.group(2).lower()
            breaks.append((plain_len, value * 1000 if unit == "s" else value))
    return breaks


def strip_ssml(ssml: str, *, keep_breaks: bool = False) -> str:
    """剥离全部 SSML 标签，返回纯文本。

    Args:
        ssml: SSML 或纯文本。
        keep_breaks: 为 True 时把 ``<break time="500ms"/>`` 转成等价的
            省略号（让不支持 SSML 的引擎也能读出停顿感）。

    Returns:
        纯文本。
    """
    if not ssml:
        return ""
    text = ssml
    if keep_breaks:
        def _to_ellipsis(m: re.Match[str]) -> str:
            value = int(m.group(1))
            ms = value * 1000 if m.group(2).lower() == "s" else value
            return "…" * max(1, min(3, ms // 300))

        text = _BREAK_RE.sub(_to_ellipsis, text)
    text = _TAG_RE.sub("", text)
    return (text.replace("&lt;", "<").replace("&gt;", ">")
            .replace("&quot;", '"').replace("&apos;", "'")
            .replace("&amp;", "&").strip())

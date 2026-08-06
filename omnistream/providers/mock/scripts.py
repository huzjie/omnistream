"""Mock Provider 的对话脚本引擎（零依赖、完全确定）。

职责
----
* 根据用户文本、视觉描述与记忆提示，规划一轮回复（句子列表 + 可选工具调用）。
* 提供确定性的中英文分词，用于逐 token 推送 ``text_delta``。
* 描述视频帧（不做真实 CV，按帧内容哈希产出稳定描述）。

设计原则：相同输入必然产出相同输出，便于演示、录屏与回归比对。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omnistream.schemas.tool import ToolResult
    from omnistream.schemas.video import VideoFrame

__all__ = ["PlannedToolCall", "ResponsePlan", "ScriptEngine"]

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;\.])\s*")
_TOKEN_SPLIT = re.compile(r"[A-Za-z0-9]+(?:[''][A-Za-z]+)?|[\u4e00-\u9fff]|[^\sA-Za-z0-9\u4e00-\u9fff]")

#: 意图关键词 → 回复模板。命中第一条即用。
_INTENT_RULES: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("你好", "您好", "hi", "hello", "在吗"),
     ("你好，我在。", "有什么可以帮你的吗？")),
    (("天气", "气温", "下雨", "weather"),
     ("我帮你查一下天气。",)),
    (("时间", "几点", "日期", "time"),
     ("我看一下当前时间。",)),
    (("打断", "别说了", "停", "stop"),
     ("好的，我停下来听你说。",)),
    (("介绍", "是什么", "干什么", "能做什么"),
     ("我是 OmniStream 的实时语音助手。",
      "我可以边听边看，随时被你打断，并在需要时调用工具。")),
    (("谢谢", "多谢", "thanks", "thank you"),
     ("不客气，随时找我。",)),
    (("再见", "拜拜", "bye"),
     ("好的，我们下次再聊。",)),
]

#: 无关键词命中时的通用回复池，按输入哈希选择，保证确定但不单调。
_GENERIC_REPLIES: tuple[tuple[str, ...], ...] = (
    ("我明白你的意思了。", "让我按你说的思路往下讲。"),
    ("这个问题挺有意思。", "我先说一下我的理解。"),
    ("收到，我这就处理。", "稍后我把结论告诉你。"),
    ("好的。", "我先给你一个简短的回答，需要展开随时打断我。"),
    ("嗯，我在听。", "你可以继续说，我会边听边整理。"),
)

#: 工具触发规则：关键词 → (工具名, 参数构造键)
_TOOL_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("天气", "气温", "下雨", "weather"), "get_weather"),
    (("时间", "几点", "日期", "time"), "get_time"),
    (("搜索", "查一下", "search", "查查"), "web_search"),
    (("计算", "算一下", "多少钱", "calculate"), "calculator"),
)

_SCENE_SUBJECTS = ("一个人", "两个人", "一张桌子", "一台笔记本电脑",
                   "一杯水", "一本书", "一面白墙", "一个显示器")
_SCENE_ACTIONS = ("正对着镜头", "在画面左侧", "在画面中央", "占据画面下方",
                  "光线偏暗", "光线充足")


def _hash_index(text: str, modulo: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % max(1, modulo)


@dataclass(slots=True)
class PlannedToolCall:
    """脚本规划出的一次工具调用。"""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ResponsePlan:
    """一轮回复的完整规划。"""

    sentences: list[str] = field(default_factory=list)
    tool_calls: list[PlannedToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def full_text(self) -> str:
        return "".join(self.sentences)


class ScriptEngine:
    """确定性对话脚本引擎。"""

    def __init__(self, *, seed: int = 42) -> None:
        self.seed = seed
        self.language: str = "zh-CN"
        self.available_tools: list[str] = []
        self._turn_index: int = 0
        self._tool_results: list[tuple[str, str]] = []
        self._last_user_text: str = ""

    # ---------------- 配置 ----------------

    def configure(self, *, language: str = "zh-CN", tools: list[str] | None = None) -> None:
        """由 provider 在 ``connect()`` 时调用。"""
        self.language = language or "zh-CN"
        self.available_tools = list(tools or [])

    def reset(self) -> None:
        self._turn_index = 0
        self._tool_results.clear()
        self._last_user_text = ""

    # ---------------- 分词 ----------------

    def tokenize(self, text: str) -> list[str]:
        """确定性分词：英文按词、中文按字、标点独立成 token。"""
        if not text:
            return []
        tokens = _TOKEN_SPLIT.findall(text)
        return tokens or [text]

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """按中英文句末标点切句，保留标点。"""
        parts = [p for p in _SENTENCE_SPLIT.split(text) if p and p.strip()]
        return parts or ([text] if text else [])

    # ---------------- 视觉 ----------------

    def describe_frame(self, frame: "VideoFrame") -> str:
        """按帧内容产出稳定的一句话描述（不做真实 CV）。"""
        fingerprint = hashlib.blake2b(
            frame.data[:512] if frame.data else b"empty", digest_size=8).hexdigest()
        subject = _SCENE_SUBJECTS[_hash_index(fingerprint, len(_SCENE_SUBJECTS))]
        action = _SCENE_ACTIONS[_hash_index(fingerprint + "a", len(_SCENE_ACTIONS))]
        return f"画面里有{subject}，{action}（{frame.width}x{frame.height}）"

    # ---------------- 工具 ----------------

    def absorb_tool_result(self, result: "ToolResult") -> None:
        """吸收工具执行结果，下一轮生成时据此作答。"""
        content = result.content
        if content is None:
            rendered = result.error or "没有拿到结果"
        elif isinstance(content, str):
            rendered = content
        elif isinstance(content, dict):
            rendered = "，".join(f"{k}是{v}" for k, v in list(content.items())[:4])
        else:
            rendered = str(content)
        self._tool_results.append((result.name or "工具", rendered))

    def _plan_tools(self, user_text: str) -> list[PlannedToolCall]:
        """仅在工具已声明且关键词命中、且本轮尚无结果时才发起调用。"""
        if not self.available_tools or self._tool_results:
            return []
        low = user_text.lower()
        for keywords, tool_name in _TOOL_RULES:
            if tool_name not in self.available_tools:
                continue
            if any(k in low for k in keywords):
                return [PlannedToolCall(name=tool_name,
                                        arguments=self._tool_arguments(tool_name, user_text))]
        return []

    @staticmethod
    def _tool_arguments(tool_name: str, user_text: str) -> dict[str, Any]:
        if tool_name == "get_weather":
            city = "北京"
            for candidate in ("北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安"):
                if candidate in user_text:
                    city = candidate
                    break
            return {"city": city, "unit": "celsius"}
        if tool_name == "get_time":
            return {"timezone": "Asia/Shanghai"}
        if tool_name == "web_search":
            return {"query": user_text[:64] or "OmniStream"}
        if tool_name == "calculator":
            expr = "".join(ch for ch in user_text if ch in "0123456789+-*/.() ")
            return {"expression": expr.strip() or "1+1"}
        return {"input": user_text[:64]}

    # ---------------- 规划 ----------------

    def plan(self, user_text: str, *, vision: str | None = None,
             memory_hint: str | None = None) -> ResponsePlan:
        """规划一轮回复。

        Args:
            user_text: 本轮用户输入（可能为空，例如纯语音场景由 ASR 另行提供）。
            vision: 最近一帧的视觉描述，命中"看"类问题时会被引用。
            memory_hint: 记忆 Agent 注入的一句提醒（``instructions_suffix``）。
        """
        self._turn_index += 1
        text = (user_text or "").strip()
        self._last_user_text = text

        tool_calls = self._plan_tools(text)
        if tool_calls:
            return ResponsePlan(sentences=[], tool_calls=tool_calls,
                                input_tokens=self._count_input(text, vision, memory_hint),
                                output_tokens=0)

        sentences: list[str] = []

        # 1) 工具结果优先播报
        if self._tool_results:
            for name, rendered in self._tool_results:
                sentences.append(f"{name}的结果是：{rendered}。")
            self._tool_results.clear()

        # 2) 记忆提醒（一句话，不喧宾夺主）
        if memory_hint:
            hint = memory_hint.strip()
            if hint:
                sentences.append(f"我记得{hint}。" if not hint.endswith("。") else f"我记得{hint}")

        # 3) 视觉问题
        low = text.lower()
        if vision and any(k in low for k in ("看到", "画面", "这是什么", "屏幕", "see", "看看")):
            sentences.append(f"我看到{vision}。")

        # 4) 意图匹配
        if not sentences or text:
            matched = False
            for keywords, replies in _INTENT_RULES:
                if any(k in low for k in keywords):
                    sentences.extend(replies)
                    matched = True
                    break
            if not matched and not sentences:
                pool = _GENERIC_REPLIES[_hash_index(text + str(self.seed), len(_GENERIC_REPLIES))]
                sentences.extend(pool)

        if not sentences:
            sentences = ["好的。"]

        return ResponsePlan(
            sentences=sentences,
            tool_calls=[],
            input_tokens=self._count_input(text, vision, memory_hint),
            output_tokens=sum(len(self.tokenize(s)) for s in sentences),
        )

    def _count_input(self, text: str, vision: str | None,
                     memory_hint: str | None) -> int:
        total = len(self.tokenize(text))
        if vision:
            total += len(self.tokenize(vision))
        if memory_hint:
            total += len(self.tokenize(memory_hint))
        # 加上系统提示的固定开销，让 usage 数字看起来真实
        return total + 24

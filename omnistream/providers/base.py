"""Provider 抽象基类：所有实时模型接入点的统一契约。

设计要点
--------
1. **单向事件流**：调用方通过 ``send_*`` 推入，通过 ``events()`` 异步迭代取出，
   不使用回调，避免回调地狱与异常吞噬。
2. **可取消**：``cancel_response()`` 必须在 100ms 内让 ``events()`` 停止吐出音频，
   这是 barge-in 体验的硬指标。
3. **能力自述**：``capabilities`` 决定编排层是否补齐 ASR/TTS 或降级。
4. **无状态副作用**：Provider 不写数据库、不发指标，只翻译协议；观测由编排层负责。
"""
from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from omnistream.core.clock import now_ms
from omnistream.protocol.enums import StopReason, TurnRole
from omnistream.schemas.audio import AudioChunk
from omnistream.schemas.provider import ProviderCapabilities, ProviderHealth
from omnistream.schemas.session import SessionConfig
from omnistream.schemas.tool import ToolCall, ToolResult
from omnistream.schemas.video import VideoFrame

# --------------------------------------------------------------------------
# Provider 内部事件（区别于对外 WS 协议事件；由 session_bridge 翻译）
# --------------------------------------------------------------------------

ProviderEventKind = Literal[
    "connected",
    "session_updated",
    "input_speech_started",     # 仅 server_vad=True 的 provider 会吐
    "input_speech_stopped",
    "input_transcript_delta",   # 用户语音的转写增量
    "input_transcript_done",
    "text_delta",               # 助手文本增量
    "text_done",
    "audio_delta",              # 助手音频增量
    "audio_done",
    "tool_call",
    "response_done",
    "response_cancelled",
    "usage",
    "error",
    "closed",
]


@dataclass(slots=True)
class ProviderEvent:
    kind: ProviderEventKind
    ts_ms: int = field(default_factory=now_ms)
    # 按 kind 取用其中一个字段
    text: str | None = None
    audio: AudioChunk | None = None
    tool_call: ToolCall | None = None
    stop_reason: StopReason | None = None
    role: TurnRole = TurnRole.ASSISTANT
    is_final: bool = False
    usage: dict[str, int] | None = None
    error: Exception | None = None
    raw: dict[str, Any] | None = field(default=None, repr=False)


# --------------------------------------------------------------------------
# 抽象基类
# --------------------------------------------------------------------------


class BaseRealtimeProvider(abc.ABC):
    """所有 Provider 的统一接口。

    生命周期::

        p = SomeProvider(credentials=..., settings=...)
        await p.connect(session_config)
        async for ev in p.events():   # 独立消费协程
            ...
        await p.send_audio(chunk)     # 生产侧并发调用
        await p.cancel_response(StopReason.INTERRUPTED)
        await p.close()
    """

    #: 注册名，唯一。用于 `provider: "doubao_seedrealtime"` 这类配置。
    name: str = "base"

    #: 该实现的能力声明。子类必须覆盖为类属性或在 __init__ 中赋值。
    capabilities: ProviderCapabilities = ProviderCapabilities()

    # ---------------- 连接生命周期 ----------------

    @abc.abstractmethod
    async def connect(self, config: SessionConfig) -> None:
        """建立与上游的连接并完成会话初始化。

        失败必须抛 ``ProviderError`` 子类（``ProviderUnauthorized`` / ``ProviderTimeout``）。
        成功后 ``events()`` 的第一个事件必须是 ``kind="connected"``。
        """

    @abc.abstractmethod
    async def close(self) -> None:
        """幂等关闭。必须让 ``events()`` 在吐出 ``kind="closed"`` 后正常结束迭代。"""

    # ---------------- 输入 ----------------

    @abc.abstractmethod
    async def send_audio(self, chunk: AudioChunk) -> None:
        """推入一段用户音频（通常 20ms）。

        实现须自行完成采样率/编码适配（可用 ``providers.transcoding``）。
        当 ``capabilities.audio_in`` 为 False 时应抛 ``CapabilityUnsupported``。
        """

    async def send_video(self, frame: VideoFrame) -> None:
        """推入一帧视频。默认实现：不支持则抛异常。"""
        from omnistream.core.errors import CapabilityUnsupported

        raise CapabilityUnsupported(f"{self.name} 不支持视频输入", provider=self.name)

    @abc.abstractmethod
    async def send_text(self, text: str, *, role: TurnRole = TurnRole.USER) -> None:
        """推入一段文本（用户输入、系统提示追加、或记忆注入）。"""

    async def commit_input(self) -> None:
        """标记本轮用户输入结束（手动 VAD 模式）。server_vad 的 provider 可空实现。"""
        return None

    async def clear_input(self) -> None:
        """清空尚未提交的输入缓冲（用户误触/取消）。"""
        return None

    # ---------------- 生成控制 ----------------

    @abc.abstractmethod
    async def request_response(self, *, instructions_suffix: str | None = None) -> None:
        """请求模型开始生成一轮回复。

        ``instructions_suffix`` 用于**本轮临时**追加系统提示 —— 记忆 Agent 的
        "一条定向提醒"就是通过这个参数注入的，不污染长期 instructions。
        """

    @abc.abstractmethod
    async def cancel_response(self, reason: StopReason = StopReason.INTERRUPTED) -> None:
        """取消当前生成。**硬性要求：100ms 内停止 audio_delta 事件。**"""

    @abc.abstractmethod
    async def submit_tool_result(self, result: ToolResult) -> None:
        """回填工具执行结果，模型据此继续生成。"""

    async def update_session(self, patch: dict[str, Any]) -> None:
        """会话中途更新配置（换音色、改 instructions、增删工具）。默认不支持则忽略。"""
        return None

    # ---------------- 输出 ----------------

    @abc.abstractmethod
    def events(self) -> AsyncIterator[ProviderEvent]:
        """返回事件异步迭代器。**同一实例只允许被一个消费者迭代。**

        实现建议：内部持有 ``BoundedStream[ProviderEvent]``，网络读协程写入，
        此方法返回 ``stream.__aiter__()``。
        """

    # ---------------- 观测 ----------------

    async def health(self) -> ProviderHealth:
        """健康探测。默认返回"已连接即健康"。"""
        return ProviderHealth(name=self.name, healthy=True, checked_at_ms=now_ms())

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name}>"

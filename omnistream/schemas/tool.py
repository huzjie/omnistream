"""工具调用（function calling）领域模型。"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field, field_validator

from omnistream.core.clock import utcnow
from omnistream.core.constants import DEFAULT_TOOL_TIMEOUT_MS
from omnistream.protocol.enums import ToolCallState
from omnistream.schemas.base import OmniModel

#: 工具名允许的字符集：字母、数字、下划线、连字符（与 OpenAI / 豆包一致）
_NAME_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


class ToolSpec(OmniModel):
    """工具声明。``parameters`` 为标准 JSON Schema（object 类型）。"""

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []})
    #: 是否需要人工确认后才真正执行（高风险工具，如转账、删除）
    requires_confirmation: bool = False
    timeout_ms: int = Field(default=DEFAULT_TOOL_TIMEOUT_MS, ge=100)
    #: 语音场景下等待工具返回时播报的过渡语，避免长时间静默
    filler_text: str | None = Field(
        default=None, description='例如 "稍等，我查一下"')
    #: 允许与其他工具并行执行
    parallel_safe: bool = True
    tags: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not v:
            raise ValueError("工具名不能为空")
        if len(v) > 64:
            raise ValueError("工具名不能超过 64 字符")
        if set(v) - _NAME_OK:
            raise ValueError(f"工具名含非法字符: {v!r}")
        return v

    @field_validator("parameters")
    @classmethod
    def _params(cls, v: dict[str, Any]) -> dict[str, Any]:
        if v.get("type") != "object":
            raise ValueError('工具 parameters 顶层 type 必须为 "object"')
        v.setdefault("properties", {})
        return v

    def to_openai(self) -> dict[str, Any]:
        """转成 OpenAI / 兼容厂商的 tools 声明格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @property
    def required_args(self) -> list[str]:
        return list(self.parameters.get("required", []))


class ToolCall(OmniModel):
    """模型发起的一次工具调用。"""

    id: str
    turn_id: str = ""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    state: ToolCallState = ToolCallState.PENDING
    created_at: datetime = Field(default_factory=utcnow)
    #: 同一轮内的并行序号
    index: int = 0
    raw_arguments: str | None = Field(default=None, repr=False,
                                      description="模型原始输出，解析失败时保留用于排障")


class ToolResult(OmniModel):
    """工具执行结果，回填给模型继续生成。"""

    call_id: str
    name: str = ""
    ok: bool = True
    content: Any = None
    error: str | None = None
    state: ToolCallState = ToolCallState.SUCCEEDED
    elapsed_ms: float = 0.0
    finished_at: datetime = Field(default_factory=utcnow)
    #: 是否把结果内容也读给用户听（例如天气查询结果）
    speak: bool = True

    @classmethod
    def success(cls, call_id: str, content: Any, *, name: str = "",
                elapsed_ms: float = 0.0) -> "ToolResult":
        return cls(call_id=call_id, name=name, ok=True, content=content,
                   state=ToolCallState.SUCCEEDED, elapsed_ms=elapsed_ms)

    @classmethod
    def failure(cls, call_id: str, error: str, *, name: str = "",
                elapsed_ms: float = 0.0,
                state: ToolCallState = ToolCallState.FAILED) -> "ToolResult":
        return cls(call_id=call_id, name=name, ok=False, error=error,
                   content=None, state=state, elapsed_ms=elapsed_ms)

    def to_model_payload(self) -> str:
        """序列化成回填给模型的字符串。"""
        import orjson

        if self.ok:
            body = self.content if self.content is not None else {"ok": True}
        else:
            body = {"error": self.error or "工具执行失败"}
        if isinstance(body, str):
            return body
        return orjson.dumps(body).decode("utf-8")


class ToolPolicy(OmniModel):
    """租户/会话级工具执行策略。"""

    allow: list[str] = Field(default_factory=list, description="为空表示不限制")
    deny: list[str] = Field(default_factory=list)
    max_parallel: int = Field(default=3, ge=1)
    max_calls_per_turn: int = Field(default=5, ge=1)
    max_calls_per_session: int = Field(default=50, ge=1)
    default_timeout_ms: int = Field(default=DEFAULT_TOOL_TIMEOUT_MS, ge=100)
    #: 工具失败时是否让模型自行重试
    allow_model_retry: bool = True
    max_retries: int = Field(default=1, ge=0)

    def permits(self, name: str) -> bool:
        if name in self.deny:
            return False
        return not self.allow or name in self.allow

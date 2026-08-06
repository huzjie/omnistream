"""Provider 注册表配置：默认 provider、各家 endpoint/key/模型名/超时与熔断。

每家一个子模型，环境变量形如::

    OMS_PROVIDERS__DEFAULT=doubao
    OMS_PROVIDERS__DOUBAO__API_KEY=xxx
    OMS_PROVIDERS__OPENAI__BASE_URL=https://api.openai.com/v1
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from omnistream.core.constants import DEFAULT_CONNECT_TIMEOUT_S, DEFAULT_PROVIDER_TIMEOUT_S


class ProviderEntry(BaseModel):
    """单个 Provider 的通用连接配置。"""

    enabled: bool = True
    api_key: str | None = Field(default=None, repr=False)
    api_secret: str | None = Field(default=None, repr=False)
    base_url: str | None = None
    ws_url: str | None = None
    region: str | None = None
    model: str | None = None
    voice: str | None = None
    #: 路由优先级，越小越优先
    priority: int = 100
    connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S
    request_timeout_s: float = DEFAULT_PROVIDER_TIMEOUT_S
    max_retries: int = 2
    extra: dict[str, str] = Field(default_factory=dict, repr=False)

    @property
    def configured(self) -> bool:
        """是否已具备最小可用凭据。"""
        return bool(self.api_key) or bool(self.base_url)


class ProvidersSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OMS_PROVIDERS_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    #: 默认 provider 名。mock 无需任何凭据。
    default: str = "mock"
    #: 主 provider 不可用时的降级顺序
    fallbacks: list[str] = Field(default_factory=lambda: ["mock"])
    #: 允许自动降级
    allow_fallback: bool = True
    #: 能力不足时是否自动走 ASR+LLM+TTS 组合链路
    allow_composed: bool = True

    # ------------------------------------------------------------ 熔断
    circuit_failure_threshold: int = Field(default=5, ge=1)
    circuit_recovery_timeout_s: float = Field(default=30.0, gt=0.0)
    circuit_half_open_max_calls: int = Field(default=2, ge=1)

    # ------------------------------------------------------------ 各家配置
    mock: ProviderEntry = Field(default_factory=lambda: ProviderEntry(
        enabled=True, model="mock-realtime-1", priority=999))
    doubao: ProviderEntry = Field(default_factory=lambda: ProviderEntry(
        enabled=False, base_url="https://openspeech.bytedance.com",
        ws_url="wss://openspeech.bytedance.com/api/v3/realtime/dialogue",
        model="doubao-seed-realtime", priority=10))
    openai: ProviderEntry = Field(default_factory=lambda: ProviderEntry(
        enabled=False, base_url="https://api.openai.com/v1",
        ws_url="wss://api.openai.com/v1/realtime",
        model="gpt-4o-realtime-preview", voice="alloy", priority=20))
    gemini: ProviderEntry = Field(default_factory=lambda: ProviderEntry(
        enabled=False, base_url="https://generativelanguage.googleapis.com",
        model="gemini-2.0-flash-live", priority=30))
    qwen: ProviderEntry = Field(default_factory=lambda: ProviderEntry(
        enabled=False, base_url="https://dashscope.aliyuncs.com/api/v1",
        model="qwen-omni-turbo-realtime", priority=40))
    stepfun: ProviderEntry = Field(default_factory=lambda: ProviderEntry(
        enabled=False, base_url="https://api.stepfun.com/v1",
        model="step-1o-audio", priority=50))
    minimax: ProviderEntry = Field(default_factory=lambda: ProviderEntry(
        enabled=False, base_url="https://api.minimax.chat/v1",
        model="abab6.5s-chat", priority=60))
    #: 组合式 provider：任意 LLM + 本地/云 ASR/TTS
    composed: ProviderEntry = Field(default_factory=lambda: ProviderEntry(
        enabled=True, base_url=None, model=None, priority=500))

    def entry(self, name: str) -> ProviderEntry | None:
        """按名取配置。未知名返回 None。"""
        value = getattr(self, name, None)
        return value if isinstance(value, ProviderEntry) else None

    def all_entries(self) -> dict[str, ProviderEntry]:
        """返回全部 provider 配置映射。"""
        out: dict[str, ProviderEntry] = {}
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, ProviderEntry):
                out[name] = value
        return out

    def enabled_names(self) -> list[str]:
        """按优先级排序的已启用 provider 名列表。"""
        items = [(e.priority, n) for n, e in self.all_entries().items() if e.enabled]
        items.sort()
        return [n for _, n in items]

    def resolve_chain(self, preferred: str | None = None) -> list[str]:
        """生成路由链：首选 → 配置的 fallbacks → mock 兜底（去重保序）。"""
        chain: list[str] = []
        for name in [preferred or self.default, *self.fallbacks, "mock"]:
            if name and name not in chain:
                entry = self.entry(name)
                if entry is not None and entry.enabled:
                    chain.append(name)
        return chain or ["mock"]

"""Provider 领域模型：能力声明、健康状态、生成参数、注册信息。"""
from __future__ import annotations

from pydantic import Field

from omnistream.protocol.enums import AudioEncoding
from omnistream.schemas.base import OmniModel


class ProviderCapabilities(OmniModel):
    """能力声明。编排层据此决定是否需要组合式补齐或降级。"""
    native_duplex: bool = False        # 端到端原生全双工（音入音出，模型内部处理打断）
    audio_in: bool = False
    audio_out: bool = False
    video_in: bool = False
    text_in: bool = True
    text_out: bool = True
    server_vad: bool = False           # provider 侧自带 VAD/端点检测
    function_calling: bool = False
    parallel_tools: bool = False
    interruption: bool = False         # 支持中途取消生成
    proactive: bool = False            # 支持模型主动发起
    streaming_transcript: bool = False
    max_context_tokens: int = 32_768
    input_audio_formats: list[AudioEncoding] = Field(default_factory=lambda: [AudioEncoding.PCM16])
    output_audio_formats: list[AudioEncoding] = Field(default_factory=lambda: [AudioEncoding.PCM16])
    input_sample_rates: list[int] = Field(default_factory=lambda: [16000])
    output_sample_rates: list[int] = Field(default_factory=lambda: [24000])
    max_video_fps: float = 0.0

    def missing_for(self, required: "ProviderCapabilities") -> list[str]:
        gaps: list[str] = []
        for flag in ("audio_in", "audio_out", "video_in", "function_calling",
                     "interruption", "parallel_tools", "proactive"):
            if getattr(required, flag) and not getattr(self, flag):
                gaps.append(flag)
        return gaps


class ProviderHealth(OmniModel):
    name: str
    healthy: bool
    circuit: str = Field(default="closed", description="closed|open|half_open")
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    error_rate: float = 0.0
    last_error: str | None = None
    checked_at_ms: int = 0


class GenerationParams(OmniModel):
    temperature: float = 0.8
    top_p: float = 0.95
    max_output_tokens: int | None = None
    stop: list[str] = Field(default_factory=list)
    seed: int | None = None


# ---------------------------------------------------------------------------
# 注册与展示信息（非契约）
# ---------------------------------------------------------------------------


class ProviderInfo(OmniModel):
    """``GET /v1/providers`` 返回的一项，供控制台渲染 Provider 矩阵。"""

    name: str
    display_name: str = ""
    vendor: str = ""
    kind: str = Field(default="native_duplex",
                      description="native_duplex|composed|local")
    models: list[str] = Field(default_factory=list)
    default_model: str | None = None
    capabilities: ProviderCapabilities = Field(default_factory=ProviderCapabilities)
    enabled: bool = True
    configured: bool = Field(default=False, description="凭据是否已配置齐全")
    priority: int = Field(default=100, description="路由优先级，越小越优先")
    docs_url: str | None = None
    notes: str = ""

    @property
    def ready(self) -> bool:
        return self.enabled and self.configured


class ProviderCredential(OmniModel):
    """租户级 Provider 凭据（明文形态，仅在内存/写入前存在）。"""

    provider: str
    api_key: str | None = Field(default=None, repr=False)
    api_secret: str | None = Field(default=None, repr=False)
    base_url: str | None = None
    region: str | None = None
    model: str | None = None
    extra: dict[str, str] = Field(default_factory=dict, repr=False)

    def masked(self) -> "ProviderCredential":
        """脱敏副本，可安全写日志或回显给控制台。"""
        def _mask(v: str | None) -> str | None:
            if not v:
                return v
            return v[:4] + "*" * max(0, len(v) - 8) + v[-4:] if len(v) > 8 else "****"

        return self.model_copy(update={
            "api_key": _mask(self.api_key),
            "api_secret": _mask(self.api_secret),
            "extra": {k: "****" for k in self.extra},
        })


class ProviderRoute(OmniModel):
    """一次会话的 Provider 路由决策结果。"""

    primary: str = "mock"
    fallbacks: list[str] = Field(default_factory=list)
    reason: str = ""
    gaps: list[str] = Field(default_factory=list, description="需组合式补齐的能力")
    composed: bool = Field(default=False, description="是否走 ASR+LLM+TTS 组合链路")

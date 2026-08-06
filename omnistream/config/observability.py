"""可观测性配置：日志、Prometheus 指标、OpenTelemetry 追踪与 SLO 阈值。"""
from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from omnistream.core.constants import BARGE_IN_BUDGET_MS


class ObservabilitySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OMS_OBSERVABILITY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------ 日志
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "console"
    log_utc: bool = True
    #: 访问日志开关与慢请求阈值
    access_log: bool = True
    slow_request_ms: int = Field(default=1000, ge=1)
    #: 高频事件（audio.delta 等）的采样率，避免日志刷屏
    event_log_sample_rate: float = Field(default=0.01, ge=0.0, le=1.0)

    # ------------------------------------------------------------ 指标
    metrics_enabled: bool = True
    metrics_path: str = "/metrics"
    #: 是否把 provider/model 作为标签（基数风险，租户多时建议关闭 tenant 标签）
    metrics_label_tenant: bool = False
    metrics_label_provider: bool = True
    #: 延迟直方图分桶（毫秒）
    latency_buckets_ms: list[float] = Field(default_factory=lambda: [
        50, 100, 150, 200, 300, 400, 600, 800, 1200, 2000, 4000, 8000])

    # ------------------------------------------------------------ 追踪
    tracing_enabled: bool = False
    otlp_endpoint: str = "http://localhost:4317"
    otlp_protocol: Literal["grpc", "http"] = "grpc"
    otlp_headers: dict[str, str] = Field(default_factory=dict, repr=False)
    service_name: str = "omnistream"
    trace_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    #: 是否把 span 上下文注入日志（trace_id/span_id）
    trace_inject_logs: bool = True

    # ------------------------------------------------------------ SLO 阈值
    slo_ttfa_ms: int = Field(default=800, ge=50, description="用户说完 → 助手出声")
    slo_barge_in_ms: int = Field(default=BARGE_IN_BUDGET_MS, ge=50)
    slo_asr_partial_ms: int = Field(default=300, ge=20)
    slo_availability: float = Field(default=0.999, gt=0.0, le=1.0)
    #: 连续违约多少次触发告警事件
    slo_violation_window: int = Field(default=5, ge=1)

    # ------------------------------------------------------------ 健康检查
    health_check_interval_s: float = Field(default=15.0, gt=0.0)
    provider_health_timeout_s: float = Field(default=3.0, gt=0.0)
    #: Sentry DSN，留空关闭
    sentry_dsn: str | None = Field(default=None, repr=False)
    sentry_traces_sample_rate: float = Field(default=0.05, ge=0.0, le=1.0)

    @property
    def structured(self) -> bool:
        return self.log_format == "json"

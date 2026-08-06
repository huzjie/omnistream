"""OpenTelemetry 追踪。

提供极简封装：span 上下文管理器 + FastAPI middleware 集成点。
依赖缺失时 no-op 降级。
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    _HAS_OTEL = True
except ImportError:  # pragma: no cover
    _HAS_OTEL = False
    trace = None  # type: ignore[assignment]

_tracer = None


def setup_tracing(service_name: str = "omnistream",
                  endpoint: str | None = None) -> bool:
    """初始化 tracer provider；endpoint 为空时输出到控制台。"""
    global _tracer
    if not _HAS_OTEL:
        logger.warning("opentelemetry 未安装，追踪降级 no-op")
        return False
    provider = TracerProvider()
    exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint else ConsoleSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)
    return True


@contextmanager
def start_span(name: str, **attrs: Any) -> Iterator[Any]:
    """span 上下文管理器（no-op 安全）。"""
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as span:
        for k, v in attrs.items():
            span.set_attribute(k, str(v))
        yield span


def get_tracer():
    return _tracer

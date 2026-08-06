"""Prometheus 指标定义。

关键实时延迟指标：
- oms_ttft_ms              —— 首字延迟（Time To First Token）
- oms_turn_rtt_ms          —— 端到端轮次 RTT
- oms_bargein_latency_ms   —— 打断响应时延
- oms_sessions_active      —— 活跃会话
- oms_events_total         —— 事件吞吐
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        start_http_server,
    )
except ImportError:  # pragma: no cover - 可选依赖
    logger.warning("prometheus-client 未安装，指标降级为 no-op")
    start_http_server = None  # type: ignore[assignment]

    class _NoOp:
        def labels(self, *a, **k):
            return self

        def inc(self, *a, **k):
            return None

        def set(self, *a, **k):
            return None

        def observe(self, *a, **k):
            return None

        def time(self, *a, **k):
            return _NoOp()

    Counter = Gauge = Histogram = _NoOp  # type: ignore[assignment,misc]


# ---------------- 延迟 ----------------


def _hist(name: str, doc: str, buckets) -> Histogram:
    try:
        return Histogram(name, doc, ["provider", "modality"], buckets=buckets)
    except Exception:  # noqa: BLE001
        return Histogram  # type: ignore[return-value]


TTFT = _hist("oms_ttft_ms", "首字延迟", [0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3, 5, 10])
TURN_RTT = _hist("oms_turn_rtt_ms", "端到端轮次 RTT",
                 [0.1, 0.3, 0.6, 1.2, 2.5, 5, 10, 20, 40])
BARGEIN = _hist("oms_bargein_latency_ms", "打断响应时延",
                [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5])
VAD_LATENCY = _hist("oms_vad_latency_ms", "VAD 检测时延",
                    [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])

# ---------------- 会话 ----------------


def _gauge(name: str, doc: str) -> Gauge:
    try:
        return Gauge(name, doc, ["provider"])
    except Exception:  # noqa: BLE001
        return Gauge  # type: ignore[return-value]


ACTIVE_SESSIONS = _gauge("oms_sessions_active", "活跃会话数")
TOTAL_SESSIONS = _gauge("oms_sessions_total", "累计会话数")
ACTIVE_TURNS = _gauge("oms_turns_active", "进行中轮次数")

# ---------------- 事件 ----------------


def _counter(name: str, doc: str) -> Counter:
    try:
        return Counter(name, doc, ["provider", "kind"])
    except Exception:  # noqa: BLE001
        return Counter  # type: ignore[return-value]


EVENTS_TOTAL = _counter("oms_events_total", "事件总数")
TOOL_CALLS = _counter("oms_tool_calls_total", "工具调用数", )
TOOL_FAILURES = _counter("oms_tool_failures_total", "工具失败数")
MEMORY_INJECTIONS = _counter("oms_memory_injections_total", "记忆注入数")
MEMORY_SILENT = _counter("oms_memory_silent_total", "记忆沉默决策数")
INTERRUPTIONS = _counter("oms_interruptions_total", "打断次数")
PROVIDER_ERRORS = _counter("oms_provider_errors_total", "Provider 错误数")
DROPPED_AUDIO = _counter("oms_dropped_audio_total", "丢弃音频块数")


def start_metrics_server(port: int = 9464) -> None:
    """启动独立 metrics HTTP 端口（可选）。"""
    if start_http_server is not None:
        start_http_server(port)

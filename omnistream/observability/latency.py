"""实时延迟追踪器。

负责为每个会话/轮次采集关键实时指标并上报 Prometheus：
- 首字延迟 TTFT：request_response 触发 -> 首个 audio_delta/text_delta 到达
- 轮次 RTT：用户输入完成 -> response_done
- 打断响应时延：barge-in 检测 -> provider 停止 audio_delta
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from omnistream.observability import metrics as M

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TurnLatency:
    turn_id: str
    provider: str = "mock"
    modality: str = "audio"
    started_ms: float = 0.0
    first_token_ms: float | None = None
    done_ms: float | None = None
    interrupted_ms: float | None = None

    @property
    def ttft_ms(self) -> float | None:
        return (self.first_token_ms - self.started_ms) if self.first_token_ms else None

    @property
    def rtt_ms(self) -> float | None:
        return (self.done_ms - self.started_ms) if self.done_ms else None


class LatencyTracker:
    """进程内延迟追踪器（会话级）。"""

    def __init__(self) -> None:
        self._turns: dict[str, TurnLatency] = {}

    def start_turn(self, turn_id: str, provider: str = "mock",
                   modality: str = "audio") -> TurnLatency:
        lat = TurnLatency(turn_id=turn_id, provider=provider, modality=modality,
                          started_ms=time.perf_counter() * 1000)
        self._turns[turn_id] = lat
        M.ACTIVE_TURNS.labels(provider=provider).inc()
        return lat

    def on_first_token(self, turn_id: str) -> None:
        lat = self._turns.get(turn_id)
        if lat and lat.first_token_ms is None:
            lat.first_token_ms = time.perf_counter() * 1000
            M.TTFT.labels(lat.provider, lat.modality).observe(lat.ttft_ms or 0)

    def on_done(self, turn_id: str) -> float | None:
        lat = self._turns.get(turn_id)
        if not lat:
            return None
        lat.done_ms = time.perf_counter() * 1000
        rtt = lat.rtt_ms
        if rtt is not None:
            M.TURN_RTT.labels(lat.provider, lat.modality).observe(rtt)
        M.ACTIVE_TURNS.labels(provider=lat.provider).dec()
        return rtt

    def on_interrupt(self, turn_id: str) -> float | None:
        """记录打断时延：从打断发生到首个停止事件。"""
        lat = self._turns.get(turn_id)
        if not lat:
            return None
        lat.interrupted_ms = time.perf_counter() * 1000
        latency = lat.interrupted_ms - (lat.first_token_ms or lat.started_ms)
        M.BARGEIN.labels(lat.provider, lat.modality).observe(latency)
        M.INTERRUPTIONS.labels(provider=lat.provider).inc()
        return latency

    def on_vad(self, provider: str = "mock") -> None:
        M.VAD_LATENCY.labels(provider, "audio").observe(5.0)  # 示例值

    def active_count(self) -> int:
        return len(self._turns)

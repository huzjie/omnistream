"""零依赖 Mock Provider 包。

无 API Key、无 GPU、无网络即可跑通完整全双工链路，是本项目"克隆即可演示"
承诺的兜底实现，也是所有集成测试的默认后端。
"""
from __future__ import annotations

from omnistream.providers.mock.dsp_voice import (
    estimate_duration_ms,
    synthesize,
    synthesize_stream,
    text_to_phonemes,
)
from omnistream.providers.mock.latency_model import LatencyModel, LatencyProfile
from omnistream.providers.mock.mock_provider import MOCK_CAPABILITIES, MockRealtimeProvider
from omnistream.providers.mock.scripts import PlannedToolCall, ResponsePlan, ScriptEngine

__all__ = [
    "MOCK_CAPABILITIES",
    "LatencyModel",
    "LatencyProfile",
    "MockRealtimeProvider",
    "PlannedToolCall",
    "ResponsePlan",
    "ScriptEngine",
    "estimate_duration_ms",
    "synthesize",
    "synthesize_stream",
    "text_to_phonemes",
]

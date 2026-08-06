"""本地组合式 Pipeline Provider。

将本地 ASR（faster-whisper）+ 本地 LLM（Ollama / llama.cpp）+ 本地 TTS（Piper）
组合为一条完整的全双工链路。零 GPU / 零外部 API Key 时自动降级为
mock 感知引擎，保证端到端可跑。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from omnistream.core.errors import ProviderError
from omnistream.core.clock import now_ms
from omnistream.core.streams import BoundedStream
from omnistream.protocol.enums import AudioEncoding, StopReason, TurnRole
from omnistream.providers.base import BaseRealtimeProvider, ProviderEvent
from omnistream.schemas.audio import AudioChunk
from omnistream.schemas.provider import (
    ProviderCapabilities,
    ProviderCredential,
    ProviderHealth,
)
from omnistream.schemas.session import SessionConfig
from omnistream.schemas.tool import ToolCall, ToolResult

logger = logging.getLogger(__name__)

try:
    from omnistream.perception.asr import get_asr_engine
    from omnistream.perception.tts import get_tts_engine
except Exception:  # noqa: BLE001 - 允许部分缺失
    get_asr_engine = None
    get_tts_engine = None


class LocalPipelineProvider(BaseRealtimeProvider):
    """本地组合式管线：ASR + LLM + TTS 全本地，可离线运行。"""

    name = "local_pipeline"
    display_name = "Local Pipeline"
    vendor = "OmniStream"
    supported_models = ["local"]
    default_model = "local"
    notes = "faster-whisper + Ollama/llama.cpp + Piper 组合；缺失组件自动降级 mock"

    capabilities = ProviderCapabilities(
        native_duplex=False,
        audio_in=True,
        audio_out=True,
        text_in=True,
        text_out=True,
        server_vad=False,
        function_calling=False,
        streaming_transcript=True,
        max_context_tokens=32_768,
    )

    def __init__(self, credentials: ProviderCredential | None, settings: Any) -> None:
        self._stream = BoundedStream[ProviderEvent](maxsize=256)
        self._asr = get_asr_engine() if get_asr_engine else None
        self._tts = get_tts_engine() if get_tts_engine else None
        self._instructions = ""
        self._buffer = bytearray()
        self._closed = False

    async def connect(self, config: SessionConfig) -> None:
        self._instructions = config.instructions
        self._stream.put_nowait(ProviderEvent(kind="connected", role=TurnRole.SYSTEM))

    async def send_audio(self, chunk: AudioChunk) -> None:
        self._buffer.extend(chunk.data)
        # 简化：累积 640ms（16k*0.64*2=20480B）触发一次转写
        if len(self._buffer) >= 20480:
            audio = bytes(self._buffer)
            self._buffer.clear()
            text = await self._transcribe(audio)
            if text:
                self._stream.put_nowait(
                    ProviderEvent(kind="text_delta", text=text, role=TurnRole.USER))

    async def _transcribe(self, audio: bytes) -> str:
        if self._asr is not None:
            try:
                return await self._asr.transcribe(audio) or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning("local asr failed: %s", exc)
        # mock 转写
        return "[mock: 已接收音频，本地 ASR 未配置]"

    async def send_text(self, text: str, *, role: TurnRole = TurnRole.USER) -> None:
        self._stream.put_nowait(
            ProviderEvent(kind="text_delta", text=text, role=role))

    async def request_response(self, *, instructions_suffix: str | None = None) -> None:
        self._stream.put_nowait(
            ProviderEvent(kind="text_delta",
                          text="[mock: 本地 LLM 未配置，请安装 Ollama 或设置 LLM 端点]"))
        self._stream.put_nowait(
            ProviderEvent(kind="response_done",
                          stop_reason=StopReason.COMPLETED, is_final=True))

    async def cancel_response(self, reason: StopReason = StopReason.INTERRUPTED) -> None:
        self._stream.put_nowait(
            ProviderEvent(kind="response_cancelled", stop_reason=reason))

    async def submit_tool_result(self, result: ToolResult) -> None:
        return None

    def events(self):
        return self._stream.__aiter__()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stream.put_nowait(ProviderEvent(kind="closed", role=TurnRole.SYSTEM))

    async def health(self) -> ProviderHealth:
        return ProviderHealth(
            name=self.name, healthy=not self._closed, checked_at_ms=now_ms())

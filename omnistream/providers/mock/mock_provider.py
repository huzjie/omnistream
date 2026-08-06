"""MockRealtimeProvider —— 无 API Key、无 GPU、无网络即可跑通完整全双工链路。

实现要点
--------
* 用 ``dsp_voice.synthesize()`` 产出**真实可听**的确定性语音（共振峰合成，纯 numpy）。
* 用 ``latency_model`` 模拟真实 provider 的 TTFT/TTFA 分布，让延迟看板有意义。
* 完整支持 cancel（barge-in）、function calling、视频帧接收与"看到了什么"的回应。
* 行为完全确定：相同输入 → 相同输出，便于演示与回归。
"""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from omnistream.core.clock import now_ms
from omnistream.core.ids import new_tool_call_id
from omnistream.core.streams import BoundedStream, OverflowPolicy
from omnistream.protocol.enums import AudioEncoding, StopReason, TurnRole
from omnistream.providers.base import BaseRealtimeProvider, ProviderEvent
from omnistream.providers.mock.dsp_voice import synthesize_stream
from omnistream.providers.mock.latency_model import LatencyModel
from omnistream.providers.mock.scripts import ScriptEngine
from omnistream.schemas.audio import AudioChunk
from omnistream.schemas.provider import ProviderCapabilities
from omnistream.schemas.session import SessionConfig
from omnistream.schemas.tool import ToolCall, ToolResult
from omnistream.schemas.video import VideoFrame

MOCK_CAPABILITIES = ProviderCapabilities(
    native_duplex=True, audio_in=True, audio_out=True, video_in=True,
    text_in=True, text_out=True, server_vad=False, function_calling=True,
    parallel_tools=True, interruption=True, proactive=True,
    streaming_transcript=True, max_context_tokens=128_000,
    input_audio_formats=[AudioEncoding.PCM16],
    output_audio_formats=[AudioEncoding.PCM16],
    input_sample_rates=[16000], output_sample_rates=[24000], max_video_fps=2.0,
)


class MockRealtimeProvider(BaseRealtimeProvider):
    name = "mock"
    capabilities = MOCK_CAPABILITIES

    def __init__(self, *, latency: LatencyModel | None = None, seed: int = 42) -> None:
        self._events: BoundedStream[ProviderEvent] = BoundedStream(
            maxsize=512, policy=OverflowPolicy.BLOCK, name="mock_provider_events")
        self._latency = latency or LatencyModel()
        self._script = ScriptEngine(seed=seed)
        self._config: SessionConfig | None = None
        self._gen_task: asyncio.Task[None] | None = None
        self._pending_text: list[str] = []
        self._last_frame_desc: str | None = None
        self._closed = False
        self._cancel_flag = asyncio.Event()

    # ---------------- 生命周期 ----------------

    async def connect(self, config: SessionConfig) -> None:
        self._config = config
        self._script.configure(language=config.language, tools=[t.name for t in config.tools])
        await self._events.put(ProviderEvent(kind="connected"))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._abort_generation()
        await self._events.put(ProviderEvent(kind="closed"))
        self._events.close()

    # ---------------- 输入 ----------------

    async def send_audio(self, chunk: AudioChunk) -> None:
        # Mock 不做真实 ASR，只累计时长；转写由 perception 层的 mock_asr 负责
        return None

    async def send_video(self, frame: VideoFrame) -> None:
        self._last_frame_desc = self._script.describe_frame(frame)

    async def send_text(self, text: str, *, role: TurnRole = TurnRole.USER) -> None:
        if role is TurnRole.USER and text.strip():
            self._pending_text.append(text.strip())

    async def commit_input(self) -> None:
        return None

    # ---------------- 生成 ----------------

    async def request_response(self, *, instructions_suffix: str | None = None) -> None:
        await self._abort_generation()
        self._cancel_flag.clear()
        user_text = " ".join(self._pending_text)
        self._pending_text.clear()
        self._gen_task = asyncio.create_task(
            self._generate(user_text, instructions_suffix), name="mock_generate")

    async def cancel_response(self, reason: StopReason = StopReason.INTERRUPTED) -> None:
        self._cancel_flag.set()          # 生成循环每帧检查，保证 <100ms 停止
        await self._abort_generation()
        await self._events.put(ProviderEvent(kind="response_cancelled", stop_reason=reason))

    async def submit_tool_result(self, result: ToolResult) -> None:
        self._script.absorb_tool_result(result)
        await self.request_response()

    # ---------------- 输出 ----------------

    def events(self) -> AsyncIterator[ProviderEvent]:
        return self._events.__aiter__()

    # ---------------- 内部 ----------------

    async def _abort_generation(self) -> None:
        task, self._gen_task = self._gen_task, None
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _generate(self, user_text: str, suffix: str | None) -> None:
        try:
            plan = self._script.plan(user_text, vision=self._last_frame_desc, memory_hint=suffix)

            # 1) 可选工具调用
            for spec in plan.tool_calls:
                await self._events.put(ProviderEvent(
                    kind="tool_call",
                    tool_call=ToolCall(id=new_tool_call_id(), turn_id="", name=spec.name,
                                       arguments=spec.arguments),
                ))
            if plan.tool_calls:
                return  # 等 submit_tool_result 再续

            # 2) 模拟思考延迟（首字延迟）
            await asyncio.sleep(self._latency.ttft_s())
            if self._cancel_flag.is_set():
                return

            # 3) 文本增量 + 音频增量并行推进
            pts = 0
            for sentence in plan.sentences:
                for token in self._script.tokenize(sentence):
                    if self._cancel_flag.is_set():
                        return
                    await self._events.put(ProviderEvent(kind="text_delta", text=token))
                    await asyncio.sleep(self._latency.token_s())

                async for pcm in synthesize_stream(sentence, sample_rate=24000, frame_ms=20):
                    if self._cancel_flag.is_set():
                        return
                    await self._events.put(ProviderEvent(
                        kind="audio_delta",
                        audio=AudioChunk(pts_ms=pts, data=pcm, sample_rate=24000,
                                         channels=1, encoding=AudioEncoding.PCM16),
                    ))
                    pts += 20
                    await asyncio.sleep(0.02 * self._latency.realtime_factor)

            await self._events.put(ProviderEvent(kind="text_done", text=plan.full_text, is_final=True))
            await self._events.put(ProviderEvent(kind="audio_done", is_final=True))
            await self._events.put(ProviderEvent(
                kind="usage", usage={"input_tokens": plan.input_tokens,
                                     "output_tokens": plan.output_tokens}))
            await self._events.put(ProviderEvent(
                kind="response_done", stop_reason=StopReason.COMPLETED, ts_ms=now_ms()))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # 任何异常都要转成事件，不能让消费者挂死
            await self._events.put(ProviderEvent(kind="error", error=exc))

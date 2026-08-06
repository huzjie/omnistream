"""TurnManager —— 轮次创建、编号、延迟指标采集与结束原因归因。

时间基准（严格遵守 §7.6）::

    所有 *_ms 指标都用 time.monotonic_ns() 计算，不受 NTP 影响。
    锚点定义：
        speech_start_ms      = VAD 判定开口        相对轮次开始
        speech_end_ms        = 端点确认说完        相对轮次开始
        asr_first_partial_ms = 首个增量转写        相对轮次开始
        asr_final_ms         = 转写终稿            相对轮次开始
        provider_request_ms  = 发出 request_response 相对轮次开始
        ttft_ms              = 首个文本 token      相对 provider_request
        ttfa_ms              = 首个音频字节        相对 provider_request  ← 关键 SLO
        e2e_ms               = 用户说完 → 助手出声  相对 speech_end        ← 关键 SLO
        barge_in_ms          = 用户开口 → 播放停止  独立测量               ← 关键 SLO
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import structlog

from omnistream.core.clock import now_ns, utcnow
from omnistream.core.ids import new_turn_id
from omnistream.protocol.enums import StopReason, TurnRole, TurnState
from omnistream.schemas.turn import Turn, TurnMetrics

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class _Anchors:
    """一轮内部的单调时钟锚点（纳秒）。"""

    turn_start_ns: int = 0
    speech_start_ns: int = 0
    speech_end_ns: int = 0
    provider_request_ns: int = 0
    first_text_ns: int = 0
    first_audio_ns: int = 0
    tool_start_ns: int = 0
    memory_start_ns: int = 0
    vision_start_ns: int = 0
    extra: dict[str, int] = field(default_factory=dict)


def _delta_ms(start_ns: int, end_ns: int) -> float | None:
    if start_ns <= 0 or end_ns <= 0 or end_ns < start_ns:
        return None
    return round((end_ns - start_ns) / 1e6, 3)


class TurnManager:
    """管理一个会话内的全部轮次。"""

    def __init__(self, session_id: str, *, history_size: int = 100) -> None:
        self.session_id = session_id
        self._index = 0
        self._current: Turn | None = None
        self._anchors = _Anchors()
        self._history: deque[Turn] = deque(maxlen=max(1, int(history_size)))
        self._completed = 0
        self._interrupted = 0

    # ------------------------------------------------------------------
    # 轮次生命周期
    # ------------------------------------------------------------------

    def start_turn(self, *, role: TurnRole = TurnRole.USER,
                   state: TurnState = TurnState.LISTENING) -> Turn:
        """开启新一轮。若上一轮未关闭，先以 ``CANCELLED`` 归因关闭它。"""
        if self._current is not None and self._current.ended_at is None:
            self.close_turn(StopReason.CANCELLED, note="superseded_by_new_turn")

        self._index += 1
        turn = Turn(
            id=new_turn_id(),
            session_id=self.session_id,
            index=self._index,
            role=role,
            state=state,
            started_at=utcnow(),
            metrics=TurnMetrics(),
        )
        self._current = turn
        self._anchors = _Anchors(turn_start_ns=now_ns())
        log.info("turn_started", session_id=self.session_id, turn_id=turn.id,
                 turn_index=turn.index, role=str(role), state=str(state))
        return turn

    def ensure_turn(self, *, role: TurnRole = TurnRole.USER,
                    state: TurnState = TurnState.LISTENING) -> Turn:
        """获取当前轮；不存在或已结束则新建。"""
        if self._current is None or self._current.ended_at is not None:
            return self.start_turn(role=role, state=state)
        return self._current

    def close_turn(self, stop_reason: StopReason, *, note: str = "",
                   interrupted_at_ms: int | None = None) -> Turn | None:
        """关闭当前轮并归因。"""
        turn = self._current
        if turn is None or turn.ended_at is not None:
            return turn

        turn.ended_at = utcnow()
        turn.stop_reason = stop_reason
        if interrupted_at_ms is not None:
            turn.interrupted_at_ms = int(interrupted_at_ms)
        self._finalize_metrics(turn)

        if stop_reason is StopReason.INTERRUPTED:
            self._interrupted += 1
        elif stop_reason is StopReason.COMPLETED:
            self._completed += 1

        self._history.append(turn)
        self._current = None
        log.info(
            "turn_completed",
            session_id=self.session_id,
            turn_id=turn.id,
            turn_index=turn.index,
            stop_reason=str(stop_reason),
            note=note,
            ttft_ms=turn.metrics.ttft_ms,
            ttfa_ms=turn.metrics.ttfa_ms,
            e2e_ms=turn.metrics.e2e_ms,
            barge_in_ms=turn.metrics.barge_in_ms,
            text_chars=len(turn.text),
        )
        return turn

    # ------------------------------------------------------------------
    # 打点
    # ------------------------------------------------------------------

    def mark_speech_start(self) -> None:
        if self._current is None:
            return
        self._anchors.speech_start_ns = now_ns()
        self._current.metrics.speech_start_ms = _delta_ms(
            self._anchors.turn_start_ns, self._anchors.speech_start_ns)

    def mark_speech_end(self) -> None:
        if self._current is None:
            return
        self._anchors.speech_end_ns = now_ns()
        self._current.metrics.speech_end_ms = _delta_ms(
            self._anchors.turn_start_ns, self._anchors.speech_end_ns)

    def mark_asr_partial(self, text: str) -> None:
        if self._current is None:
            return
        if self._current.metrics.asr_first_partial_ms is None:
            self._current.metrics.asr_first_partial_ms = _delta_ms(
                self._anchors.turn_start_ns, now_ns())
        self._current.text = text

    def mark_asr_final(self, text: str) -> None:
        if self._current is None:
            return
        self._current.metrics.asr_final_ms = _delta_ms(self._anchors.turn_start_ns, now_ns())
        self._current.text = text

    def mark_provider_request(self) -> None:
        if self._current is None:
            return
        self._anchors.provider_request_ns = now_ns()
        self._current.metrics.provider_request_ms = _delta_ms(
            self._anchors.turn_start_ns, self._anchors.provider_request_ns)

    def mark_first_text(self) -> None:
        if self._current is None or self._anchors.first_text_ns:
            return
        self._anchors.first_text_ns = now_ns()
        self._current.metrics.ttft_ms = _delta_ms(
            self._anchors.provider_request_ns or self._anchors.turn_start_ns,
            self._anchors.first_text_ns)

    def mark_first_audio(self) -> None:
        """首个音频字节 —— TTFA 与 E2E 两个关键 SLO 都在这里落定。"""
        if self._current is None or self._anchors.first_audio_ns:
            return
        self._anchors.first_audio_ns = now_ns()
        metrics = self._current.metrics
        metrics.ttfa_ms = _delta_ms(
            self._anchors.provider_request_ns or self._anchors.turn_start_ns,
            self._anchors.first_audio_ns)
        metrics.e2e_ms = _delta_ms(
            self._anchors.speech_end_ns or self._anchors.turn_start_ns,
            self._anchors.first_audio_ns)

    def mark_barge_in(self, latency_ms: float) -> None:
        if self._current is None:
            return
        self._current.metrics.barge_in_ms = round(float(latency_ms), 3)

    def begin_tool(self) -> None:
        self._anchors.tool_start_ns = now_ns()

    def end_tool(self) -> None:
        if self._current is None or not self._anchors.tool_start_ns:
            return
        elapsed = _delta_ms(self._anchors.tool_start_ns, now_ns()) or 0.0
        self._current.metrics.tool_ms = round((self._current.metrics.tool_ms or 0.0) + elapsed, 3)
        self._anchors.tool_start_ns = 0

    def begin_memory(self) -> None:
        self._anchors.memory_start_ns = now_ns()

    def end_memory(self) -> None:
        if self._current is None or not self._anchors.memory_start_ns:
            return
        self._current.metrics.memory_ms = _delta_ms(self._anchors.memory_start_ns, now_ns())
        self._anchors.memory_start_ns = 0

    def begin_vision(self) -> None:
        self._anchors.vision_start_ns = now_ns()

    def end_vision(self) -> None:
        if self._current is None or not self._anchors.vision_start_ns:
            return
        self._current.metrics.vision_ms = _delta_ms(self._anchors.vision_start_ns, now_ns())
        self._anchors.vision_start_ns = 0

    def add_tokens(self, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
        if self._current is None:
            return
        self._current.metrics.input_tokens += int(input_tokens)
        self._current.metrics.output_tokens += int(output_tokens)

    def append_text(self, delta: str) -> None:
        if self._current is None or not delta:
            return
        self._current.text += delta

    def set_text(self, text: str) -> None:
        if self._current is None:
            return
        self._current.text = text

    def add_audio_ms(self, ms: float) -> None:
        if self._current is None:
            return
        self._current.audio_ms += int(ms)

    def set_state(self, state: TurnState) -> None:
        if self._current is None:
            return
        self._current.state = state

    def record_tool_call(self, call_id: str) -> None:
        if self._current is None or not call_id:
            return
        if call_id not in self._current.tool_call_ids:
            self._current.tool_call_ids.append(call_id)

    def record_memory_injection(self, item_id: str) -> None:
        if self._current is None or not item_id:
            return
        if item_id not in self._current.memory_injected_ids:
            self._current.memory_injected_ids.append(item_id)

    # ------------------------------------------------------------------
    # 只读
    # ------------------------------------------------------------------

    @property
    def current(self) -> Turn | None:
        return self._current

    @property
    def current_id(self) -> str:
        return self._current.id if self._current else ""

    @property
    def index(self) -> int:
        return self._index

    @property
    def history(self) -> list[Turn]:
        return list(self._history)

    @property
    def completed_count(self) -> int:
        return self._completed

    @property
    def interrupted_count(self) -> int:
        return self._interrupted

    @property
    def speech_start_ns(self) -> int:
        return self._anchors.speech_start_ns

    def elapsed_ms(self) -> float:
        """当前轮已持续毫秒数（无当前轮返回 0）。"""
        if self._current is None or not self._anchors.turn_start_ns:
            return 0.0
        return (now_ns() - self._anchors.turn_start_ns) / 1e6

    def _finalize_metrics(self, turn: Turn) -> None:
        """结束时补齐仍为空的派生指标，避免看板出现空洞。"""
        metrics = turn.metrics
        if metrics.e2e_ms is None and self._anchors.speech_end_ns and self._anchors.first_audio_ns:
            metrics.e2e_ms = _delta_ms(self._anchors.speech_end_ns, self._anchors.first_audio_ns)
        if metrics.ttft_ms is None and self._anchors.first_text_ns:
            metrics.ttft_ms = _delta_ms(
                self._anchors.provider_request_ns or self._anchors.turn_start_ns,
                self._anchors.first_text_ns)

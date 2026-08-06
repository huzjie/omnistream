"""barge-in 检测器：在助手说话/思考期间识别用户插话。

双阈值防误触（§4.1）：

1. **能量闸门** —— 帧能量必须超过 ``BargeInConfig.energy_threshold_db``（默认 -38 dBFS），
   过滤扬声器回声泄漏与环境底噪；额外要求信噪比高于 ``min_snr_db``，
   使其在嘈杂环境下自动抬高门槛。
2. **时长闸门** —— 连续满足能量闸门的语音帧累计时长必须超过
   ``BargeInConfig.min_duration_ms``（默认 160ms），过滤咳嗽、键盘声、单音节语气词。

时序对齐 §4.1::

    ING->>VAD: 帧能量 -31dB > 阈值 -38dB
    VAD-->>BID: is_speech=True 且 SM.state==SPEAKING
    BID->>BID: 持续 170ms > min_duration 160ms → 确认打断
    BID->>SM: transition(INTERRUPTED)

检测器本身完全同步、零 IO，单帧处理耗时在微秒级，保证 barge-in 链路
（用户开口 → 播放停止）能压进 200ms 的 SLO 预算。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import structlog

from omnistream.core.clock import now_ns
from omnistream.duplex.vad.base import VADResult
from omnistream.protocol.enums import TurnState
from omnistream.schemas.session import BargeInConfig

log = structlog.get_logger(__name__)

#: 助手正在输出（或即将输出）音频的状态，只有这些状态才可能被打断。
DEFAULT_ACTIVE_STATES: Final[tuple[TurnState, ...]] = (TurnState.SPEAKING, TurnState.THINKING)

#: 允许的语音内部微停顿；超过该时长的非语音帧才会清空累计器。
DEFAULT_GAP_TOLERANCE_MS: Final[float] = 60.0

#: 高置信度阈值：VAD 概率高于此值时可豁免信噪比要求。
HIGH_CONFIDENCE_PROB: Final[float] = 0.9


@dataclass(slots=True, frozen=True)
class BargeInSignal:
    """一次被确认的打断信号。"""

    #: 触发确认的那一帧的 pts（毫秒，相对会话起点）
    detected_at_ms: int
    #: 用户开口的第一帧 pts（毫秒），用于把截断点回溯到语音真正的起点
    onset_pts_ms: int
    #: 用户开口时刻的单调时钟（纳秒），barge_in_ms 以此为基准计算
    onset_mono_ns: int
    #: 从开口到确认累计的有效语音时长（毫秒）
    duration_ms: float
    #: 确认帧的能量（dBFS）
    energy_db: float
    #: 确认帧的峰值能量（dBFS），取累计窗口内的最大值
    peak_energy_db: float
    #: 确认帧的 VAD 概率
    prob: float
    #: 累计的有效帧数
    frames: int
    #: 触发时的轮次状态
    state: TurnState = TurnState.SPEAKING
    #: VAD 后端名，便于归因误触
    backend: str = "unknown"

    @property
    def confirm_latency_ms(self) -> float:
        """从开口到确认所经过的真实时间（含累计与容差 gap）。"""
        return max(0.0, (now_ns() - self.onset_mono_ns) / 1e6)


@dataclass(slots=True)
class BargeInStats:
    """检测器累计统计，供 /metrics 与控制台展示。"""

    frames_seen: int = 0
    qualified_frames: int = 0
    candidates: int = 0
    confirmed: int = 0
    aborted_by_gap: int = 0
    suppressed_by_state: int = 0
    max_energy_db: float = -120.0

    def as_dict(self) -> dict[str, float]:
        return {
            "frames_seen": float(self.frames_seen),
            "qualified_frames": float(self.qualified_frames),
            "candidates": float(self.candidates),
            "confirmed": float(self.confirmed),
            "aborted_by_gap": float(self.aborted_by_gap),
            "suppressed_by_state": float(self.suppressed_by_state),
            "max_energy_db": self.max_energy_db,
        }


class BargeInDetector:
    """speaking/thinking 态下的用户插话检测器。

    用法::

        det = BargeInDetector(config=session.config.barge_in, session_id=sid)
        for chunk in frames:
            vr = vad.process_frame(chunk)
            sig = det.observe(vr, state=machine.state)
            if sig is not None:
                ...  # 交给 BargeInPolicy + BargeInExecutor

    检测器在确认一次打断后会自动"缴械"（``armed=False``），
    直到 :meth:`rearm`、:meth:`reset` 被调用，或轮次状态离开可打断状态集合。
    """

    def __init__(
        self,
        config: BargeInConfig | None = None,
        *,
        session_id: str = "",
        active_states: tuple[TurnState, ...] = DEFAULT_ACTIVE_STATES,
        gap_tolerance_ms: float = DEFAULT_GAP_TOLERANCE_MS,
        min_snr_db: float = 4.0,
    ) -> None:
        self.config = config or BargeInConfig()
        self.session_id = session_id
        self.active_states = tuple(active_states)
        self.gap_tolerance_ms = float(gap_tolerance_ms)
        self.min_snr_db = float(min_snr_db)

        self.stats = BargeInStats()

        self._armed = True
        self._accum_ms = 0.0
        self._gap_ms = 0.0
        self._frames = 0
        self._onset_pts_ms = 0
        self._onset_mono_ns = 0
        self._peak_db = -120.0
        self._last_state: TurnState = TurnState.IDLE

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """配置层面是否启用打断。"""
        return bool(self.config.enabled) and self.config.mode != "off"

    @property
    def armed(self) -> bool:
        """是否处于待命状态（未在本轮内已经触发过）。"""
        return self._armed

    @property
    def accumulated_ms(self) -> float:
        """当前候选语音的累计时长（毫秒）。"""
        return self._accum_ms

    @property
    def progress(self) -> float:
        """距离确认打断的进度 [0,1]，控制台可用于画进度条。"""
        need = max(1.0, float(self.config.min_duration_ms))
        return min(1.0, self._accum_ms / need)

    # ------------------------------------------------------------------
    # 核心
    # ------------------------------------------------------------------

    def observe(self, result: VADResult, *, state: TurnState) -> BargeInSignal | None:
        """喂入一帧 VAD 结果，返回被确认的打断信号（否则 ``None``）。

        Args:
            result: 单帧 VAD 判定结果。
            state: 当前轮次状态机状态。

        Returns:
            确认打断时返回 :class:`BargeInSignal`，否则 ``None``。
        """
        self.stats.frames_seen += 1

        # 状态离开可打断集合：清空候选并自动重新待命。
        if state not in self.active_states:
            if self._accum_ms > 0.0:
                self.stats.suppressed_by_state += 1
            self._clear_candidate()
            if state != self._last_state:
                self._armed = True
            self._last_state = state
            return None

        self._last_state = state

        if not self.enabled or not self._armed:
            self._clear_candidate()
            return None

        if result.energy_db > self.stats.max_energy_db:
            self.stats.max_energy_db = result.energy_db

        if self._qualifies(result):
            return self._on_qualified_frame(result, state)

        return self._on_unqualified_frame(result)

    def _qualifies(self, result: VADResult) -> bool:
        """单帧是否通过能量闸门。"""
        if not result.is_speech:
            return False
        if result.energy_db < float(self.config.energy_threshold_db):
            return False
        if result.prob >= HIGH_CONFIDENCE_PROB:
            return True
        return result.snr_db >= self.min_snr_db

    def _on_qualified_frame(self, result: VADResult, state: TurnState) -> BargeInSignal | None:
        if self._frames == 0:
            self._onset_pts_ms = int(result.pts_ms)
            self._onset_mono_ns = now_ns()
            self.stats.candidates += 1

        self._frames += 1
        self._accum_ms += float(result.duration_ms)
        self._gap_ms = 0.0
        self._peak_db = max(self._peak_db, result.energy_db)
        self.stats.qualified_frames += 1

        if self._accum_ms < float(self.config.min_duration_ms):
            return None

        signal = BargeInSignal(
            detected_at_ms=int(result.pts_ms),
            onset_pts_ms=self._onset_pts_ms,
            onset_mono_ns=self._onset_mono_ns,
            duration_ms=self._accum_ms,
            energy_db=result.energy_db,
            peak_energy_db=self._peak_db,
            prob=result.prob,
            frames=self._frames,
            state=state,
            backend=result.backend,
        )
        self._armed = False
        self.stats.confirmed += 1
        self._clear_candidate()

        log.info(
            "barge_in.confirmed",
            session_id=self.session_id,
            state=str(state),
            duration_ms=round(signal.duration_ms, 1),
            energy_db=round(signal.energy_db, 1),
            peak_db=round(signal.peak_energy_db, 1),
            onset_pts_ms=signal.onset_pts_ms,
            backend=signal.backend,
        )
        return signal

    def _on_unqualified_frame(self, result: VADResult) -> None:
        if self._frames == 0:
            return None
        self._gap_ms += float(result.duration_ms)
        if self._gap_ms > self.gap_tolerance_ms:
            self.stats.aborted_by_gap += 1
            self._clear_candidate()
        return None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def rearm(self) -> None:
        """重新待命，允许下一次打断被确认。"""
        self._armed = True
        self._clear_candidate()

    def reset(self) -> None:
        """整体复位（换轮次/换会话时调用），统计计数保留。"""
        self._armed = True
        self._last_state = TurnState.IDLE
        self._clear_candidate()

    def _clear_candidate(self) -> None:
        self._accum_ms = 0.0
        self._gap_ms = 0.0
        self._frames = 0
        self._onset_pts_ms = 0
        self._onset_mono_ns = 0
        self._peak_db = -120.0

    def __repr__(self) -> str:
        return (
            f"<BargeInDetector mode={self.config.mode} "
            f"thr={self.config.energy_threshold_db}dB "
            f"min={self.config.min_duration_ms}ms armed={self._armed}>"
        )


__all__ = [
    "DEFAULT_ACTIVE_STATES",
    "DEFAULT_GAP_TOLERANCE_MS",
    "HIGH_CONFIDENCE_PROB",
    "BargeInDetector",
    "BargeInSignal",
    "BargeInStats",
]

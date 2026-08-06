"""音频时间轴：pts 生成、空洞检测与时钟漂移监控。

为什么需要它
------------
上行链路的 pts 有三个互相打架的来源：

1. **网络到达时刻** —— 受抖动污染，不能直接当时基。
2. **发送端自报 pts** —— 可信但可能跳变（客户端重连、页面挂起）。
3. **累计样本数** —— 最稳，但只有在没有丢帧的前提下才等于真实时间。

:class:`AudioTimeline` 以"累计样本数"为权威时基，用发送端 pts 做漂移观测：
小漂移记账并告警，大漂移直接抛 :class:`~omnistream.media.errors.ClockDriftError`
由上层决定重建会话。空洞（gap）会被显式识别并可用静音补齐，
保证下游 ASR 拿到的是**连续无洞**的时间轴。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from omnistream.media.audio.silence import silence_chunk
from omnistream.media.constants import (
    CLOCK_DRIFT_FATAL_MS,
    CLOCK_DRIFT_WARN_MS,
    DEFAULT_CHANNELS,
    DEFAULT_FRAME_MS,
    DEFAULT_INPUT_SAMPLE_RATE,
)
from omnistream.media.errors import ClockDriftError
from omnistream.protocol.enums import AudioEncoding
from omnistream.schemas.audio import AudioChunk

__all__ = ["AudioTimeline", "TimelineStats"]


@dataclass(slots=True)
class TimelineStats:
    """时间轴滚动统计。"""

    chunks: int = 0
    samples: int = 0
    gaps: int = 0
    gap_ms_total: float = 0.0
    overlaps: int = 0
    overlap_ms_total: float = 0.0
    rebases: int = 0
    drift_warnings: int = 0
    max_abs_drift_ms: float = 0.0
    last_drift_ms: float = 0.0
    rate_changes: int = 0
    filled_ms: float = 0.0
    history: list[float] = field(default_factory=list, repr=False)

    def as_dict(self) -> dict[str, float]:
        """转为可序列化字典。"""
        return {
            "chunks": float(self.chunks),
            "samples": float(self.samples),
            "gaps": float(self.gaps),
            "gap_ms_total": round(self.gap_ms_total, 2),
            "overlaps": float(self.overlaps),
            "overlap_ms_total": round(self.overlap_ms_total, 2),
            "rebases": float(self.rebases),
            "drift_warnings": float(self.drift_warnings),
            "max_abs_drift_ms": round(self.max_abs_drift_ms, 2),
            "last_drift_ms": round(self.last_drift_ms, 2),
            "rate_changes": float(self.rate_changes),
            "filled_ms": round(self.filled_ms, 2),
        }


class AudioTimeline:
    """以累计样本数为权威时基的音频时间轴。

    典型用法::

        timeline = AudioTimeline(sample_rate=16000)
        for chunk in incoming:
            fillers = timeline.fill_gap(chunk)      # 需要时补静音
            aligned = timeline.align(chunk)          # 改写为权威 pts
            yield from fillers
            yield aligned
    """

    #: 历史漂移采样保留点数。
    HISTORY_LIMIT: int = 240

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_INPUT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        frame_ms: int = DEFAULT_FRAME_MS,
        start_pts_ms: int = 0,
        gap_tolerance_ms: float = 5.0,
        drift_warn_ms: float = CLOCK_DRIFT_WARN_MS,
        drift_fatal_ms: float = CLOCK_DRIFT_FATAL_MS,
        strict: bool = False,
    ) -> None:
        """初始化时间轴。

        Args:
            sample_rate: 当前采样率。
            channels: 声道数。
            frame_ms: 标准帧长，仅用于补洞时的分片粒度。
            start_pts_ms: 时间轴起点。
            gap_tolerance_ms: 小于该值的错位视为正常抖动，不算空洞。
            drift_warn_ms: 漂移告警阈值。
            drift_fatal_ms: 漂移致命阈值，超过则抛异常（``strict=True`` 时）。
            strict: 为 ``True`` 时致命漂移抛 :class:`ClockDriftError`，
                否则自动 rebase 并计数。
        """
        self.sample_rate = max(1, int(sample_rate))
        self.channels = max(1, int(channels))
        self.frame_ms = max(1, int(frame_ms))
        self.start_pts_ms = int(start_pts_ms)
        self.gap_tolerance_ms = max(0.0, float(gap_tolerance_ms))
        self.drift_warn_ms = float(drift_warn_ms)
        self.drift_fatal_ms = float(drift_fatal_ms)
        self.strict = bool(strict)

        self._samples: int = 0
        self._base_pts_ms: float = float(start_pts_ms)
        self.stats = TimelineStats()

    # ------------------------------------------------------------------
    # 只读视图
    # ------------------------------------------------------------------

    @property
    def elapsed_ms(self) -> float:
        """已消费样本对应的时长（毫秒）。"""
        return self._samples * 1000.0 / self.sample_rate

    @property
    def next_pts_ms(self) -> int:
        """下一块音频**应当**具有的 pts。"""
        return int(round(self._base_pts_ms + self.elapsed_ms))

    @property
    def total_samples(self) -> int:
        """累计样本数（每声道）。"""
        return self._samples

    @property
    def drift_ms(self) -> float:
        """最近一次观测到的漂移（正数表示发送端跑快了）。"""
        return self.stats.last_drift_ms

    # ------------------------------------------------------------------
    # 时基推进
    # ------------------------------------------------------------------

    def advance(self, num_samples: int) -> int:
        """按样本数推进时间轴，返回推进前的 pts（即该段的起始 pts）。"""
        pts = self.next_pts_ms
        self._samples += max(0, int(num_samples))
        self.stats.samples = self._samples
        return pts

    def advance_ms(self, duration_ms: float) -> int:
        """按毫秒推进时间轴，返回该段起始 pts。"""
        return self.advance(int(round(self.sample_rate * max(0.0, duration_ms) / 1000.0)))

    def rebase(self, pts_ms: int) -> None:
        """把时间轴强制对齐到给定 pts（重连、seek、致命漂移恢复）。"""
        self._base_pts_ms = float(pts_ms)
        self._samples = 0
        self.stats.rebases += 1

    def set_sample_rate(self, sample_rate: int) -> None:
        """切换采样率并保持时间连续。

        做法：把已累计的时间折算进 ``base_pts``，然后样本计数归零，
        这样切换前后 :attr:`next_pts_ms` 不会跳变。
        """
        new_rate = max(1, int(sample_rate))
        if new_rate == self.sample_rate:
            return
        self._base_pts_ms += self.elapsed_ms
        self._samples = 0
        self.sample_rate = new_rate
        self.stats.rate_changes += 1

    def reset(self, *, start_pts_ms: int | None = None) -> None:
        """清空时间轴与统计。"""
        self.start_pts_ms = int(start_pts_ms) if start_pts_ms is not None else self.start_pts_ms
        self._base_pts_ms = float(self.start_pts_ms)
        self._samples = 0
        self.stats = TimelineStats()

    # ------------------------------------------------------------------
    # 观测与对齐
    # ------------------------------------------------------------------

    def observe(self, chunk: AudioChunk) -> float:
        """观测一个音频块的自报 pts，返回漂移（毫秒）。

        漂移 = 发送端 pts - 本地权威 pts。正值表示发送端时基跑在前面。

        Raises:
            ClockDriftError: ``strict=True`` 且漂移超过 :attr:`drift_fatal_ms`。
        """
        drift = float(chunk.pts_ms) - float(self.next_pts_ms)
        self.stats.last_drift_ms = drift
        self.stats.max_abs_drift_ms = max(self.stats.max_abs_drift_ms, abs(drift))
        self.stats.history.append(round(drift, 3))
        if len(self.stats.history) > self.HISTORY_LIMIT:
            del self.stats.history[: len(self.stats.history) - self.HISTORY_LIMIT]

        if abs(drift) >= self.drift_fatal_ms:
            if self.strict:
                raise ClockDriftError(
                    "audio timeline drift exceeded fatal threshold",
                    detail={
                        "drift_ms": round(drift, 2),
                        "fatal_ms": self.drift_fatal_ms,
                        "expected_pts_ms": self.next_pts_ms,
                        "actual_pts_ms": chunk.pts_ms,
                    },
                )
            self.rebase(chunk.pts_ms)
            self.stats.drift_warnings += 1
        elif abs(drift) >= self.drift_warn_ms:
            self.stats.drift_warnings += 1
        return drift

    def gap_ms(self, chunk: AudioChunk) -> float:
        """返回该块与当前时间轴之间的空洞时长；重叠时返回负值。"""
        return float(chunk.pts_ms) - float(self.next_pts_ms)

    def fill_gap(
        self,
        chunk: AudioChunk,
        *,
        max_fill_ms: float = 2000.0,
    ) -> list[AudioChunk]:
        """为即将到来的块补齐前置空洞，返回若干静音块。

        Args:
            chunk: 下一个待处理的音频块。
            max_fill_ms: 单次最多补多少毫秒；超出则改为 rebase（说明中断太久，
                补静音已无意义，反而会撑爆下游缓冲）。

        Returns:
            按时间升序排列的静音块列表；无需补洞时为空列表。
        """
        delta = self.gap_ms(chunk)
        if delta <= self.gap_tolerance_ms:
            if delta < -self.gap_tolerance_ms:
                self.stats.overlaps += 1
                self.stats.overlap_ms_total += abs(delta)
            return []

        self.stats.gaps += 1
        self.stats.gap_ms_total += delta
        if delta > max_fill_ms:
            self.rebase(chunk.pts_ms)
            return []

        fillers: list[AudioChunk] = []
        remaining = delta
        while remaining > 0.5:
            slot_ms = min(float(self.frame_ms), remaining)
            pts = self.advance_ms(slot_ms)
            fillers.append(
                silence_chunk(
                    pts_ms=pts,
                    duration_ms=slot_ms,
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                    encoding=(
                        chunk.encoding
                        if chunk.encoding in (AudioEncoding.PCM16, AudioEncoding.PCM_F32)
                        else AudioEncoding.PCM16
                    ),
                )
            )
            self.stats.filled_ms += slot_ms
            remaining -= slot_ms
        return fillers

    def align(self, chunk: AudioChunk) -> AudioChunk:
        """把块的 pts 改写为权威时基并推进时间轴。

        会顺带同步采样率/声道变化（例如上游中途换了编码参数）。
        """
        if chunk.sample_rate != self.sample_rate:
            self.set_sample_rate(chunk.sample_rate)
        if chunk.channels != self.channels:
            self.channels = max(1, chunk.channels)

        self.observe(chunk)
        pts = self.advance(chunk.num_samples)
        self.stats.chunks += 1
        if chunk.pts_ms == pts:
            return chunk
        return chunk.model_copy(update={"pts_ms": pts})

    def process(
        self,
        chunk: AudioChunk,
        *,
        fill_gaps: bool = True,
        max_fill_ms: float = 2000.0,
    ) -> list[AudioChunk]:
        """一步完成"补洞 + 对齐"，返回应当下发的块序列。"""
        result: list[AudioChunk] = []
        if fill_gaps:
            result.extend(self.fill_gap(chunk, max_fill_ms=max_fill_ms))
        result.append(self.align(chunk))
        return result

    # ------------------------------------------------------------------
    # 换算工具
    # ------------------------------------------------------------------

    def samples_to_ms(self, num_samples: int) -> float:
        """样本数 → 毫秒。"""
        return num_samples * 1000.0 / self.sample_rate

    def ms_to_samples(self, duration_ms: float) -> int:
        """毫秒 → 样本数。"""
        return int(round(self.sample_rate * duration_ms / 1000.0))

    def pts_to_sample_index(self, pts_ms: int) -> int:
        """pts → 相对时间轴起点的样本序号。"""
        return int(round((pts_ms - self._base_pts_ms) * self.sample_rate / 1000.0))

    def snapshot(self) -> dict[str, float]:
        """导出统计快照。"""
        data = self.stats.as_dict()
        data["sample_rate"] = float(self.sample_rate)
        data["channels"] = float(self.channels)
        data["next_pts_ms"] = float(self.next_pts_ms)
        data["elapsed_ms"] = round(self.elapsed_ms, 2)
        return data

    def __repr__(self) -> str:
        """调试用文本表示。"""
        return (
            f"AudioTimeline(pts={self.next_pts_ms}ms, {self.sample_rate}Hz/"
            f"{self.channels}ch, gaps={self.stats.gaps}, drift={self.drift_ms:+.1f}ms)"
        )

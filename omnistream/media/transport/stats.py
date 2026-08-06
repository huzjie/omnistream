"""传输统计采集：RTT、丢包、抖动、码率的滚动窗口与时间序列。

两个消费方
----------
* **Prometheus** —— 聚合视图，看的是整个集群的分布。
* **控制台 / ``/v1/media/stats``** —— 单会话视图，需要**时间序列**才能画出
  "这次卡顿发生在第 37 秒"。Prometheus 的 scrape 间隔（15s）太粗，
  所以这里自己保留 :data:`STATS_HISTORY_POINTS` 个 1 秒粒度的采样点。

抖动估计沿用 RFC 3550 的一阶滤波（与
:class:`~omnistream.media.audio.jitter_buffer.AdaptiveJitterBuffer` 同一套公式），
保证两处口径一致。
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field

from omnistream.core.clock import monotonic_s, now_ms
from omnistream.core.logging import get_logger
from omnistream.media.constants import (
    JITTER_FILTER_GAIN,
    STATS_HISTORY_POINTS,
    STATS_INTERVAL_MS,
)
from omnistream.media.metrics import (
    observe_ms,
    rtt_ms as rtt_metric,
    set_gauge,
    track_loss_ratio,
    transport_bytes,
)

__all__ = ["StatsSample", "TransportStats", "TransportStatsCollector"]

_log = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class StatsSample:
    """一个 1 秒粒度的统计采样点。"""

    ts_ms: int
    rtt_ms: float
    loss_ratio: float
    jitter_ms: float
    send_bps: int
    recv_bps: int
    audio_frames: int
    video_frames: int

    def as_dict(self) -> dict[str, float]:
        """转为可序列化字典。"""
        return {
            "ts_ms": float(self.ts_ms),
            "rtt_ms": round(self.rtt_ms, 2),
            "loss_ratio": round(self.loss_ratio, 5),
            "jitter_ms": round(self.jitter_ms, 2),
            "send_bps": float(self.send_bps),
            "recv_bps": float(self.recv_bps),
            "audio_frames": float(self.audio_frames),
            "video_frames": float(self.video_frames),
        }


@dataclass(slots=True)
class TransportStats:
    """单条传输链路的滚动统计。"""

    kind: str = "unknown"
    session_id: str = ""

    bytes_sent: int = 0
    bytes_recv: int = 0
    audio_bytes_sent: int = 0
    audio_bytes_recv: int = 0
    video_bytes_sent: int = 0
    video_bytes_recv: int = 0
    event_bytes_sent: int = 0
    event_bytes_recv: int = 0

    audio_frames_sent: int = 0
    audio_frames_recv: int = 0
    video_frames_sent: int = 0
    video_frames_recv: int = 0
    events_sent: int = 0
    events_recv: int = 0

    packets_expected: int = 0
    packets_lost: int = 0
    errors: int = 0
    reconnects: int = 0

    rtt_ms: float = 0.0
    jitter_ms: float = 0.0
    send_bps: int = 0
    recv_bps: int = 0

    started_ms: int = field(default_factory=now_ms)
    last_send_ms: int = 0
    last_recv_ms: int = 0

    _rtt_samples: deque[float] = field(
        default_factory=lambda: deque(maxlen=64), repr=False
    )
    _prev_transit_ms: float | None = field(default=None, repr=False)
    history: deque[StatsSample] = field(
        default_factory=lambda: deque(maxlen=STATS_HISTORY_POINTS), repr=False
    )

    # ------------------------------------------------------------------
    # 派生量
    # ------------------------------------------------------------------

    @property
    def loss_ratio(self) -> float:
        """累计丢包率 ``0 ~ 1``。"""
        if self.packets_expected <= 0:
            return 0.0
        return min(1.0, max(0.0, self.packets_lost / float(self.packets_expected)))

    @property
    def uptime_ms(self) -> int:
        """链路存活时长。"""
        return max(0, now_ms() - self.started_ms)

    @property
    def idle_ms(self) -> int:
        """距离最近一次收包的静默时长；从未收包时返回存活时长。"""
        if self.last_recv_ms <= 0:
            return self.uptime_ms
        return max(0, now_ms() - self.last_recv_ms)

    @property
    def avg_rtt_ms(self) -> float:
        """最近若干次 RTT 的平均值。"""
        if not self._rtt_samples:
            return self.rtt_ms
        return sum(self._rtt_samples) / len(self._rtt_samples)

    # ------------------------------------------------------------------
    # 记账
    # ------------------------------------------------------------------

    def on_sent(self, nbytes: int, media: str = "audio") -> None:
        """记录一次发送。"""
        self.bytes_sent += nbytes
        self.last_send_ms = now_ms()
        if media == "audio":
            self.audio_bytes_sent += nbytes
            self.audio_frames_sent += 1
        elif media == "video":
            self.video_bytes_sent += nbytes
            self.video_frames_sent += 1
        else:
            self.event_bytes_sent += nbytes
            self.events_sent += 1
        try:
            transport_bytes.labels(kind=self.kind, direction="out", media=media).inc(nbytes)
        except Exception:  # 指标绝不能影响主链路
            pass

    def on_recv(self, nbytes: int, media: str = "audio") -> None:
        """记录一次接收。"""
        self.bytes_recv += nbytes
        self.last_recv_ms = now_ms()
        if media == "audio":
            self.audio_bytes_recv += nbytes
            self.audio_frames_recv += 1
        elif media == "video":
            self.video_bytes_recv += nbytes
            self.video_frames_recv += 1
        else:
            self.event_bytes_recv += nbytes
            self.events_recv += 1
        try:
            transport_bytes.labels(kind=self.kind, direction="in", media=media).inc(nbytes)
        except Exception:  # 指标绝不能影响主链路
            pass

    def on_rtt(self, value_ms: float) -> None:
        """记录一次 RTT 观测。"""
        self.rtt_ms = float(value_ms)
        self._rtt_samples.append(self.rtt_ms)
        observe_ms(rtt_metric, self.rtt_ms, kind=self.kind)

    def on_transit(self, transit_ms: float) -> None:
        """记录一次单程时延观测，用 RFC 3550 公式更新抖动。"""
        if self._prev_transit_ms is not None:
            delta = abs(transit_ms - self._prev_transit_ms)
            self.jitter_ms += (delta - self.jitter_ms) / float(JITTER_FILTER_GAIN)
        self._prev_transit_ms = transit_ms

    def on_loss(self, lost: int, expected: int) -> None:
        """记录丢包观测。"""
        self.packets_lost += max(0, int(lost))
        self.packets_expected += max(0, int(expected))
        set_gauge(track_loss_ratio, self.loss_ratio, kind=self.kind, media="audio")

    def on_error(self) -> None:
        """记录一次传输错误。"""
        self.errors += 1

    def on_reconnect(self) -> None:
        """记录一次重连。"""
        self.reconnects += 1

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------

    def as_dict(self) -> dict[str, float | str]:
        """导出当前快照（不含历史序列）。"""
        return {
            "kind": self.kind,
            "session_id": self.session_id,
            "uptime_ms": float(self.uptime_ms),
            "idle_ms": float(self.idle_ms),
            "bytes_sent": float(self.bytes_sent),
            "bytes_recv": float(self.bytes_recv),
            "audio_frames_sent": float(self.audio_frames_sent),
            "audio_frames_recv": float(self.audio_frames_recv),
            "video_frames_sent": float(self.video_frames_sent),
            "video_frames_recv": float(self.video_frames_recv),
            "events_sent": float(self.events_sent),
            "events_recv": float(self.events_recv),
            "rtt_ms": round(self.rtt_ms, 2),
            "avg_rtt_ms": round(self.avg_rtt_ms, 2),
            "jitter_ms": round(self.jitter_ms, 2),
            "loss_ratio": round(self.loss_ratio, 5),
            "send_bps": float(self.send_bps),
            "recv_bps": float(self.recv_bps),
            "errors": float(self.errors),
            "reconnects": float(self.reconnects),
        }

    def series(self, limit: int = 0) -> list[dict[str, float]]:
        """导出时间序列，``limit=0`` 表示全部。"""
        items = list(self.history)
        if limit > 0:
            items = items[-limit:]
        return [sample.as_dict() for sample in items]

    def reset(self) -> None:
        """清零全部计数（重连后重新开始统计）。"""
        self.bytes_sent = self.bytes_recv = 0
        self.audio_bytes_sent = self.audio_bytes_recv = 0
        self.video_bytes_sent = self.video_bytes_recv = 0
        self.event_bytes_sent = self.event_bytes_recv = 0
        self.audio_frames_sent = self.audio_frames_recv = 0
        self.video_frames_sent = self.video_frames_recv = 0
        self.events_sent = self.events_recv = 0
        self.packets_expected = self.packets_lost = 0
        self.errors = 0
        self.rtt_ms = self.jitter_ms = 0.0
        self.send_bps = self.recv_bps = 0
        self.started_ms = now_ms()
        self._rtt_samples.clear()
        self._prev_transit_ms = None
        self.history.clear()


class TransportStatsCollector:
    """周期采样器：把 :class:`TransportStats` 的累计量差分成瞬时码率。"""

    def __init__(
        self,
        stats: TransportStats,
        *,
        interval_ms: int = STATS_INTERVAL_MS,
        name: str = "transport_stats",
    ) -> None:
        """初始化采样器。

        Args:
            stats: 被采样的统计对象。
            interval_ms: 采样周期。
            name: 后台任务名，便于排查泄漏。
        """
        self.stats = stats
        self.interval_ms = max(100, int(interval_ms))
        self.name = name
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self._last_sent = 0
        self._last_recv = 0
        self._last_audio = 0
        self._last_video = 0
        self._last_tick_s = monotonic_s()

    @property
    def running(self) -> bool:
        """采样任务是否在跑。"""
        return self._task is not None and not self._task.done()

    def tick(self) -> StatsSample:
        """立即执行一次采样（同步，供测试与手动触发使用）。"""
        now_s = monotonic_s()
        elapsed = max(1e-3, now_s - self._last_tick_s)
        self._last_tick_s = now_s

        sent_delta = max(0, self.stats.bytes_sent - self._last_sent)
        recv_delta = max(0, self.stats.bytes_recv - self._last_recv)
        audio_delta = max(0, self.stats.audio_frames_recv - self._last_audio)
        video_delta = max(0, self.stats.video_frames_recv - self._last_video)

        self._last_sent = self.stats.bytes_sent
        self._last_recv = self.stats.bytes_recv
        self._last_audio = self.stats.audio_frames_recv
        self._last_video = self.stats.video_frames_recv

        self.stats.send_bps = int(sent_delta * 8 / elapsed)
        self.stats.recv_bps = int(recv_delta * 8 / elapsed)

        sample = StatsSample(
            ts_ms=now_ms(),
            rtt_ms=self.stats.rtt_ms,
            loss_ratio=self.stats.loss_ratio,
            jitter_ms=self.stats.jitter_ms,
            send_bps=self.stats.send_bps,
            recv_bps=self.stats.recv_bps,
            audio_frames=audio_delta,
            video_frames=video_delta,
        )
        self.stats.history.append(sample)
        return sample

    async def _run(self) -> None:
        """后台采样循环。"""
        period = self.interval_ms / 1000.0
        while not self._stopping.is_set():
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=period)
                return
            except TimeoutError:
                pass
            try:
                self.tick()
            except Exception as exc:  # 采样失败不能拖垮链路
                _log.warning(
                    "transport_stats_tick_failed",
                    collector=self.name,
                    session_id=self.stats.session_id,
                    error=str(exc),
                )

    def start(self) -> None:
        """启动后台采样（幂等）。"""
        if self.running:
            return
        self._stopping.clear()
        self._last_tick_s = monotonic_s()
        self._task = asyncio.create_task(self._run(), name=self.name)

    async def stop(self) -> None:
        """停止后台采样（幂等）。"""
        self._stopping.set()
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # 停止阶段吞掉一切
            pass

    def __repr__(self) -> str:
        """调试用文本表示。"""
        return (
            f"TransportStatsCollector({self.name}, running={self.running}, "
            f"points={len(self.stats.history)})"
        )

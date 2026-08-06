"""时间与时钟原语。

约定（见架构文档 §7.6）:

===================  =====================  ==========================
场景                 单位                   函数
===================  =====================  ==========================
数据库 / API 字段    tz-aware ``datetime``  :func:`utcnow`
协议 ``Envelope.ts`` UTC 毫秒 ``int``       :func:`now_ms`
媒体 ``pts_ms``      会话内相对毫秒 ``int`` :meth:`MediaClock.pts`
延迟测量             单调纳秒 ``int``       :func:`now_ns`
===================  =====================  ==========================
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

__all__ = [
    "MediaClock",
    "from_iso",
    "monotonic_s",
    "now_ms",
    "now_ns",
    "now_us",
    "to_iso",
    "utcnow",
    "utcnow_iso",
]


def utcnow() -> datetime:
    """当前 UTC 时间（tz-aware）。禁止使用 ``datetime.utcnow()``。"""
    return datetime.now(UTC)


def utcnow_iso() -> str:
    """ISO-8601 UTC 字符串，形如 ``2026-08-06T09:12:33.456Z``。"""
    return to_iso(utcnow())


def to_iso(dt: datetime) -> str:
    """把任意 datetime 规范化成毫秒精度的 ISO-8601 Z 结尾字符串。"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    dt = dt.astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def from_iso(value: str) -> datetime:
    """解析 ISO-8601 字符串（容忍结尾 ``Z``），返回 tz-aware UTC datetime。"""
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def now_ms() -> int:
    """UTC 墙钟毫秒。用于协议时间戳，跨进程可比。"""
    return time.time_ns() // 1_000_000


def now_us() -> int:
    """UTC 墙钟微秒。"""
    return time.time_ns() // 1_000


def now_ns() -> int:
    """单调纳秒。用于延迟测量，不受 NTP 校时影响，跨进程不可比。"""
    return time.monotonic_ns()


def monotonic_s() -> float:
    """单调秒（float）。"""
    return time.monotonic()


@dataclass(slots=True)
class MediaClock:
    """会话级媒体时钟。

    职责：
      1. 把单调纳秒换算成"从会话开始为 0"的 ``pts_ms``；
      2. 在 RTP timestamp（按采样率计数、32bit 回绕）与 ``pts_ms`` 之间双向映射；
      3. 估算发送端与本地时钟的漂移，供 ``CLOCK_DRIFT`` 告警使用。

    典型用法::

        clock = MediaClock(sample_rate=16000)
        clock.start()
        pts = clock.pts()                     # 相对会话起点的毫秒
        pts = clock.pts_from_rtp(rtp_ts)      # 由 RTP 时间戳换算
    """

    sample_rate: int = 16_000
    #: 会话起点的单调纳秒；``start()`` 时写入。
    origin_ns: int = field(default_factory=now_ns)
    #: 会话起点的墙钟毫秒，用于把 pts 还原为绝对时间。
    origin_wall_ms: int = field(default_factory=now_ms)
    #: 首个 RTP 时间戳，作为 RTP 映射的基准。
    rtp_base: int | None = None
    _last_pts_ms: int = 0
    _drift_ms: float = 0.0

    RTP_MODULO: int = 1 << 32

    def start(self) -> None:
        """（重新）标定会话起点。幂等调用会重置时间基。"""
        self.origin_ns = now_ns()
        self.origin_wall_ms = now_ms()
        self.rtp_base = None
        self._last_pts_ms = 0
        self._drift_ms = 0.0

    # ------------------------------------------------------------------ pts
    def pts(self) -> int:
        """当前时刻相对会话起点的毫秒数（单调，永不回退）。"""
        value = (now_ns() - self.origin_ns) // 1_000_000
        if value < self._last_pts_ms:
            value = self._last_pts_ms
        self._last_pts_ms = value
        return int(value)

    def pts_at(self, capture_ts_ns: int) -> int:
        """把某个采集时刻的单调纳秒换算成 pts_ms。"""
        return max(0, (capture_ts_ns - self.origin_ns) // 1_000_000)

    def advance(self, duration_ms: int) -> int:
        """按固定帧长推进逻辑时钟并返回推进前的 pts。

        用于"生成侧"（TTS 输出）这类没有真实采集时刻的场景：连续调用
        ``advance(20)`` 即可得到 0, 20, 40 ... 的严格等间隔 pts 序列。
        """
        current = self._last_pts_ms
        self._last_pts_ms = current + int(duration_ms)
        return current

    def reset_pts(self, pts_ms: int = 0) -> None:
        """把逻辑 pts 游标复位（barge-in 后重开播放队列时用）。"""
        self._last_pts_ms = max(0, int(pts_ms))

    # ------------------------------------------------------------------ 墙钟
    def wall_ms(self, pts_ms: int) -> int:
        """把 pts_ms 还原为 UTC 墙钟毫秒。"""
        return self.origin_wall_ms + int(pts_ms)

    def wall_datetime(self, pts_ms: int) -> datetime:
        """把 pts_ms 还原为 tz-aware datetime。"""
        base = datetime.fromtimestamp(self.origin_wall_ms / 1000.0, tz=UTC)
        return base + timedelta(milliseconds=int(pts_ms))

    # ------------------------------------------------------------------ RTP
    def pts_from_rtp(self, rtp_timestamp: int, *, sample_rate: int | None = None) -> int:
        """RTP timestamp -> pts_ms。首次调用会把该值锚定为 0 点。"""
        rate = sample_rate or self.sample_rate
        if rate <= 0:
            raise ValueError(f"非法采样率: {rate}")
        ts = int(rtp_timestamp) % self.RTP_MODULO
        if self.rtp_base is None:
            self.rtp_base = ts
            return 0
        delta = (ts - self.rtp_base) % self.RTP_MODULO
        # 处理 32bit 回绕造成的"看似巨大"的正向跳变：超过半个周期视为回退。
        if delta > self.RTP_MODULO // 2:
            delta -= self.RTP_MODULO
        return int(delta * 1000 // rate)

    def rtp_from_pts(self, pts_ms: int, *, sample_rate: int | None = None) -> int:
        """pts_ms -> RTP timestamp。"""
        rate = sample_rate or self.sample_rate
        base = self.rtp_base or 0
        return int((base + int(pts_ms) * rate // 1000) % self.RTP_MODULO)

    # ------------------------------------------------------------------ 漂移
    def observe_drift(self, remote_pts_ms: int) -> float:
        """记录一次远端 pts 与本地 pts 的偏差（毫秒），返回指数平滑后的漂移。"""
        instant = float(self.pts() - int(remote_pts_ms))
        self._drift_ms = 0.9 * self._drift_ms + 0.1 * instant
        return self._drift_ms

    @property
    def drift_ms(self) -> float:
        """当前估计的时钟漂移（毫秒，正数表示本地领先）。"""
        return self._drift_ms

    @property
    def elapsed_ms(self) -> int:
        """会话已持续的毫秒数。"""
        return int((now_ns() - self.origin_ns) // 1_000_000)

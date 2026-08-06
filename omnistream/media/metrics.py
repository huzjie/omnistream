"""媒体层 Prometheus 指标定义与埋点辅助。

设计要点
--------
* ``prometheus_client`` 未安装时全部退化为 no-op，模块仍可 import、业务代码零分支。
* 重复注册（reload / 多次 import）时复用已注册的 collector，而不是抛
  ``Duplicate timeseries`` 把服务打挂。
* 指标名统一 ``oms_media_*``，直方图单位后缀 ``_ms``（与 §7.6 一致）。
"""
from __future__ import annotations

from typing import Any, Sequence

__all__ = [
    "PROMETHEUS_AVAILABLE",
    "audio_frames_in",
    "audio_frames_out",
    "bandwidth_estimate_bps",
    "bitrate_audio_bps",
    "bitrate_video_bps",
    "codec_errors",
    "frames_dropped",
    "jitter_buffer_depth_ms",
    "jitter_buffer_plc",
    "jitter_buffer_target_ms",
    "media_stalls",
    "observe_ms",
    "pipeline_latency_ms",
    "playback_cancelled_ms",
    "playback_queue_depth_ms",
    "resample_ms",
    "rtt_ms",
    "set_gauge",
    "track_loss_ratio",
    "transport_bytes",
    "transport_connections",
    "transport_errors",
    "video_frames_accepted",
    "video_frames_in",
    "video_sampler_fps",
]

try:  # pragma: no cover - 环境相关
    from prometheus_client import REGISTRY, Counter, Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - 环境相关
    REGISTRY = None  # type: ignore[assignment]
    Counter = Gauge = Histogram = None  # type: ignore[assignment]
    PROMETHEUS_AVAILABLE = False


class _NoopMetric:
    """无 prometheus_client 时的空实现，接口与真实指标一致。"""

    __slots__ = ("_name",)

    def __init__(self, name: str = "noop") -> None:
        self._name = name

    def labels(self, *args: Any, **kwargs: Any) -> "_NoopMetric":
        return self

    def inc(self, amount: float = 1.0) -> None:
        return None

    def dec(self, amount: float = 1.0) -> None:
        return None

    def set(self, value: float) -> None:
        return None

    def observe(self, value: float) -> None:
        return None

    def __repr__(self) -> str:
        return f"<NoopMetric {self._name}>"


def _existing(name: str) -> Any | None:
    """从默认 registry 中找回已注册的同名 collector。"""
    if not PROMETHEUS_AVAILABLE or REGISTRY is None:
        return None
    mapping = getattr(REGISTRY, "_names_to_collectors", {})
    for candidate in (name, f"{name}_total"):
        found = mapping.get(candidate)
        if found is not None:
            return found
    return None


def _build(kind: str, name: str, doc: str, labels: Sequence[str] = (), **kw: Any) -> Any:
    if not PROMETHEUS_AVAILABLE:
        return _NoopMetric(name)
    factory = {"counter": Counter, "gauge": Gauge, "histogram": Histogram}[kind]
    try:
        return factory(name, doc, list(labels), **kw)  # type: ignore[misc]
    except ValueError:
        found = _existing(name)
        return found if found is not None else _NoopMetric(name)


_LATENCY_BUCKETS = (1, 2, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560)
_DEPTH_BUCKETS = (10, 20, 40, 80, 120, 200, 320, 500, 800, 1500, 3000)

# --------------------------------------------------------------------------
# 传输
# --------------------------------------------------------------------------

transport_connections = _build(
    "gauge", "oms_media_transport_connections", "当前活跃媒体传输连接数", ("kind",)
)
transport_bytes = _build(
    "counter", "oms_media_transport_bytes", "媒体传输字节数", ("kind", "direction", "media")
)
transport_errors = _build(
    "counter", "oms_media_transport_errors", "媒体传输错误计数", ("kind", "reason")
)
rtt_ms = _build(
    "histogram", "oms_media_rtt_ms", "传输往返时延（毫秒）", ("kind",), buckets=_LATENCY_BUCKETS
)
track_loss_ratio = _build("gauge", "oms_media_loss_ratio", "丢包率 0~1", ("kind", "media"))
media_stalls = _build("counter", "oms_media_stalls", "媒体链路静默/停滞次数", ("media", "reason"))

# --------------------------------------------------------------------------
# 音频
# --------------------------------------------------------------------------

audio_frames_in = _build("counter", "oms_media_audio_frames_in", "入口音频帧数", ("session_kind",))
audio_frames_out = _build("counter", "oms_media_audio_frames_out", "出口音频帧数", ("session_kind",))
frames_dropped = _build("counter", "oms_media_frames_dropped", "被丢弃的帧数", ("media", "stage", "reason"))
jitter_buffer_depth_ms = _build(
    "histogram", "oms_media_jitter_depth_ms", "抖动缓冲实际深度（毫秒）", (), buckets=_DEPTH_BUCKETS
)
jitter_buffer_target_ms = _build("gauge", "oms_media_jitter_target_ms", "抖动缓冲目标时延（毫秒）", ())
jitter_buffer_plc = _build("counter", "oms_media_jitter_plc_frames", "丢包补偿生成的帧数", ("mode",))
playback_queue_depth_ms = _build(
    "histogram", "oms_media_playback_depth_ms", "播放队列堆积（毫秒）", (), buckets=_DEPTH_BUCKETS
)
playback_cancelled_ms = _build(
    "counter", "oms_media_playback_cancelled_ms", "barge-in 撤销的播放毫秒数", ()
)
resample_ms = _build(
    "histogram", "oms_media_resample_ms", "单次重采样耗时（毫秒）", ("backend",), buckets=_LATENCY_BUCKETS
)
codec_errors = _build("counter", "oms_media_codec_errors", "编解码错误", ("codec", "direction"))

# --------------------------------------------------------------------------
# 视频
# --------------------------------------------------------------------------

video_frames_in = _build("counter", "oms_media_video_frames_in", "入口视频帧数", ())
video_frames_accepted = _build(
    "counter", "oms_media_video_frames_accepted", "自适应采样接受的视频帧数", ("reason",)
)
video_sampler_fps = _build("gauge", "oms_media_video_sampler_fps", "当前视频目标采样率", ())

# --------------------------------------------------------------------------
# 自适应
# --------------------------------------------------------------------------

bandwidth_estimate_bps = _build("gauge", "oms_media_bwe_bps", "带宽估计（bit/s）", ())
bitrate_audio_bps = _build("gauge", "oms_media_audio_bitrate_bps", "音频分配码率（bit/s）", ())
bitrate_video_bps = _build("gauge", "oms_media_video_bitrate_bps", "视频分配码率（bit/s）", ())

# --------------------------------------------------------------------------
# 管线
# --------------------------------------------------------------------------

pipeline_latency_ms = _build(
    "histogram", "oms_media_pipeline_latency_ms", "管线单帧处理耗时（毫秒）",
    ("pipeline",), buckets=_LATENCY_BUCKETS,
)


def observe_ms(metric: Any, value_ms: float, **labels: str) -> None:
    """安全地记录一个毫秒观测值。

    Args:
        metric: histogram 指标对象。
        value_ms: 观测值（毫秒）。
        **labels: 标签键值。
    """
    try:
        target = metric.labels(**labels) if labels else metric
        target.observe(float(value_ms))
    except Exception:  # 指标绝不能影响主链路
        return None


def set_gauge(metric: Any, value: float, **labels: str) -> None:
    """安全地设置一个 gauge 值。"""
    try:
        target = metric.labels(**labels) if labels else metric
        target.set(float(value))
    except Exception:
        return None

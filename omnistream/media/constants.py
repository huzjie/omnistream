"""媒体层常量与配置安全读取。

本模块不依赖任何可选第三方库，任何情况下都必须可以 import。

所有可调参数都遵循以下优先级：

1. 调用方显式传参
2. ``omnistream.config.get_settings().media.<name>``（WS1 提供）
3. 本模块的默认常量

第 2 步通过 :func:`media_setting` 做了防御性读取 —— 当配置体系尚未就绪或字段名不一致时
自动退回默认值，保证媒体层可以脱离完整应用独立运行（回环 Demo / 压测脚本 / 单文件调试）。
"""
from __future__ import annotations

from typing import Any, TypeVar

T = TypeVar("T")

# --------------------------------------------------------------------------
# 时基与分帧
# --------------------------------------------------------------------------

#: 全链路标准帧长（毫秒）。VAD / ASR / 打断撤销的最小粒度。
DEFAULT_FRAME_MS: int = 20

#: 允许的帧长集合（毫秒）。Opus 只接受这些值。
SUPPORTED_FRAME_MS: tuple[int, ...] = (2, 5, 10, 20, 40, 60)

#: 上行（用户 → 平台）默认采样率。
DEFAULT_INPUT_SAMPLE_RATE: int = 16000

#: 下行（平台 → 用户）默认采样率，与主流 TTS 输出对齐。
DEFAULT_OUTPUT_SAMPLE_RATE: int = 24000

#: WebRTC 内部工作采样率（Opus 原生）。
WEBRTC_SAMPLE_RATE: int = 48000

#: 受支持的采样率，与 ``schemas.audio.AudioFormat`` 的校验保持一致。
SUPPORTED_SAMPLE_RATES: tuple[int, ...] = (8000, 16000, 22050, 24000, 44100, 48000)

#: 默认声道数。
DEFAULT_CHANNELS: int = 1

# --------------------------------------------------------------------------
# 抖动缓冲
# --------------------------------------------------------------------------

JITTER_MIN_DELAY_MS: int = 40
JITTER_MAX_DELAY_MS: int = 400
JITTER_INITIAL_TARGET_MS: int = 80
#: RFC 3550 抖动估计的一阶滤波系数分母（J += (|D| - J) / 16）。
JITTER_FILTER_GAIN: int = 16
#: 目标时延 = base + JITTER_SAFETY_FACTOR * jitter。
JITTER_SAFETY_FACTOR: float = 3.0
#: 连续多少帧无需扩张后才允许收缩目标时延（滞回，避免抖动）。
JITTER_SHRINK_PATIENCE: int = 150
#: 最多允许连续生成多少帧丢包补偿（PLC），超过后输出静音。
JITTER_MAX_PLC_FRAMES: int = 5

# --------------------------------------------------------------------------
# 缓冲上限（全部以"条目数"计）
# --------------------------------------------------------------------------

AUDIO_STREAM_MAXSIZE: int = 256          # 20ms/帧 → 约 5.1s
VIDEO_STREAM_MAXSIZE: int = 24           # DROP_OLDEST
EVENT_STREAM_MAXSIZE: int = 256
PLAYBACK_QUEUE_MAX_MS: int = 12_000      # 播放队列最大堆积毫秒
VIDEO_FRAME_BUFFER_MAXLEN: int = 24

# --------------------------------------------------------------------------
# 音频处理
# --------------------------------------------------------------------------

AGC_TARGET_DBFS: float = -18.0
AGC_MAX_GAIN_DB: float = 24.0
AGC_MIN_GAIN_DB: float = -12.0
AGC_ATTACK_MS: float = 12.0
AGC_RELEASE_MS: float = 220.0
LIMITER_CEILING_DBFS: float = -1.0

NOISE_GATE_OPEN_DBFS: float = -45.0
NOISE_GATE_CLOSE_DBFS: float = -52.0
NOISE_GATE_HOLD_MS: float = 180.0
NOISE_GATE_ATTACK_MS: float = 5.0
NOISE_GATE_RELEASE_MS: float = 60.0

SILENCE_THRESHOLD_DBFS: float = -50.0

#: 时间轴漂移超过该阈值即告警 / 抛 ClockDriftError。
CLOCK_DRIFT_WARN_MS: float = 250.0
CLOCK_DRIFT_FATAL_MS: float = 2000.0

# --------------------------------------------------------------------------
# 码率区间（bit/s）
# --------------------------------------------------------------------------

AUDIO_BITRATE_FLOOR_BPS: int = 24_000     # 硬保底，任何拥塞下都不让路
AUDIO_BITRATE_DEFAULT_BPS: int = 32_000
AUDIO_BITRATE_CEILING_BPS: int = 64_000

VIDEO_BITRATE_FLOOR_BPS: int = 60_000
VIDEO_BITRATE_DEFAULT_BPS: int = 600_000
VIDEO_BITRATE_CEILING_BPS: int = 2_500_000

TOTAL_BITRATE_START_BPS: int = 700_000
TOTAL_BITRATE_MIN_BPS: int = 40_000
TOTAL_BITRATE_MAX_BPS: int = 4_000_000

#: 传输层协议与容器开销的经验系数。
NETWORK_OVERHEAD_FACTOR: float = 1.08

# --------------------------------------------------------------------------
# 视频
# --------------------------------------------------------------------------

#: 送 VLM 的默认长边分辨率。
VLM_LONG_SIDE: int = 768
#: 屏幕共享保留更高分辨率（文字可读性）。
SCREENSHARE_LONG_SIDE: int = 1280

JPEG_QUALITY_DEFAULT: int = 72
JPEG_QUALITY_THUMBNAIL: int = 60
WEBP_QUALITY_DEFAULT: int = 75

VIDEO_FPS_MIN: float = 0.2
VIDEO_FPS_MAX: float = 5.0
VIDEO_FPS_DEFAULT: float = 0.5
SCREENSHARE_FPS_DEFAULT: float = 0.25

#: 场景变化分数高于该值时触发瞬时提采样。
SCENE_CHANGE_BOOST_THRESHOLD: float = 0.55
#: 场景变化分数低于该值视为静止画面。
SCENE_CHANGE_IDLE_THRESHOLD: float = 0.08
#: 提采样持续时长。
SCENE_BOOST_WINDOW_MS: int = 3_000

#: 分辨率档位（长边，宽 x 高 以 16:9 为参考基准）。
RESOLUTION_LADDER: tuple[tuple[int, int], ...] = (
    (320, 180),
    (480, 270),
    (640, 360),
    (854, 480),
    (1280, 720),
    (1920, 1080),
)

THUMBNAIL_LONG_SIDE: int = 240
SPRITE_COLUMNS: int = 5

# --------------------------------------------------------------------------
# 传输
# --------------------------------------------------------------------------

WS_HEARTBEAT_INTERVAL_MS: int = 15_000
WS_IDLE_TIMEOUT_MS: int = 45_000
WS_SEND_TIMEOUT_MS: int = 5_000

#: DataChannel 单条消息安全上限（SCTP 实现普遍在 64KB，取 16KB 更稳）。
DATACHANNEL_MTU: int = 16 * 1024
DATACHANNEL_REASSEMBLY_TIMEOUT_MS: int = 10_000
DATACHANNEL_MAX_PENDING_MESSAGES: int = 64

ICE_GATHER_TIMEOUT_MS: int = 5_000
ICE_CREDENTIAL_TTL_S: int = 24 * 3600
DEFAULT_STUN_URLS: tuple[str, ...] = ("stun:stun.l.google.com:19302",)

#: 媒体链路静默多久判定为 stalled。
MEDIA_STALL_TIMEOUT_MS: int = 8_000
HEALTH_CHECK_INTERVAL_MS: int = 2_000

#: 统计采样周期。
STATS_INTERVAL_MS: int = 1_000
#: 统计时间序列保留点数（1s 一个点 → 10 分钟）。
STATS_HISTORY_POINTS: int = 600

# --------------------------------------------------------------------------
# 配置安全读取
# --------------------------------------------------------------------------

_SETTINGS_CACHE: dict[str, Any] = {}


def _media_settings_obj() -> Any:
    """获取 ``settings.media``，任何异常都退化为 ``None``。"""
    if "obj" in _SETTINGS_CACHE:
        return _SETTINGS_CACHE["obj"]
    obj: Any = None
    try:
        from omnistream.config import get_settings  # 延迟导入，避免循环依赖

        obj = getattr(get_settings(), "media", None)
    except Exception:  # 配置体系不可用时媒体层仍需可用
        obj = None
    _SETTINGS_CACHE["obj"] = obj
    return obj


def media_setting(name: str, default: T) -> T:
    """读取 ``settings.media.<name>``，缺失时返回 ``default``。

    Args:
        name: 配置字段名。
        default: 缺省值，同时决定返回类型。

    Returns:
        配置值或缺省值。
    """
    obj = _media_settings_obj()
    if obj is None:
        return default
    value = getattr(obj, name, None)
    if value is None:
        return default
    if isinstance(default, bool):
        return bool(value)  # type: ignore[return-value]
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(value)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return default
    if isinstance(default, float):
        try:
            return float(value)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return default
    return value  # type: ignore[return-value]


def reset_settings_cache() -> None:
    """清空配置缓存（配置热更新或测试期使用）。"""
    _SETTINGS_CACHE.clear()


def input_sample_rate() -> int:
    """上行目标采样率。"""
    return int(media_setting("input_sample_rate", DEFAULT_INPUT_SAMPLE_RATE))


def output_sample_rate() -> int:
    """下行目标采样率。"""
    return int(media_setting("output_sample_rate", DEFAULT_OUTPUT_SAMPLE_RATE))


def frame_ms() -> int:
    """标准帧长。"""
    value = int(media_setting("frame_ms", DEFAULT_FRAME_MS))
    return value if value in SUPPORTED_FRAME_MS else DEFAULT_FRAME_MS


__all__ = [
    "AGC_ATTACK_MS",
    "AGC_MAX_GAIN_DB",
    "AGC_MIN_GAIN_DB",
    "AGC_RELEASE_MS",
    "AGC_TARGET_DBFS",
    "AUDIO_BITRATE_CEILING_BPS",
    "AUDIO_BITRATE_DEFAULT_BPS",
    "AUDIO_BITRATE_FLOOR_BPS",
    "AUDIO_STREAM_MAXSIZE",
    "CLOCK_DRIFT_FATAL_MS",
    "CLOCK_DRIFT_WARN_MS",
    "DATACHANNEL_MAX_PENDING_MESSAGES",
    "DATACHANNEL_MTU",
    "DATACHANNEL_REASSEMBLY_TIMEOUT_MS",
    "DEFAULT_CHANNELS",
    "DEFAULT_FRAME_MS",
    "DEFAULT_INPUT_SAMPLE_RATE",
    "DEFAULT_OUTPUT_SAMPLE_RATE",
    "DEFAULT_STUN_URLS",
    "EVENT_STREAM_MAXSIZE",
    "HEALTH_CHECK_INTERVAL_MS",
    "ICE_CREDENTIAL_TTL_S",
    "ICE_GATHER_TIMEOUT_MS",
    "JITTER_FILTER_GAIN",
    "JITTER_INITIAL_TARGET_MS",
    "JITTER_MAX_DELAY_MS",
    "JITTER_MAX_PLC_FRAMES",
    "JITTER_MIN_DELAY_MS",
    "JITTER_SAFETY_FACTOR",
    "JITTER_SHRINK_PATIENCE",
    "JPEG_QUALITY_DEFAULT",
    "JPEG_QUALITY_THUMBNAIL",
    "LIMITER_CEILING_DBFS",
    "MEDIA_STALL_TIMEOUT_MS",
    "NETWORK_OVERHEAD_FACTOR",
    "NOISE_GATE_ATTACK_MS",
    "NOISE_GATE_CLOSE_DBFS",
    "NOISE_GATE_HOLD_MS",
    "NOISE_GATE_OPEN_DBFS",
    "NOISE_GATE_RELEASE_MS",
    "PLAYBACK_QUEUE_MAX_MS",
    "RESOLUTION_LADDER",
    "SCENE_BOOST_WINDOW_MS",
    "SCENE_CHANGE_BOOST_THRESHOLD",
    "SCENE_CHANGE_IDLE_THRESHOLD",
    "SCREENSHARE_FPS_DEFAULT",
    "SCREENSHARE_LONG_SIDE",
    "SILENCE_THRESHOLD_DBFS",
    "SPRITE_COLUMNS",
    "STATS_HISTORY_POINTS",
    "STATS_INTERVAL_MS",
    "SUPPORTED_FRAME_MS",
    "SUPPORTED_SAMPLE_RATES",
    "THUMBNAIL_LONG_SIDE",
    "TOTAL_BITRATE_MAX_BPS",
    "TOTAL_BITRATE_MIN_BPS",
    "TOTAL_BITRATE_START_BPS",
    "VIDEO_BITRATE_CEILING_BPS",
    "VIDEO_BITRATE_DEFAULT_BPS",
    "VIDEO_BITRATE_FLOOR_BPS",
    "VIDEO_FPS_DEFAULT",
    "VIDEO_FPS_MAX",
    "VIDEO_FPS_MIN",
    "VIDEO_FRAME_BUFFER_MAXLEN",
    "VIDEO_STREAM_MAXSIZE",
    "VLM_LONG_SIDE",
    "WEBP_QUALITY_DEFAULT",
    "WEBRTC_SAMPLE_RATE",
    "WS_HEARTBEAT_INTERVAL_MS",
    "WS_IDLE_TIMEOUT_MS",
    "WS_SEND_TIMEOUT_MS",
    "frame_ms",
    "input_sample_rate",
    "media_setting",
    "output_sample_rate",
    "reset_settings_cache",
]

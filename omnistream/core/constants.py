"""全局常量。任何模块想改这些数值，改这里，不要各自硬编码。"""
from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------- 协议
PROTOCOL_VERSION: Final[str] = "oms.v1"
BINARY_FRAME_MAGIC: Final[bytes] = b"OMS1"

# ---------------------------------------------------------------- 音频
#: 上行（用户 -> 平台）默认采样率，与绝大多数 ASR 引擎原生输入一致。
DEFAULT_INPUT_SAMPLE_RATE: Final[int] = 16_000
#: 下行（平台 -> 用户）默认采样率，与主流 TTS 输出一致。
DEFAULT_OUTPUT_SAMPLE_RATE: Final[int] = 24_000
DEFAULT_CHANNELS: Final[int] = 1
#: 全链路统一帧长；barge-in 的撤销粒度即为此值。
DEFAULT_FRAME_MS: Final[int] = 20
BYTES_PER_PCM16_SAMPLE: Final[int] = 2
BYTES_PER_F32_SAMPLE: Final[int] = 4
SUPPORTED_SAMPLE_RATES: Final[tuple[int, ...]] = (8000, 16000, 22050, 24000, 44100, 48000)

#: 20ms @ 16kHz mono pcm16 = 320 samples = 640 bytes
INPUT_FRAME_SAMPLES: Final[int] = DEFAULT_INPUT_SAMPLE_RATE * DEFAULT_FRAME_MS // 1000
INPUT_FRAME_BYTES: Final[int] = INPUT_FRAME_SAMPLES * BYTES_PER_PCM16_SAMPLE
OUTPUT_FRAME_SAMPLES: Final[int] = DEFAULT_OUTPUT_SAMPLE_RATE * DEFAULT_FRAME_MS // 1000
OUTPUT_FRAME_BYTES: Final[int] = OUTPUT_FRAME_SAMPLES * BYTES_PER_PCM16_SAMPLE

#: 音频环形缓冲默认容量（秒）。
RING_BUFFER_SECONDS: Final[float] = 30.0

# ---------------------------------------------------------------- 视频
DEFAULT_VIDEO_FPS: Final[float] = 1.0
MAX_VIDEO_FPS: Final[float] = 5.0
DEFAULT_VIDEO_WIDTH: Final[int] = 768
DEFAULT_VIDEO_HEIGHT: Final[int] = 432
DEFAULT_JPEG_QUALITY: Final[int] = 80

# ---------------------------------------------------------------- 全双工
#: barge-in 从"用户开口"到"停止播放"的硬性 SLO。
BARGE_IN_BUDGET_MS: Final[int] = 200
#: provider.cancel_response() 之后必须停止吐音频的时间上限。
CANCEL_BUDGET_MS: Final[int] = 100
DEFAULT_MIN_SPEECH_MS: Final[int] = 120
DEFAULT_MIN_SILENCE_MS: Final[int] = 480
DEFAULT_PREFIX_PADDING_MS: Final[int] = 300
DEFAULT_MAX_TURN_MS: Final[int] = 120_000
DEFAULT_IDLE_TIMEOUT_MS: Final[int] = 300_000

# ---------------------------------------------------------------- 超时
DEFAULT_CONNECT_TIMEOUT_S: Final[float] = 10.0
DEFAULT_REQUEST_TIMEOUT_S: Final[float] = 30.0
DEFAULT_PROVIDER_TIMEOUT_S: Final[float] = 60.0
DEFAULT_TOOL_TIMEOUT_MS: Final[int] = 15_000
DEFAULT_SHUTDOWN_TIMEOUT_S: Final[float] = 20.0

# ---------------------------------------------------------------- 队列容量
AUDIO_INGRESS_QUEUE: Final[int] = 200      # 4s @ 20ms
AUDIO_EGRESS_QUEUE: Final[int] = 400       # 8s @ 20ms
VIDEO_QUEUE: Final[int] = 8
PROVIDER_EVENT_QUEUE: Final[int] = 512
METRICS_QUEUE: Final[int] = 64

# ---------------------------------------------------------------- 分页
DEFAULT_PAGE_LIMIT: Final[int] = 20
MAX_PAGE_LIMIT: Final[int] = 200

# ---------------------------------------------------------------- 记忆
DEFAULT_EMBEDDING_DIM: Final[int] = 384
DEFAULT_RELEVANCE_THRESHOLD: Final[float] = 0.62
DEFAULT_NOVELTY_THRESHOLD: Final[float] = 0.35
MAX_INJECTIONS_PER_TURN: Final[int] = 1
MAX_INJECTIONS_PER_SESSION: Final[int] = 6

# ---------------------------------------------------------------- HTTP
REQUEST_ID_HEADER: Final[str] = "X-Request-Id"
IDEMPOTENCY_KEY_HEADER: Final[str] = "Idempotency-Key"
API_KEY_HEADER: Final[str] = "X-API-Key"
API_PREFIX: Final[str] = "/v1"
API_KEY_PLAINTEXT_PREFIX: Final[str] = "oms_live_"

__all__ = [name for name in dir() if name.isupper()]

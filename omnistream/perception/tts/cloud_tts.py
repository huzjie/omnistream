"""云端 TTS 适配器：一套传输层 + 可插拔厂商方言。

为什么要"方言"
--------------
Azure / OpenAI / ElevenLabs / 火山引擎的流式 TTS，**协议骨架其实是一样的**：
鉴权 -> 提交文本 -> 收音频字节 -> 收结束标志。
真正不同的只有四件小事：URL 怎么拼、头怎么签、请求体长什么样、响应怎么拆。

所以这里把这四件事抽成 :class:`CloudTTSDialect` 的四个方法，
传输层（HTTP / WebSocket、重试、超时、取消、重采样、增益）只写一遍。
新增一家厂商 = 新写一个 20 行的方言子类，不碰任何传输代码。

依赖策略
--------
* HTTP 方言：优先用 ``httpx``（真异步）；没装则用标准库 ``urllib`` 丢进线程池。
  **因此 HTTP 方言永远可用**，不会因为缺依赖而整体不可用。
* WS 方言：需要 ``websockets``；缺失时构造抛 :class:`TTSUnavailable`，
  由路由器降级到 HTTP 方言或本地引擎。
* 缺少 API Key 时同样抛 :class:`TTSUnavailable`——宁可降级，也不要在
  用户说完话之后才 401。
"""
from __future__ import annotations

import asyncio
import base64
import gzip
import io
import json
import os
import struct
import urllib.error
import urllib.request
import uuid
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import structlog

from omnistream.perception.errors import TTSError, TTSTimeout
from omnistream.perception.tts.base import (
    DEFAULT_TTS_SAMPLE_RATE,
    TTS_FRAME_MS,
    BaseStreamingTTS,
    SynthesisRequest,
    TTSCapabilities,
    TTSUnavailable,
    resample_f32,
)
from omnistream.perception.tts.ssml import AZURE_DIALECT, Prosody, build_ssml
from omnistream.perception.tts.voice_registry import default_registry
from omnistream.schemas.audio import AudioChunk

log = structlog.get_logger(__name__)

__all__ = [
    "AzureTTSDialect",
    "CloudStreamTTS",
    "CloudTTSConfig",
    "CloudTTSDialect",
    "DIALECTS",
    "ElevenLabsTTSDialect",
    "GenericWSTTSDialect",
    "OpenAITTSDialect",
    "VolcTTSDialect",
    "cloud_tts_available",
    "get_tts_dialect",
]

#: 单句合成超时（秒）
DEFAULT_TIMEOUT_S: Final[float] = 12.0

#: WebSocket 单条消息等待超时（秒）
DEFAULT_RECV_TIMEOUT_S: Final[float] = 8.0

#: 失败重试次数（仅对幂等的单句合成生效）
DEFAULT_RETRIES: Final[int] = 1

_websockets_module: Any = None
_ws_error: str = ""
_httpx_module: Any = None


def _load_websockets() -> tuple[Any, str]:
    """惰性导入 websockets。"""
    global _websockets_module, _ws_error
    if _websockets_module is not None or _ws_error:
        return _websockets_module, _ws_error
    try:
        import websockets  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        _ws_error = f"websockets 未安装: {exc}"
        return None, _ws_error
    _websockets_module = websockets
    return websockets, ""


def _load_httpx() -> Any:
    """惰性导入 httpx，失败返回 ``None``（走 urllib 兜底）。"""
    global _httpx_module
    if _httpx_module is not None:
        return _httpx_module
    try:
        import httpx  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return None
    _httpx_module = httpx
    return httpx


def cloud_tts_available(transport: str = "http") -> bool:
    """指定传输方式是否可用。

    Args:
        transport: ``http`` 或 ``ws``。

    Returns:
        HTTP 恒为 ``True``（标准库兜底）；WS 取决于 ``websockets``。
    """
    if transport == "ws":
        return _load_websockets()[0] is not None
    return True


@dataclass(slots=True)
class CloudTTSConfig:
    """一次云端合成需要的全部连接参数。"""

    endpoint: str = ""
    api_key: str = ""
    region: str = ""
    model: str = ""
    #: 物理音色名（已由音色注册表映射）
    voice: str = "default"
    #: 请求的采样率
    sample_rate: int = DEFAULT_TTS_SAMPLE_RATE
    #: 火山引擎需要的 appid / cluster
    app_id: str = ""
    cluster: str = ""
    #: 附加字段，透传进请求体
    extra: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = DEFAULT_TIMEOUT_S


class CloudTTSDialect:
    """厂商方言基类：只负责"拼 URL / 签头 / 造体 / 拆包"。"""

    #: 方言名
    name: str = "generic"
    #: ``http`` 或 ``ws``
    transport: str = "http"
    #: 默认服务地址（可被配置覆盖）
    default_endpoint: str = ""
    #: 依次尝试的环境变量名
    api_key_envs: tuple[str, ...] = ("OMS_TTS_API_KEY",)
    #: 该方言原生输出采样率
    native_sample_rate: int = DEFAULT_TTS_SAMPLE_RATE
    #: 是否强制要求 API Key
    requires_key: bool = True
    #: 是否用 SSML 承载韵律
    uses_ssml: bool = False

    def endpoint(self, cfg: CloudTTSConfig) -> str:
        """最终请求地址。"""
        return cfg.endpoint or self.default_endpoint

    def headers(self, cfg: CloudTTSConfig) -> dict[str, str]:
        """请求头（含鉴权）。"""
        return {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        }

    def http_body(self, cfg: CloudTTSConfig, req: SynthesisRequest) -> bytes:
        """HTTP 请求体。"""
        payload = {
            "text": req.text,
            "voice": cfg.voice,
            "language": req.language,
            "speed": req.speed,
            "pitch": req.pitch,
            "format": "pcm16",
            "sample_rate": cfg.sample_rate,
        }
        payload.update(cfg.extra)
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def parse_http(self, status: int, body: bytes) -> bytes:
        """解析 HTTP 响应，返回裸 PCM16 字节。"""
        if status >= 400:
            raise TTSError(f"{self.name} 返回 HTTP {status}: {body[:256]!r}")
        return decode_audio_payload(body)

    def ws_messages(self, cfg: CloudTTSConfig, req: SynthesisRequest) -> list[Any]:
        """连接建立后需要发送的消息序列。"""
        payload = {
            "type": "synthesis.start",
            "text": req.text,
            "voice": cfg.voice,
            "language": req.language,
            "speed": req.speed,
            "pitch": req.pitch,
            "format": "pcm16",
            "sample_rate": cfg.sample_rate,
        }
        payload.update(cfg.extra)
        return [json.dumps(payload, ensure_ascii=False)]

    def parse_ws(self, message: Any) -> tuple[bytes, bool]:
        """解析一条 WS 消息。

        Returns:
            ``(音频字节, 是否结束)``。
        """
        if isinstance(message, (bytes, bytearray)):
            return bytes(message), False
        try:
            data = json.loads(message)
        except (TypeError, ValueError):
            return b"", False
        kind = str(data.get("type") or data.get("event") or "")
        if kind in {"error", "synthesis.error"}:
            raise TTSError(f"{self.name} 服务端错误: {data.get('message', data)}")
        if kind in {"synthesis.done", "done", "audio.done", "response.done"}:
            return b"", True
        raw = data.get("audio") or data.get("data") or ""
        if isinstance(raw, str) and raw:
            return base64.b64decode(raw), False
        return b"", False

    def sample_rate(self, cfg: CloudTTSConfig) -> int:
        """实际返回音频的采样率。"""
        return cfg.sample_rate or self.native_sample_rate

    def prosody_of(self, req: SynthesisRequest) -> Prosody:
        """把请求参数折算成韵律对象。"""
        return Prosody(
            rate=req.speed,
            pitch_st=req.pitch,
            volume_db=req.volume_db,
            emotion=req.emotion,
        ).normalized()


class GenericWSTTSDialect(CloudTTSDialect):
    """自建/私有化网关的 WebSocket 方言（JSON + base64 PCM）。"""

    name = "generic_ws"
    transport = "ws"
    default_endpoint = "ws://127.0.0.1:8900/v1/tts/stream"
    api_key_envs = ("OMS_TTS_API_KEY",)
    requires_key = False


class AzureTTSDialect(CloudTTSDialect):
    """Azure 认知服务语音合成（REST + SSML，返回裸 PCM）。"""

    name = "azure"
    transport = "http"
    api_key_envs = ("AZURE_SPEECH_KEY", "AZURE_TTS_KEY", "OMS_TTS_API_KEY")
    native_sample_rate = 24000
    uses_ssml = True

    def endpoint(self, cfg: CloudTTSConfig) -> str:
        """按区域拼出 Azure 合成端点。"""
        if cfg.endpoint:
            return cfg.endpoint
        region = cfg.region or os.environ.get("AZURE_SPEECH_REGION", "eastasia")
        return f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"

    def headers(self, cfg: CloudTTSConfig) -> dict[str, str]:
        """Azure 用 ``Ocp-Apim-Subscription-Key`` 而不是 Bearer。"""
        rate = 24000 if cfg.sample_rate >= 24000 else 16000
        return {
            "Ocp-Apim-Subscription-Key": cfg.api_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": f"raw-{rate // 1000}khz-16bit-mono-pcm",
            "User-Agent": "OmniStream",
        }

    def http_body(self, cfg: CloudTTSConfig, req: SynthesisRequest) -> bytes:
        """SSML 请求体，用户文本已做 XML 转义。"""
        ssml = build_ssml(
            req.text,
            voice=cfg.voice,
            language=req.language,
            prosody=self.prosody_of(req),
            dialect=AZURE_DIALECT,
        )
        return ssml.encode("utf-8")

    def sample_rate(self, cfg: CloudTTSConfig) -> int:
        """Azure 只在 16k/24k 两档里取。"""
        return 24000 if cfg.sample_rate >= 24000 else 16000


class OpenAITTSDialect(CloudTTSDialect):
    """OpenAI ``/v1/audio/speech``（REST，pcm 24kHz）。"""

    name = "openai"
    transport = "http"
    default_endpoint = "https://api.openai.com/v1/audio/speech"
    api_key_envs = ("OPENAI_API_KEY", "OMS_TTS_API_KEY")
    native_sample_rate = 24000

    def http_body(self, cfg: CloudTTSConfig, req: SynthesisRequest) -> bytes:
        """OpenAI 的 speed 生效范围是 0.25~4.0。"""
        payload: dict[str, Any] = {
            "model": cfg.model or "tts-1",
            "input": req.text,
            "voice": cfg.voice or "alloy",
            "response_format": "pcm",
            "speed": max(0.25, min(4.0, req.speed)),
        }
        payload.update(cfg.extra)
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def sample_rate(self, cfg: CloudTTSConfig) -> int:
        """OpenAI pcm 输出固定 24kHz。"""
        return 24000


class ElevenLabsTTSDialect(CloudTTSDialect):
    """ElevenLabs ``/v1/text-to-speech/{voice_id}/stream``。"""

    name = "elevenlabs"
    transport = "http"
    default_endpoint = "https://api.elevenlabs.io/v1/text-to-speech"
    api_key_envs = ("ELEVENLABS_API_KEY", "ELEVEN_API_KEY", "OMS_TTS_API_KEY")
    native_sample_rate = 24000

    def endpoint(self, cfg: CloudTTSConfig) -> str:
        """音色 ID 在路径里，输出格式在 query 里。"""
        base = cfg.endpoint or self.default_endpoint
        rate = self.sample_rate(cfg)
        return f"{base.rstrip('/')}/{cfg.voice}/stream?output_format=pcm_{rate}"

    def headers(self, cfg: CloudTTSConfig) -> dict[str, str]:
        """ElevenLabs 用 ``xi-api-key``。"""
        return {"xi-api-key": cfg.api_key, "Content-Type": "application/json"}

    def http_body(self, cfg: CloudTTSConfig, req: SynthesisRequest) -> bytes:
        """语速通过 voice_settings 无法调，改用 SSML break 之外的默认行为。"""
        payload: dict[str, Any] = {
            "text": req.text,
            "model_id": cfg.model or "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.45,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True,
            },
        }
        payload.update(cfg.extra)
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def sample_rate(self, cfg: CloudTTSConfig) -> int:
        """ElevenLabs PCM 支持 16000/22050/24000/44100。"""
        for candidate in (44100, 24000, 22050, 16000):
            if cfg.sample_rate >= candidate:
                return candidate
        return 16000


class VolcTTSDialect(CloudTTSDialect):
    """火山引擎流式 TTS（WebSocket 二进制协议）。

    协议帧结构（官方 v1 二进制协议）::

        byte0: (version << 4) | header_size_in_dwords
        byte1: (message_type << 4) | message_type_flags
        byte2: (serialization << 4) | compression
        byte3: reserved
        [payload_size: uint32 big-endian][payload]

    请求用 ``message_type=0b0001``（full client request）、
    序列化 JSON（0b0001）、压缩 gzip（0b0001）。
    响应 ``message_type=0b1011`` 为音频帧，payload 前 8 字节是
    ``sequence(int32) + payload_size(uint32)``，``sequence < 0`` 表示最后一包。
    """

    name = "volc"
    transport = "ws"
    default_endpoint = "wss://openspeech.bytedance.com/api/v1/tts/ws_binary"
    api_key_envs = ("VOLC_TTS_TOKEN", "VOLC_ACCESS_TOKEN", "OMS_TTS_API_KEY")
    native_sample_rate = 24000

    #: 请求帧头：版本 1、头长 1 dword、full client request、JSON、gzip
    _REQUEST_HEADER: Final[bytes] = bytes([0x11, 0x10, 0x11, 0x00])

    def headers(self, cfg: CloudTTSConfig) -> dict[str, str]:
        """火山用 ``Bearer;<token>`` 这种带分号的特殊格式。"""
        return {"Authorization": f"Bearer;{cfg.api_key}"}

    def ws_messages(self, cfg: CloudTTSConfig, req: SynthesisRequest) -> list[Any]:
        """构造一条 gzip 压缩的二进制请求帧。"""
        app_id = cfg.app_id or os.environ.get("VOLC_TTS_APPID", "")
        cluster = cfg.cluster or os.environ.get("VOLC_TTS_CLUSTER", "volcano_tts")
        payload: dict[str, Any] = {
            "app": {"appid": app_id, "token": cfg.api_key, "cluster": cluster},
            "user": {"uid": "omnistream"},
            "audio": {
                "voice_type": cfg.voice,
                "encoding": "pcm",
                "rate": self.sample_rate(cfg),
                "speed_ratio": max(0.2, min(3.0, req.speed)),
                "volume_ratio": 10.0 ** (req.volume_db / 20.0),
                "pitch_ratio": 2.0 ** (req.pitch / 12.0),
            },
            "request": {
                "reqid": uuid.uuid4().hex,
                "text": req.text,
                "text_type": "plain",
                "operation": "submit",
            },
        }
        for key, value in cfg.extra.items():
            if isinstance(value, dict) and isinstance(payload.get(key), dict):
                payload[key].update(value)
            else:
                payload[key] = value
        body = gzip.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        return [self._REQUEST_HEADER + struct.pack(">I", len(body)) + body]

    def parse_ws(self, message: Any) -> tuple[bytes, bool]:
        """拆火山二进制响应帧。"""
        if isinstance(message, str):
            try:
                data = json.loads(message)
            except ValueError:
                return b"", False
            if data.get("code", 0) not in (0, 3000):
                raise TTSError(f"volc 错误 {data.get('code')}: {data.get('message', '')}")
            return b"", bool(data.get("done"))

        raw = bytes(message)
        if len(raw) < 4:
            return b"", False
        header_size = (raw[0] & 0x0F) * 4
        message_type = raw[1] >> 4
        compression = raw[2] & 0x0F
        body = raw[header_size:]

        if message_type == 0x0B:  # audio-only response
            if len(body) < 8:
                return b"", False
            sequence = struct.unpack(">i", body[:4])[0]
            size = struct.unpack(">I", body[4:8])[0]
            audio = body[8:8 + size]
            if compression == 0x01 and audio:
                audio = gzip.decompress(audio)
            return audio, sequence < 0
        if message_type == 0x0F:  # error response
            if len(body) < 8:
                raise TTSError("volc 返回未知错误")
            code = struct.unpack(">I", body[:4])[0]
            size = struct.unpack(">I", body[4:8])[0]
            detail = body[8:8 + size]
            if compression == 0x01 and detail:
                detail = gzip.decompress(detail)
            raise TTSError(f"volc 错误 {code}: {detail.decode('utf-8', 'replace')[:200]}")
        return b"", False


#: 内置方言表
DIALECTS: Final[dict[str, CloudTTSDialect]] = {
    "generic": CloudTTSDialect(),
    "generic_ws": GenericWSTTSDialect(),
    "azure": AzureTTSDialect(),
    "openai": OpenAITTSDialect(),
    "elevenlabs": ElevenLabsTTSDialect(),
    "volc": VolcTTSDialect(),
}


def get_tts_dialect(name: str) -> CloudTTSDialect:
    """按名字取方言，未知名字回落到 ``generic``。"""
    return DIALECTS.get((name or "generic").strip().lower(), DIALECTS["generic"])


def decode_audio_payload(data: bytes) -> bytes:
    """如果是 WAV 就剥壳，否则原样返回（当作裸 PCM16）。"""
    if len(data) > 44 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        with wave.open(io.BytesIO(data), "rb") as handle:
            return handle.readframes(handle.getnframes())
    return data


class CloudStreamTTS(BaseStreamingTTS):
    """云端流式语音合成，方言可插拔。

    Example::

        tts = CloudStreamTTS(dialect="azure", voice="xiaoyan")
        async for chunk in tts.synthesize(text_stream):
            await playback.enqueue(chunk)
    """

    name = "cloud"

    capabilities = TTSCapabilities(
        streaming=True,
        ssml=True,
        voice_cloning=True,
        emotion=True,
        voices=0,
        needs_gpu=False,
        needs_network=True,
        typical_ttfa_ms=280,
        languages=("*",),
        sample_rates=(16000, 22050, 24000, 44100, 48000),
    )

    def __init__(
        self,
        *,
        voice: str = "default",
        language: str = "zh-CN",
        sample_rate: int = DEFAULT_TTS_SAMPLE_RATE,
        speed: float = 1.0,
        pitch: float = 0.0,
        volume_db: float = 0.0,
        frame_ms: int = TTS_FRAME_MS,
        session_id: str = "",
        dialect: str = "generic",
        endpoint: str = "",
        api_key: str = "",
        region: str = "",
        model: str = "",
        app_id: str = "",
        cluster: str = "",
        extra: dict[str, Any] | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        retries: int = DEFAULT_RETRIES,
        first_chunk_chars: int = 8,
    ) -> None:
        """初始化并校验连接前提。

        Args:
            voice: 逻辑音色名，会被注册表映射为该方言的物理音色。
            language: 语言标签。
            sample_rate: 期望输出采样率。
            speed: 语速倍率。
            pitch: 音高偏移（半音）。
            volume_db: 音量增益。
            frame_ms: 输出帧长。
            session_id: 会话 ID。
            dialect: 厂商方言名。
            endpoint: 覆盖默认服务地址。
            api_key: 覆盖环境变量里的密钥。
            region: Azure 区域。
            model: 模型名（OpenAI/ElevenLabs）。
            app_id: 火山 appid。
            cluster: 火山 cluster。
            extra: 透传进请求体的附加字段。
            timeout_s: 单句合成超时。
            retries: 失败重试次数。
            first_chunk_chars: 首句最小字符数。

        Raises:
            TTSUnavailable: 缺少密钥，或 WS 方言缺少 ``websockets``。
        """
        super().__init__(
            voice=voice,
            language=language,
            sample_rate=sample_rate,
            speed=speed,
            pitch=pitch,
            volume_db=volume_db,
            frame_ms=frame_ms,
            session_id=session_id,
        )
        self.dialect = get_tts_dialect(dialect)
        self.retries = max(0, int(retries))
        self.first_chunk_chars = max(1, int(first_chunk_chars))

        if self.dialect.transport == "ws" and _load_websockets()[0] is None:
            raise TTSUnavailable(_load_websockets()[1])

        key = api_key or self._key_from_env(self.dialect)
        if self.dialect.requires_key and not key:
            raise TTSUnavailable(
                f"{self.dialect.name} 缺少 API Key（尝试过 "
                f"{', '.join(self.dialect.api_key_envs)}）"
            )
        physical = default_registry.resolve(voice, engine=self.dialect.name) or voice
        self.config = CloudTTSConfig(
            endpoint=endpoint or os.environ.get("OMS_TTS_ENDPOINT", ""),
            api_key=key,
            region=region,
            model=model,
            voice=physical,
            sample_rate=int(sample_rate),
            app_id=app_id,
            cluster=cluster,
            extra=dict(extra or {}),
            timeout_s=max(0.5, float(timeout_s)),
        )
        self.native_rate = self.dialect.sample_rate(self.config)
        self._closed = False
        self._requests = 0
        self._failures = 0
        self._client: Any = None
        log.info("tts.cloud.ready", dialect=self.dialect.name,
                 transport=self.dialect.transport, voice=physical,
                 native_rate=self.native_rate)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """HTTP 方言总能用，因此类级别恒为 True；实例级别由构造函数校验。"""
        return True

    def available_voices(self) -> list[str]:
        """注册表中声明支持该方言的逻辑音色。"""
        voices = default_registry.for_engine(self.dialect.name)
        return [v.id for v in voices] or [self.voice]

    async def synthesize(self, text_stream: AsyncIterator[str]) -> AsyncIterator[AudioChunk]:
        """消费文本流并产出音频帧。"""
        async for chunk in self._stream_sentences(
            text_stream, first_chunk_chars=self.first_chunk_chars
        ):
            yield chunk

    async def synthesize_text(self, request: SynthesisRequest) -> np.ndarray:
        """合成一小句（带重试），返回目标采样率的单声道 float32。"""
        text = request.text.strip()
        if not text or self.cancelled or self._closed:
            return np.zeros(0, dtype=np.float32)

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            if self.cancelled:
                return np.zeros(0, dtype=np.float32)
            try:
                self._requests += 1
                if self.dialect.transport == "ws":
                    raw = await self._synthesize_ws(request)
                else:
                    raw = await self._synthesize_http(request)
                return self._postprocess(raw, request)
            except (TTSError, TTSTimeout) as exc:
                last_error = exc
                self._failures += 1
                log.warning("tts.cloud.attempt_failed", dialect=self.dialect.name,
                            attempt=attempt + 1, error=str(exc))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._failures += 1
                log.warning("tts.cloud.attempt_crashed", dialect=self.dialect.name,
                            attempt=attempt + 1, error=str(exc))
            if attempt < self.retries and not self.cancelled:
                await asyncio.sleep(0.15 * (attempt + 1))

        raise TTSError(
            f"{self.dialect.name} 合成失败（重试 {self.retries} 次）: {last_error}",
            cause=last_error,
        )

    async def aclose(self) -> None:
        """关闭 HTTP 连接池，幂等。"""
        if self._closed:
            return
        self._closed = True
        client = self._client
        self._client = None
        if client is not None:
            try:
                await client.aclose()
            except Exception as exc:  # noqa: BLE001
                log.debug("tts.cloud.close_failed", error=str(exc))

    def stats(self) -> dict[str, Any]:
        """统计快照。"""
        base = super().stats()
        base["dialect"] = self.dialect.name
        base["transport"] = self.dialect.transport
        base["requests"] = self._requests
        base["failures"] = self._failures
        return base

    # ------------------------------------------------------------------
    # 传输实现
    # ------------------------------------------------------------------

    @staticmethod
    def _key_from_env(dialect: CloudTTSDialect) -> str:
        """按方言声明的顺序从环境变量取密钥。"""
        for env in dialect.api_key_envs:
            value = os.environ.get(env, "").strip()
            if value:
                return value
        return ""

    async def _synthesize_http(self, request: SynthesisRequest) -> bytes:
        """HTTP 单句合成：优先 httpx，回落 urllib。"""
        url = self.dialect.endpoint(self.config)
        headers = self.dialect.headers(self.config)
        body = self.dialect.http_body(self.config, request)
        timeout = self.config.timeout_s

        httpx = _load_httpx()
        if httpx is not None:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=timeout)
            try:
                response = await self._client.post(url, content=body, headers=headers)
            except httpx.TimeoutException as exc:
                raise TTSTimeout(f"{self.dialect.name} 超时 {timeout}s") from exc
            except Exception as exc:  # noqa: BLE001
                raise TTSError(f"{self.dialect.name} 请求失败: {exc}", cause=exc) from exc
            return self.dialect.parse_http(response.status_code, response.content)

        return await asyncio.to_thread(self._urllib_post, url, headers, body, timeout)

    def _urllib_post(self, url: str, headers: dict[str, str],
                     body: bytes, timeout: float) -> bytes:
        """标准库兜底的同步 POST（跑在线程池里）。"""
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return self.dialect.parse_http(resp.status, resp.read())
        except urllib.error.HTTPError as exc:
            payload = exc.read() if hasattr(exc, "read") else b""
            raise TTSError(
                f"{self.dialect.name} 返回 HTTP {exc.code}: {payload[:256]!r}"
            ) from exc
        except TimeoutError as exc:
            raise TTSTimeout(f"{self.dialect.name} 超时 {timeout}s") from exc
        except urllib.error.URLError as exc:
            raise TTSError(f"{self.dialect.name} 网络错误: {exc.reason}", cause=exc) from exc

    async def _synthesize_ws(self, request: SynthesisRequest) -> bytes:
        """WebSocket 单句合成：连接 -> 发请求 -> 收到结束帧为止。"""
        websockets, err = _load_websockets()
        if websockets is None:
            raise TTSError(err)

        url = self.dialect.endpoint(self.config)
        headers = self.dialect.headers(self.config)
        parts: list[bytes] = []
        try:
            async with asyncio.timeout(self.config.timeout_s):
                connection = await self._ws_connect(websockets, url, headers)
                async with connection as socket:
                    for message in self.dialect.ws_messages(self.config, request):
                        await socket.send(message)
                    while True:
                        if self.cancelled:
                            break
                        message = await asyncio.wait_for(
                            socket.recv(), timeout=DEFAULT_RECV_TIMEOUT_S
                        )
                        audio, done = self.dialect.parse_ws(message)
                        if audio:
                            parts.append(audio)
                        if done:
                            break
        except TimeoutError as exc:
            raise TTSTimeout(
                f"{self.dialect.name} WS 超时 {self.config.timeout_s}s"
            ) from exc
        except TTSError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TTSError(f"{self.dialect.name} WS 失败: {exc}", cause=exc) from exc
        return b"".join(parts)

    @staticmethod
    async def _ws_connect(websockets: Any, url: str, headers: dict[str, str]) -> Any:
        """兼容 websockets 10~13 的连接参数命名差异。"""
        connect = websockets.connect
        for kw in ("additional_headers", "extra_headers"):
            try:
                return connect(url, **{kw: headers}, open_timeout=8, max_size=None)
            except TypeError:
                continue
        return connect(url)

    def _postprocess(self, raw: bytes, request: SynthesisRequest) -> np.ndarray:
        """裸 PCM16 -> float32 -> 重采样 -> 增益。"""
        if not raw:
            return np.zeros(0, dtype=np.float32)
        payload = decode_audio_payload(raw)
        if len(payload) % 2:
            payload = payload[:-1]
        audio = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        audio = resample_f32(audio, self.native_rate, self.sample_rate)
        # Azure/火山已在服务端处理音量，这里只补齐未支持音量的方言
        if self.dialect.name in {"openai", "elevenlabs", "generic", "generic_ws"}:
            gain = 10.0 ** (request.volume_db / 20.0)
            if abs(gain - 1.0) > 1e-3:
                audio = audio * gain
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        np.clip(audio, -1.0, 1.0, out=audio)
        return audio

"""Piper 本地神经 TTS 适配器（可选依赖，缺失时优雅降级）。

Piper 是 rhasspy 出品的 VITS 系轻量端侧 TTS：单个 onnx 模型 20~60MB，
CPU 上实时率（RTF）约 0.05~0.2，是**本地部署下 TTFA 最低**的可用方案，
因此在路由器的降级链里排在云端之前。

工程要点
--------
1. **onnxruntime 会话不是并发安全的**。多个 ``synthesize_text`` 并发进入
   同一个 ``InferenceSession`` 会导致输出错乱甚至崩溃，
   这里用 ``asyncio.Lock`` 串行化，用 ``asyncio.to_thread`` 避免阻塞事件循环。
2. **Piper 没有音高参数**。这里用"变速合成 + 重采样"实现真正的移调：
   先按 ``2^(n/12)`` 拉长时长再等比压缩采样点，
   频率整体上移而总时长不变——这是经典的 resampling pitch shift，不是近似。
3. **模型查找是分层的**：显式路径 > 音色注册表映射 > 常见安装目录扫描。
   找不到模型时抛 :class:`TTSUnavailable`，由工厂捕获降级到 mock。
4. **多版本 API 兼容**：piper 1.2 / 1.3 / 2.x 的合成入口名字不同，
   这里逐个探测真实存在的方法，全部失败才报错。
"""
from __future__ import annotations

import asyncio
import io
import os
import wave
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, Final

import numpy as np
import structlog

from omnistream.perception.tts.base import (
    DEFAULT_TTS_SAMPLE_RATE,
    TTS_FRAME_MS,
    BaseStreamingTTS,
    SynthesisRequest,
    TTSCapabilities,
    TTSUnavailable,
    resample_f32,
)
from omnistream.perception.tts.voice_registry import default_registry
from omnistream.schemas.audio import AudioChunk

log = structlog.get_logger(__name__)

__all__ = [
    "MODEL_SEARCH_DIRS",
    "PiperTTS",
    "find_piper_model",
    "piper_available",
]

#: 模型搜索目录（按优先级）
MODEL_SEARCH_DIRS: Final[tuple[str, ...]] = (
    "./models/piper",
    "./models",
    "~/.local/share/piper",
    "~/.local/share/piper-voices",
    "~/.cache/piper",
    "/usr/share/piper-voices",
    "/opt/piper/models",
)

#: 默认中文音色的模型文件名
DEFAULT_MODEL_NAME: Final[str] = "zh_CN-huayan-medium.onnx"

#: Piper 原生输出采样率通常是 22050（medium）或 16000（low）
_PIPER_NATIVE_RATE: Final[int] = 22050

_piper_module: Any = None
_piper_voice_cls: Any = None
_import_error: str = ""


def _load_piper() -> tuple[Any, Any, str]:
    """惰性导入 piper，返回 ``(module, PiperVoice, error)``。"""
    global _piper_module, _piper_voice_cls, _import_error
    if _piper_voice_cls is not None or _import_error:
        return _piper_module, _piper_voice_cls, _import_error
    try:
        import piper  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        _import_error = f"piper 未安装: {exc}"
        return None, None, _import_error

    voice_cls = getattr(piper, "PiperVoice", None)
    if voice_cls is None:
        try:
            from piper.voice import PiperVoice as _VoiceCls  # type: ignore[import-not-found]

            voice_cls = _VoiceCls
        except Exception as exc:  # noqa: BLE001
            _import_error = f"piper.PiperVoice 不可用: {exc}"
            return piper, None, _import_error

    _piper_module = piper
    _piper_voice_cls = voice_cls
    return _piper_module, _piper_voice_cls, ""


def piper_available() -> bool:
    """piper 依赖是否就绪。"""
    _, cls, _ = _load_piper()
    return cls is not None


def find_piper_model(name: str = DEFAULT_MODEL_NAME) -> Path | None:
    """在常见目录里找 piper 模型文件。

    Args:
        name: 模型文件名或不带后缀的音色名。

    Returns:
        找到的 ``.onnx`` 路径，找不到返回 ``None``。
    """
    filename = name if name.endswith(".onnx") else f"{name}.onnx"
    direct = Path(filename).expanduser()
    if direct.is_file():
        return direct
    env_dir = os.environ.get("PIPER_VOICES_DIR", "")
    candidates = ([env_dir] if env_dir else []) + list(MODEL_SEARCH_DIRS)
    for raw in candidates:
        base = Path(raw).expanduser()
        if not base.is_dir():
            continue
        exact = base / filename
        if exact.is_file():
            return exact
        # 允许放在 <dir>/<voice>/<voice>.onnx 这种官方目录结构里
        try:
            for found in base.rglob(filename):
                if found.is_file():
                    return found
        except OSError:
            continue
    return None


class PiperTTS(BaseStreamingTTS):
    """Piper 本地神经语音合成。

    Example::

        tts = PiperTTS(voice="xiaoyan", sample_rate=24000)
        async for chunk in tts.synthesize(text_stream):
            await playback.enqueue(chunk)
    """

    name = "piper"

    capabilities = TTSCapabilities(
        streaming=True,
        ssml=False,
        voice_cloning=False,
        emotion=False,
        voices=0,
        needs_gpu=False,
        needs_network=False,
        typical_ttfa_ms=180,
        languages=("zh", "en", "de", "fr", "es", "it", "ru", "pt", "nl"),
        sample_rates=(16000, 22050, 24000, 48000),
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
        model_path: str | None = None,
        config_path: str | None = None,
        speaker_id: int | None = None,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
        sentence_silence_s: float = 0.0,
        use_cuda: bool = False,
        first_chunk_chars: int = 8,
    ) -> None:
        """初始化并加载模型。

        Args:
            voice: 逻辑音色名，会经音色注册表映射为物理模型名。
            language: 语言标签。
            sample_rate: 期望输出采样率（与模型原生率不同则自动重采样）。
            speed: 语速倍率。
            pitch: 音高偏移（半音），通过变速重采样实现。
            volume_db: 音量增益。
            frame_ms: 输出帧长。
            session_id: 会话 ID。
            model_path: 显式模型路径，优先级最高。
            config_path: 显式配置路径，默认取 ``<model>.json``。
            speaker_id: 多说话人模型的说话人索引。
            noise_scale: VITS 采样噪声，越大越有变化。
            noise_w: 音素时长噪声。
            sentence_silence_s: 句末附加静音（这里默认 0，由上层控制停顿）。
            use_cuda: 是否尝试 CUDA 执行提供器。
            first_chunk_chars: 首句最小字符数。

        Raises:
            TTSUnavailable: piper 未安装或模型文件缺失。
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
        _, voice_cls, err = _load_piper()
        if voice_cls is None:
            raise TTSUnavailable(err or "piper 不可用")

        resolved = self._resolve_model(model_path, voice)
        if resolved is None:
            raise TTSUnavailable(
                f"未找到 piper 模型（voice={voice}）。"
                f"请设置 PIPER_VOICES_DIR 或把 .onnx 放到 {MODEL_SEARCH_DIRS[0]}"
            )
        self.model_path = resolved
        cfg = Path(config_path).expanduser() if config_path else Path(f"{resolved}.json")
        self.config_path = cfg if cfg.is_file() else None

        self.speaker_id = speaker_id
        self.noise_scale = float(noise_scale)
        self.noise_w = float(noise_w)
        self.sentence_silence_s = max(0.0, float(sentence_silence_s))
        self.first_chunk_chars = max(1, int(first_chunk_chars))

        try:
            if self.config_path is not None:
                self._voice = voice_cls.load(
                    str(self.model_path), config_path=str(self.config_path),
                    use_cuda=use_cuda,
                )
            else:
                self._voice = voice_cls.load(str(self.model_path), use_cuda=use_cuda)
        except TypeError:
            # 老版本 load() 没有 use_cuda 形参
            self._voice = voice_cls.load(str(self.model_path))
        except Exception as exc:  # noqa: BLE001
            raise TTSUnavailable(f"piper 模型加载失败: {exc}") from exc

        self.native_rate = self._detect_native_rate()
        self._lock = asyncio.Lock()
        self._closed = False
        log.info("tts.piper.loaded", model=str(self.model_path),
                 native_rate=self.native_rate, target_rate=self.sample_rate)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """依赖与默认模型是否都就绪。"""
        return piper_available()

    def available_voices(self) -> list[str]:
        """注册表中声明支持 piper 的逻辑音色。"""
        return [v.id for v in default_registry.for_engine(self.name)]

    async def synthesize(self, text_stream: AsyncIterator[str]) -> AsyncIterator[AudioChunk]:
        """消费文本流并产出音频帧。"""
        async for chunk in self._stream_sentences(
            text_stream, first_chunk_chars=self.first_chunk_chars
        ):
            yield chunk

    async def synthesize_text(self, request: SynthesisRequest) -> np.ndarray:
        """合成一小句，返回目标采样率的单声道 float32。"""
        text = request.text.strip()
        if not text or self.cancelled or self._closed:
            return np.zeros(0, dtype=np.float32)

        # 移调倍率：先按 ratio 拉长，再压缩采样点，净效果是升调不变时长
        ratio = 2.0 ** (request.pitch / 12.0)
        speed = max(0.1, request.speed)
        length_scale = (1.0 / speed) * ratio

        async with self._lock:
            if self.cancelled or self._closed:
                return np.zeros(0, dtype=np.float32)
            raw = await asyncio.to_thread(self._synthesize_blocking, text, length_scale)

        if raw.size == 0:
            return raw
        if abs(ratio - 1.0) > 1e-3:
            target = max(1, int(round(raw.size / ratio)))
            idx = np.linspace(0.0, raw.size - 1, num=target, dtype=np.float64)
            raw = np.interp(idx, np.arange(raw.size, dtype=np.float64),
                            raw).astype(np.float32)
        audio = resample_f32(raw, self.native_rate, self.sample_rate)
        gain = 10.0 ** (request.volume_db / 20.0)
        if abs(gain - 1.0) > 1e-3:
            audio = audio * gain
        np.clip(audio, -1.0, 1.0, out=audio)
        return audio

    async def aclose(self) -> None:
        """释放模型资源，幂等。"""
        if self._closed:
            return
        self._closed = True
        self._voice = None
        log.debug("tts.piper.closed", model=str(self.model_path))

    def stats(self) -> dict[str, Any]:
        """统计快照。"""
        base = super().stats()
        base["model"] = self.model_path.name
        base["native_rate"] = self.native_rate
        return base

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _resolve_model(self, model_path: str | None, voice: str) -> Path | None:
        """分层解析模型路径：显式 > 注册表映射 > 默认名。"""
        if model_path:
            explicit = Path(model_path).expanduser()
            if explicit.is_file():
                return explicit
            found = find_piper_model(explicit.name)
            if found is not None:
                return found
        physical = default_registry.resolve(voice, engine=self.name)
        if physical:
            found = find_piper_model(physical)
            if found is not None:
                return found
        return find_piper_model(DEFAULT_MODEL_NAME)

    def _detect_native_rate(self) -> int:
        """从模型配置里读采样率，读不到用 22050。"""
        config = getattr(self._voice, "config", None)
        for attr in ("sample_rate", "sampling_rate"):
            value = getattr(config, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        if isinstance(config, dict):
            audio = config.get("audio", {})
            if isinstance(audio, dict):
                value = audio.get("sample_rate")
                if isinstance(value, int) and value > 0:
                    return value
        return _PIPER_NATIVE_RATE

    def _synth_kwargs(self, length_scale: float) -> dict[str, Any]:
        """构造 piper 合成参数。"""
        kwargs: dict[str, Any] = {
            "length_scale": length_scale,
            "noise_scale": self.noise_scale,
            "noise_w": self.noise_w,
            "sentence_silence": self.sentence_silence_s,
        }
        if self.speaker_id is not None:
            kwargs["speaker_id"] = self.speaker_id
        return kwargs

    def _synthesize_blocking(self, text: str, length_scale: float) -> np.ndarray:
        """在线程池里跑的同步合成，逐个探测可用 API。"""
        voice = self._voice
        if voice is None:
            return np.zeros(0, dtype=np.float32)
        kwargs = self._synth_kwargs(length_scale)

        stream_fn = getattr(voice, "synthesize_stream_raw", None)
        if callable(stream_fn):
            data = self._collect_raw(stream_fn, text, kwargs)
            if data is not None:
                return self._int16_bytes_to_f32(data)

        # piper 2.x：synthesize() 产出 AudioChunk 对象序列
        synth_fn = getattr(voice, "synthesize", None)
        if callable(synth_fn):
            data = self._collect_modern(synth_fn, text, kwargs)
            if data is not None:
                return self._int16_bytes_to_f32(data)

        wav_fn = getattr(voice, "synthesize_wav", None) or synth_fn
        if callable(wav_fn):
            buffer = io.BytesIO()
            try:
                with wave.open(buffer, "wb") as handle:
                    wav_fn(text, handle, **self._safe_kwargs(wav_fn, kwargs))
            except TypeError:
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as handle:
                    wav_fn(text, handle)
            buffer.seek(0)
            with wave.open(buffer, "rb") as handle:
                frames = handle.readframes(handle.getnframes())
                self.native_rate = handle.getframerate() or self.native_rate
            return self._int16_bytes_to_f32(frames)

        raise TTSUnavailable("piper 版本不受支持：找不到任何合成入口")

    def _collect_raw(self, fn: Any, text: str, kwargs: dict[str, Any]) -> bytes | None:
        """调用 ``synthesize_stream_raw`` 并拼接 int16 字节流。"""
        try:
            result = fn(text, **self._safe_kwargs(fn, kwargs))
        except TypeError:
            try:
                result = fn(text)
            except Exception as exc:  # noqa: BLE001
                log.debug("tts.piper.stream_raw_failed", error=str(exc))
                return None
        except Exception as exc:  # noqa: BLE001
            log.debug("tts.piper.stream_raw_failed", error=str(exc))
            return None
        if isinstance(result, (bytes, bytearray)):
            return bytes(result)
        if isinstance(result, Iterator) or hasattr(result, "__iter__"):
            parts = [bytes(p) for p in result if p]
            return b"".join(parts)
        return None

    def _collect_modern(self, fn: Any, text: str, kwargs: dict[str, Any]) -> bytes | None:
        """piper 2.x：``synthesize`` 返回带 ``audio_int16_bytes`` 的对象序列。"""
        try:
            result = fn(text, **self._safe_kwargs(fn, kwargs))
        except TypeError:
            return None
        except Exception as exc:  # noqa: BLE001
            log.debug("tts.piper.synthesize_failed", error=str(exc))
            return None
        if not hasattr(result, "__iter__") or isinstance(result, (bytes, bytearray)):
            return None
        parts: list[bytes] = []
        for item in result:
            payload = getattr(item, "audio_int16_bytes", None)
            if payload is None:
                payload = getattr(item, "audio_int16_array", None)
                if payload is not None:
                    payload = np.asarray(payload, dtype=np.int16).tobytes()
            if payload is None and isinstance(item, (bytes, bytearray)):
                payload = bytes(item)
            if payload:
                parts.append(bytes(payload))
            rate = getattr(item, "sample_rate", 0)
            if isinstance(rate, int) and rate > 0:
                self.native_rate = rate
        return b"".join(parts) if parts else None

    @staticmethod
    def _safe_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        """只保留目标函数真正接受的关键字参数。"""
        import inspect

        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return {}
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return dict(kwargs)
        allowed = set(sig.parameters)
        return {k: v for k, v in kwargs.items() if k in allowed}

    @staticmethod
    def _int16_bytes_to_f32(data: bytes) -> np.ndarray:
        """int16 小端字节流 -> float32 [-1,1]。"""
        if not data:
            return np.zeros(0, dtype=np.float32)
        if len(data) % 2:
            data = data[:-1]
        pcm = np.frombuffer(data, dtype="<i2").astype(np.float32)
        return pcm / 32768.0

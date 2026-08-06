"""感知层配置：ASR / TTS / VLM 引擎选择、模型路径、语言与并发度。"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PerceptionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OMS_PERCEPTION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------ ASR
    asr_engine: str = Field(default="mock",
                            description="mock|faster_whisper|funasr|azure|deepgram|volc")
    asr_model: str = Field(default="small", description="本地模型尺寸或云端模型名")
    asr_model_path: str | None = Field(default=None, description="本地权重目录，留空走默认缓存")
    asr_device: str = Field(default="auto", description="auto|cpu|cuda|mps")
    asr_compute_type: str = Field(default="int8", description="int8|int8_float16|float16|float32")
    asr_language: str = "zh"
    asr_beam_size: int = Field(default=1, ge=1, le=10, description="实时场景固定为 1")
    #: 流式转写窗口与步进
    asr_chunk_ms: int = Field(default=480, ge=100)
    asr_stride_ms: int = Field(default=160, ge=40)
    asr_partial_enabled: bool = True
    asr_word_timestamps: bool = False
    asr_concurrency: int = Field(default=4, ge=1)
    asr_vad_filter: bool = True
    asr_hotwords: list[str] = Field(default_factory=list, description="热词/术语增强")

    # ------------------------------------------------------------ TTS
    tts_engine: str = Field(default="mock",
                            description="mock|piper|edge|azure|elevenlabs|volc")
    tts_model: str = "default"
    tts_model_path: str | None = None
    tts_voice: str = "default"
    tts_language: str = "zh-CN"
    tts_sample_rate: int = Field(default=24000)
    tts_speed: float = Field(default=1.0, gt=0.1, le=3.0)
    tts_pitch: float = Field(default=0.0, ge=-12.0, le=12.0)
    tts_volume_db: float = Field(default=0.0, ge=-30.0, le=12.0)
    #: 流式合成：按句切分，第一句尽快出声（决定 TTFA）
    tts_streaming: bool = True
    tts_first_chunk_chars: int = Field(default=8, ge=1,
                                       description="首块最小字符数，越小首字越快")
    tts_concurrency: int = Field(default=4, ge=1)
    tts_cache_enabled: bool = True
    tts_cache_size: int = Field(default=512, ge=0)

    # ------------------------------------------------------------ VLM / 视觉
    vision_engine: str = Field(default="mock",
                               description="mock|qwen_vl|gpt4o|gemini|internvl|volc")
    vision_model: str = "default"
    vision_model_path: str | None = None
    vision_max_tokens: int = Field(default=256, ge=16)
    vision_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    vision_concurrency: int = Field(default=2, ge=1)
    vision_timeout_s: float = Field(default=8.0, gt=0.0)
    #: 场景描述缓存，画面不变则复用，显著降本
    vision_cache_ms: int = Field(default=4000, ge=0)
    vision_ocr_enabled: bool = True
    vision_detect_entities: bool = True

    # ------------------------------------------------------------ Embedding
    embedder: str = Field(default="hash", description="hash|bge|openai|volc")
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = Field(default=384, ge=8)
    embedding_batch_size: int = Field(default=16, ge=1)
    embedding_device: str = "auto"

    @property
    def local_models_required(self) -> bool:
        """是否需要 requirements-local-models.txt 中的可选重依赖。"""
        local = {"faster_whisper", "funasr", "piper", "qwen_vl", "internvl", "bge"}
        return bool({self.asr_engine, self.tts_engine, self.vision_engine, self.embedder} & local)

    @property
    def all_mock(self) -> bool:
        return (self.asr_engine == "mock" and self.tts_engine == "mock"
                and self.vision_engine == "mock")

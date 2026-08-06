"""全局配置。读取顺序：默认值 < config/*.yaml < 环境变量 < .env。

嵌套用双下划线：``OMS_PROVIDERS__DEFAULT=mock``、``OMS_STORAGE__REDIS_URL=redis://...``
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from omnistream.config.duplex import DuplexSettings
from omnistream.config.media import MediaSettings
from omnistream.config.memory import MemorySettings
from omnistream.config.observability import ObservabilitySettings
from omnistream.config.perception import PerceptionSettings
from omnistream.config.providers import ProvidersSettings
from omnistream.config.security import SecuritySettings
from omnistream.config.storage import StorageSettings


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OMS_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    env: Literal["dev", "staging", "prod"] = "dev"
    debug: bool = False
    app_name: str = "omnistream"
    host: str = "0.0.0.0"
    port: int = 8080
    root_path: str = ""
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    #: 关键开关：True 时全栈使用零依赖 Mock 组件（Provider/ASR/TTS/Vision/Embedding/Store）
    mock_mode: bool = False

    media: MediaSettings = Field(default_factory=MediaSettings)
    duplex: DuplexSettings = Field(default_factory=DuplexSettings)
    perception: PerceptionSettings = Field(default_factory=PerceptionSettings)
    providers: ProvidersSettings = Field(default_factory=ProvidersSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    def apply_mock_overrides(self) -> "Settings":
        """mock_mode 下强制切换所有外部依赖为零依赖实现。"""
        if not self.mock_mode:
            return self
        self.providers.default = "mock"
        self.perception.asr_engine = "mock"
        self.perception.tts_engine = "mock"
        self.perception.vision_engine = "mock"
        self.duplex.vad_backend = "energy"
        self.memory.store = "inmemory"
        self.memory.embedder = "hash"
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings().apply_mock_overrides()

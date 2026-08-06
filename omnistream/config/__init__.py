"""配置包入口。全局唯一取配置方式：``from omnistream.config import get_settings``。"""
from __future__ import annotations

from omnistream.config.app import AppSettings
from omnistream.config.duplex import DuplexSettings
from omnistream.config.media import MediaSettings
from omnistream.config.memory import MemorySettings
from omnistream.config.observability import ObservabilitySettings
from omnistream.config.perception import PerceptionSettings
from omnistream.config.providers import ProviderEntry, ProvidersSettings
from omnistream.config.security import SecuritySettings
from omnistream.config.settings import Settings, get_settings
from omnistream.config.storage import StorageSettings


def reload_settings() -> Settings:
    """清空缓存并重新读取配置。仅供测试与运维热更新使用。"""
    get_settings.cache_clear()
    return get_settings()


__all__ = [
    "AppSettings",
    "DuplexSettings",
    "MediaSettings",
    "MemorySettings",
    "ObservabilitySettings",
    "PerceptionSettings",
    "ProviderEntry",
    "ProvidersSettings",
    "SecuritySettings",
    "Settings",
    "StorageSettings",
    "get_settings",
    "reload_settings",
]

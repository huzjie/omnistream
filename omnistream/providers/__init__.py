"""Provider 模块统一出口。

同时提供 transcode 辅助工具与 ProviderCapabilities 快捷常量。
"""
from __future__ import annotations

from omnistream.providers.base import BaseRealtimeProvider, ProviderEvent
from omnistream.providers.registry import (
    ProviderRegistry,
    get_registry,
    registry,
    reset_registry,
)

__all__ = [
    "BaseRealtimeProvider",
    "ProviderEvent",
    "ProviderRegistry",
    "get_registry",
    "registry",
    "reset_registry",
]

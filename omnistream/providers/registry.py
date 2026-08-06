"""Provider 注册表与工厂。

负责：
- 收集所有内置 Provider 实现
- 按配置允许名单注册
- 依据 ``providers.default`` 提供默认 Provider
- 依据 ``session.provider`` 按名解析并创建实例

路由约定：``ProviderRegistry.get(name)`` 返回未连接的实例，
连接生命周期由调用方（对话编排器）负责。
"""
from __future__ import annotations

import importlib
import logging
from typing import Any, Type

from omnistream.config.providers import ProvidersSettings
from omnistream.core.errors import (
    ConfigError,
    NotFoundError,
    ProviderError,
    ProviderUnauthorized,
)
from omnistream.providers.base import BaseRealtimeProvider
from omnistream.schemas.provider import (
    ProviderCapabilities,
    ProviderCredential,
    ProviderInfo,
)

logger = logging.getLogger(__name__)

#: 内置 Provider 实现映射：注册名 -> (模块路径, 类名)
_BUILTIN_PROVIDERS: dict[str, tuple[str, str]] = {
    "mock": ("omnistream.providers.mock.mock_provider", "MockRealtimeProvider"),
    "doubao_seedrealtime": ("omnistream.providers.doubao", "DoubaoSeedRealtimeProvider"),
    "openai_realtime": ("omnistream.providers.openai_realtime", "OpenAIRealtimeProvider"),
    "gemini_live": ("omnistream.providers.gemini_live", "GeminiLiveProvider"),
    "qwen_omni": ("omnistream.providers.qwen_omni", "QwenOmniProvider"),
    "deepseek_v4": ("omnistream.providers.deepseek_v4", "DeepSeekV4Provider"),
    "kimi_k3": ("omnistream.providers.kimi_k3", "KimiK3Provider"),
    "local_pipeline": ("omnistream.providers.local_pipeline", "LocalPipelineProvider"),
}


class ProviderRegistry:
    """Provider 注册表（进程级单例）。"""

    _instance: "ProviderRegistry | None" = None

    def __init__(self, settings: ProvidersSettings) -> None:
        self._settings = settings
        self._classes: dict[str, Type[BaseRealtimeProvider]] = {}
        self._load_builtins()

    # ---------------- 构建 ----------------

    @classmethod
    def get_instance(cls, settings: ProvidersSettings | None = None) -> "ProviderRegistry":
        if cls._instance is None:
            from omnistream.config.settings import get_settings

            cfg = settings or get_settings().providers
            cls._instance = cls(cfg)
        return cls._instance

    def _load_builtins(self) -> None:
        # 以配置中启用的 provider 为准；未显式配置时全部加载
        enabled = set(self._settings.enabled_names())
        allowed = enabled or set(_BUILTIN_PROVIDERS.keys())
        for name, (mod, cls) in _BUILTIN_PROVIDERS.items():
            if name not in allowed:
                continue
            try:
                module = importlib.import_module(mod)
                klass: Type[BaseRealtimeProvider] = getattr(module, cls)
                self._classes[name] = klass
            except Exception as exc:  # noqa: BLE001 - 单点失败不应阻断整体
                logger.warning("provider 加载失败 name=%s err=%s", name, exc)

    # ---------------- 查询 ----------------

    def names(self) -> list[str]:
        return sorted(self._classes.keys())

    def info(self, name: str) -> ProviderInfo:
        klass = self._resolve(name)
        caps: ProviderCapabilities = klass.capabilities
        configured = self._is_configured(name)
        return ProviderInfo(
            name=name,
            display_name=getattr(klass, "display_name", name),
            vendor=getattr(klass, "vendor", ""),
            kind="native_duplex" if caps.native_duplex else "composite",
            models=list(getattr(klass, "supported_models", [])),
            default_model=getattr(klass, "default_model", None),
            capabilities=caps,
            enabled=name in self._settings.enabled_names(),
            configured=configured,
            docs_url=getattr(klass, "docs_url", None),
            notes=getattr(klass, "notes", ""),
        )

    def list_info(self) -> list[ProviderInfo]:
        return [self.info(n) for n in self.names()]

    # ---------------- 实例化 ----------------

    def create(self, name: str | None = None) -> BaseRealtimeProvider:
        target = name or self._settings.default
        klass = self._resolve(target)
        cred = self._credential_for(target)
        if not self._is_configured(target) and target != "mock":
            raise ConfigError(
                f"provider '{target}' 未配置凭据，请设置对应的 API_KEY 环境变量",
            )
        return self._instantiate(klass, cred)

    @staticmethod
    def _instantiate(klass: Type[BaseRealtimeProvider],
                     cred: ProviderCredential) -> BaseRealtimeProvider:
        """按类签名自适应实例化（不同 provider 的 __init__ 签名不同）。"""
        import inspect

        sig = inspect.signature(klass.__init__)
        params = sig.parameters
        kwargs: dict[str, Any] = {}
        if "credentials" in params:
            kwargs["credentials"] = cred
        if "settings" in params:
            kwargs["settings"] = None  # provider 内部按需解析
        return klass(**kwargs)

    def create_or_mock(self, name: str | None = None) -> BaseRealtimeProvider:
        """创建 provider；未配置时优雅回退 mock（保证零依赖可跑）。"""
        target = name or self._settings.default
        try:
            return self.create(target)
        except (ConfigError, ProviderUnauthorized):
            logger.warning("provider '%s' 不可用，回退 mock", target)
            return self.create("mock")

    # ---------------- 内部 ----------------

    def _resolve(self, name: str) -> Type[BaseRealtimeProvider]:
        klass = self._classes.get(name)
        if klass is None:
            raise NotFoundError(f"未知 provider: {name}，可用: {self.names()}")
        return klass

    def _is_configured(self, name: str) -> bool:
        if name == "mock":
            return True
        if name == "local_pipeline":
            return True  # 本地组合式管线：由本地模型配置决定，宽松处理
        entry = self._settings.entry(name)
        if entry is None:
            return False
        return bool(entry.configured)

    def _credential_for(self, name: str) -> ProviderCredential:
        """从 ProviderEntry 构建凭据对象。"""
        from omnistream.schemas.provider import ProviderCredential

        entry = self._settings.entry(name)
        if entry is None:
            return ProviderCredential(provider=name)
        return ProviderCredential(
            provider=name,
            api_key=entry.api_key,
            api_secret=getattr(entry, "api_secret", None),
            base_url=entry.base_url or getattr(entry, "ws_url", None),
        )


registry: ProviderRegistry | None = None


def get_registry(settings: ProvidersSettings | None = None) -> ProviderRegistry:
    """进程级便捷入口。"""
    global registry
    if registry is None:
        registry = ProviderRegistry.get_instance(settings)
    return registry


def reset_registry() -> None:
    """测试/热重载用。"""
    global registry
    registry = None

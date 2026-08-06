"""omnistream - enterprise-grade realtime full-duplex omni-modal interaction platform.

The top level package is intentionally lightweight: importing ``omnistream`` must not
pull in FastAPI, SQLAlchemy, aiortc or any other heavy optional dependency. Convenience
symbols are resolved lazily through :pep:`562` module ``__getattr__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omnistream.__version__ import __version__, __version_info__

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnistream.config.settings import Settings
    from omnistream.core.error_codes import ErrorCode
    from omnistream.core.errors import OmniError

__all__ = [
    "ErrorCode",
    "OmniError",
    "Settings",
    "__version__",
    "__version_info__",
    "create_app",
    "get_settings",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "OmniError": ("omnistream.core.errors", "OmniError"),
    "ErrorCode": ("omnistream.core.error_codes", "ErrorCode"),
    "Settings": ("omnistream.config.settings", "Settings"),
    "get_settings": ("omnistream.config.settings", "get_settings"),
    "create_app": ("omnistream.api.app", "create_app"),
}


def __getattr__(name: str) -> Any:
    """Resolve heavy top level symbols on first access."""
    try:
        module_path, attr = _LAZY_EXPORTS[name]
    except KeyError:  # pragma: no cover - defensive
        raise AttributeError(f"module 'omnistream' has no attribute {name!r}") from None
    import importlib

    module = importlib.import_module(module_path)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)

"""泛型注册表：provider / tool / asr 引擎 / tts 引擎 / vad 后端共用。

用法::

    PROVIDERS: Registry[type[BaseRealtimeProvider]] = Registry("provider")

    @PROVIDERS.register("mock")
    class MockRealtimeProvider(BaseRealtimeProvider):
        ...

    cls = PROVIDERS.get("mock")
"""
from __future__ import annotations

from collections.abc import Callable, ItemsView, Iterator
from typing import Generic, TypeVar

from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import ConflictError, NotFoundError
from omnistream.core.logging import get_logger

__all__ = ["Registry"]

log = get_logger(__name__)

T = TypeVar("T")


class Registry(Generic[T]):
    """按名字注册与发现对象的容器。

    * 注册名规范：小写下划线（见架构文档 §7.8）；
    * 重复注册默认抛 :class:`ConflictError`，显式 ``override=True`` 可覆盖；
    * 支持别名，便于兼容旧配置值。
    """

    def __init__(self, kind: str, *, not_found_code: ErrorCode = ErrorCode.NOT_FOUND) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}
        self._aliases: dict[str, str] = {}
        self._not_found_code = not_found_code

    # ------------------------------------------------------------- 注册
    def register(self, name: str, item: T | None = None, *, override: bool = False,
                 aliases: tuple[str, ...] = ()) -> Callable[[T], T] | T:
        """注册一个对象。既可直接调用也可作装饰器。

        ::

            registry.register("mock", MockProvider)

            @registry.register("mock")
            class MockProvider: ...
        """
        key = self._normalize(name)

        def _do(value: T) -> T:
            if key in self._items and not override:
                raise ConflictError(
                    f"{self.kind} '{key}' 已注册，如需覆盖请传 override=True",
                    code=ErrorCode.CONFLICT,
                    detail={"kind": self.kind, "name": key},
                )
            self._items[key] = value
            for alias in aliases:
                self._aliases[self._normalize(alias)] = key
            log.debug("registry_registered", kind=self.kind, name=key, aliases=list(aliases))
            return value

        if item is not None:
            return _do(item)
        return _do

    def unregister(self, name: str) -> None:
        """移除注册项及其全部别名。不存在时静默返回。"""
        key = self._normalize(name)
        self._items.pop(key, None)
        for alias, target in list(self._aliases.items()):
            if target == key:
                self._aliases.pop(alias, None)

    def alias(self, alias_name: str, target: str) -> None:
        """为已注册项增加一个别名。"""
        key = self._normalize(target)
        if key not in self._items:
            raise NotFoundError(
                f"{self.kind} '{key}' 未注册，无法为其建立别名",
                code=self._not_found_code,
                detail={"kind": self.kind, "name": key},
            )
        self._aliases[self._normalize(alias_name)] = key

    # ------------------------------------------------------------- 查询
    def get(self, name: str) -> T:
        """取出注册项，不存在抛 :class:`NotFoundError`。"""
        key = self._resolve(name)
        try:
            return self._items[key]
        except KeyError:
            raise NotFoundError(
                f"未找到 {self.kind}: '{name}'，可用值: {', '.join(self.names()) or '(空)'}",
                code=self._not_found_code,
                detail={"kind": self.kind, "name": name, "available": self.names()},
            ) from None

    def get_or(self, name: str, default: T) -> T:
        """取出注册项，不存在返回 ``default``。"""
        key = self._resolve(name)
        return self._items.get(key, default)

    def has(self, name: str) -> bool:
        return self._resolve(name) in self._items

    def names(self) -> list[str]:
        """全部注册名（不含别名），按字典序。"""
        return sorted(self._items)

    def items(self) -> ItemsView[str, T]:
        return self._items.items()

    def clear(self) -> None:
        self._items.clear()
        self._aliases.clear()

    # ------------------------------------------------------------- dunder
    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.has(name)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __getitem__(self, name: str) -> T:
        return self.get(name)

    def __repr__(self) -> str:
        return f"<Registry kind={self.kind} size={len(self._items)}>"

    # ------------------------------------------------------------- 内部
    @staticmethod
    def _normalize(name: str) -> str:
        return name.strip().lower().replace("-", "_")

    def _resolve(self, name: str) -> str:
        key = self._normalize(name)
        return self._aliases.get(key, key)

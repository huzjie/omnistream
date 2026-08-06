"""路由包（空包标记）。

**并行开发约定**：各工作包把自己的路由文件直接放进本目录，例如
``sessions.py`` / ``realtime_ws.py`` / ``memory.py``。每个模块只需导出：

* ``router: APIRouter`` —— 必需；
* ``ROUTER_ORDER: int`` —— 可选，默认 100，数值小的先挂载；
* ``ROUTER_TAGS: list[str]`` —— 可选，OpenAPI 分组标签。

:func:`omnistream.api.registry.mount_routers` 会在应用启动时自动扫描并挂载，
**不需要修改任何公共文件**，因此多人同时新增路由不会产生冲突。
某个模块导入失败（缺可选依赖）只会记一条 warning，不会拖垮整个服务。
"""
from __future__ import annotations

__all__: list[str] = []

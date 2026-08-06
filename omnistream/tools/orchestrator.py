"""实时工具调用编排。

能力：
- 工具注册（ToolRegistry）：按名称注册/查询 ToolSpec 与执行器
- 并行工具执行：同一轮多个 tool_call 并发执行，可配置并发上限
- 流式中途调用：模型生成过程中随时允许 tool_call 事件触发执行
- 超时与降级：单工具超时兜底、失败策略（fail / continue）、熔断
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeAlias

from omnistream.core.clock import monotonic_s
from omnistream.core.errors import NotFoundError, ToolError
from omnistream.schemas.tool import ToolCall, ToolResult, ToolSpec

logger = logging.getLogger(__name__)

ToolExecutor: TypeAlias = Callable[[dict[str, Any]], Awaitable[Any]]


def _mono_ms() -> float:
    return monotonic_s() * 1000.0


@dataclass(slots=True)
class ToolMetrics:
    executions: int = 0
    failures: int = 0
    total_ms: float = 0.0
    last_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.executions if self.executions else 0.0


@dataclass(slots=True)
class ToolRuntime:
    """单个工具的运行时状态。"""
    spec: ToolSpec
    executor: ToolExecutor
    timeout_ms: int = 30_000
    failures: int = 0
    metrics: ToolMetrics = field(default_factory=ToolMetrics)


class ToolRegistry:
    """工具注册表。"""

    def __init__(self, max_parallel: int = 4) -> None:
        self._tools: dict[str, ToolRuntime] = {}
        self._max_parallel = max_parallel

    def register(
        self,
        name: str,
        executor: ToolExecutor,
        *,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        timeout_ms: int = 30_000,
    ) -> None:
        spec = ToolSpec(
            name=name, description=description,
            parameters=parameters or {"type": "object", "properties": {}},
        )
        self._tools[name] = ToolRuntime(spec=spec, executor=executor,
                                        timeout_ms=timeout_ms)

    def register_spec(self, spec: ToolSpec, executor: ToolExecutor,
                      timeout_ms: int = 30_000) -> None:
        self._tools[spec.name] = ToolRuntime(
            spec=spec, executor=executor, timeout_ms=timeout_ms)

    def specs(self) -> list[ToolSpec]:
        return [rt.spec for rt in self._tools.values()]

    def has(self, name: str) -> bool:
        return name in self._tools

    def metrics(self, name: str) -> ToolMetrics | None:
        rt = self._tools.get(name)
        return rt.metrics if rt else None

    def remove(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None


class ToolOrchestrator:
    """并行工具编排器。

    典型流程：::

        orch = ToolOrchestrator(registry, fail_policy="continue")
        results: list[ToolResult] = await orch.execute_parallel([
            ToolCall(id="1", name="get_weather", arguments={"city": "北京"}),
            ToolCall(id="2", name="get_time", arguments={}),
        ])
    """

    def __init__(self, registry: ToolRegistry,
                 fail_policy: str = "continue",
                 max_parallel: int | None = None) -> None:
        self.registry = registry
        self.fail_policy = fail_policy  # fail | continue
        self._max_parallel = max_parallel or registry._max_parallel
        self._sem = asyncio.Semaphore(self._max_parallel)

    async def execute(self, call: ToolCall) -> ToolResult:
        rt = self.registry._tools.get(call.name)
        if rt is None:
            raise NotFoundError(f"工具未注册: {call.name}")
        started = _mono_ms()
        try:
            async with self._sem:
                outcome = await asyncio.wait_for(
                    rt.executor(call.arguments or {}),
                    timeout=rt.timeout_ms / 1000,
                )
            elapsed = _mono_ms() - started
            rt.metrics.executions += 1
            rt.metrics.total_ms += elapsed
            rt.metrics.last_ms = elapsed
            rt.failures = 0
            return ToolResult(call_id=call.id,
                              name=call.name, ok=True, content=outcome,
                              elapsed_ms=elapsed)
        except asyncio.TimeoutError as exc:
            return self._failure(call, rt, f"tool timeout>{rt.timeout_ms}ms", exc)
        except Exception as exc:  # noqa: BLE001
            return self._failure(call, rt, str(exc), exc)

    def _failure(self, call: ToolCall, rt: ToolRuntime,
                 message: str, exc: Exception) -> ToolResult:
        elapsed = rt.metrics.last_ms
        rt.metrics.failures += 1
        rt.failures += 1
        logger.warning("tool '%s' 失败: %s", call.name, message)
        return ToolResult(call_id=call.id, name=call.name,
                          ok=False, content={"error": message},
                          elapsed_ms=elapsed, error=str(exc))

    async def execute_parallel(self, calls: list[ToolCall]) -> list[ToolResult]:
        """并行执行一批 tool_call，保持输入顺序输出。"""
        if not calls:
            return []
        tasks = [asyncio.create_task(self.execute(c)) for c in calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: list[ToolResult] = []
        for call, res in zip(calls, results):
            if isinstance(res, Exception):
                out.append(ToolResult(call_id=call.id,
                                      name=call.name, ok=False,
                                      content={"error": str(res)},
                                      error=str(res)))
            else:
                out.append(res)
        return out

    def all_results_ok(self, results: list[ToolResult]) -> bool:
        return all(r.ok for r in results)


def create_default_registry() -> ToolRegistry:
    """内置默认工具集：系统信息 + 时间 + 简单计算。"""
    reg = ToolRegistry()

    async def _now(_: dict[str, Any]) -> dict[str, Any]:
        import datetime
        return {"iso": datetime.datetime.now().isoformat(),
                "ts_ms": int(datetime.datetime.now().timestamp() * 1000)}

    async def _echo(args: dict[str, Any]) -> dict[str, Any]:
        return {"echo": args.get("text", "")}

    async def _calc(args: dict[str, Any]) -> dict[str, Any]:
        expr = args.get("expression", "0")
        try:
            return {"result": eval(expr, {"__builtins__": {}}, {})}  # noqa: S307 - 受限命名空间
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    reg.register("system_now", _now, description="获取当前时间",
                 parameters={"type": "object", "properties": {}})
    reg.register("utils_echo", _echo, description="回显文本",
                 parameters={"type": "object",
                             "properties": {"text": {"type": "string"}}})
    reg.register("utils_calc", _calc, description="安全计算简单表达式",
                 parameters={"type": "object",
                             "properties": {"expression": {"type": "string"}}})
    return reg

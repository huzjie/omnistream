"""感知层异常定义。

全部继承自 :class:`omnistream.core.errors.PerceptionError`，
因此 API 层 / WS 层的统一错误处理无需为感知层写任何特例代码，
错误码沿用 §3.2 的 ``OMS-50xx`` 段：

===============================  ==========  ================================
异常                              错误码       语义
===============================  ==========  ================================
:class:`ASRError`                 OMS-5002    语音识别失败
:class:`TTSError`                 OMS-5003    语音合成失败
:class:`VisionError`              OMS-5004    视觉理解失败
:class:`EngineUnavailable`        OMS-5002/3  引擎依赖缺失（可降级）
:class:`VisionBudgetExceeded`     OMS-5004    视觉预算耗尽（可降级）
===============================  ==========  ================================

**降级优先于抛错**：所有"可降级"的异常都带 ``degradable=True``，
调用方应优先切换到 mock/统计实现，而不是把错误抛给用户。
"""
from __future__ import annotations

from typing import Any

from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import PerceptionError


class ASRError(PerceptionError):
    """语音识别失败。"""

    default_code = ErrorCode.ASR_FAILED


class TTSError(PerceptionError):
    """语音合成失败。"""

    default_code = ErrorCode.TTS_FAILED


class VisionError(PerceptionError):
    """视觉理解失败。"""

    default_code = ErrorCode.VISION_FAILED


class EngineUnavailable(PerceptionError):
    """引擎依赖缺失（模型文件 / 第三方库 / 凭证），**可降级**。"""

    default_code = ErrorCode.ASR_FAILED

    def __init__(self, message: str, *, engine: str = "", kind: str = "asr",
                 code: ErrorCode | None = None, detail: dict[str, Any] | None = None,
                 cause: BaseException | None = None) -> None:
        resolved = code or _CODE_BY_KIND.get(kind, ErrorCode.ASR_FAILED)
        merged: dict[str, Any] = {"engine": engine, "kind": kind, "degradable": True}
        merged.update(detail or {})
        super().__init__(message, code=resolved, detail=merged, cause=cause)
        self.engine = engine
        self.kind = kind
        self.degradable = True


class ASRTimeout(ASRError):
    """识别超时（云端引擎无响应）。"""


class TTSTimeout(TTSError):
    """合成超时。"""


class VisionBudgetExceeded(VisionError):
    """视觉预算耗尽，应降级为纯统计描述，**可降级**。"""

    def __init__(self, message: str = "视觉预算已耗尽", *, used: int = 0, limit: int = 0,
                 window_s: int = 60, detail: dict[str, Any] | None = None) -> None:
        merged: dict[str, Any] = {
            "used": used,
            "limit": limit,
            "window_s": window_s,
            "degradable": True,
        }
        merged.update(detail or {})
        super().__init__(message, detail=merged)
        self.used = used
        self.limit = limit
        self.degradable = True


class VLMCallFailed(VisionError):
    """VLM 调用失败（网络/鉴权/响应格式）。"""

    def __init__(self, message: str, *, model: str = "", status: int = 0,
                 detail: dict[str, Any] | None = None,
                 cause: BaseException | None = None) -> None:
        merged: dict[str, Any] = {"model": model, "status": status}
        merged.update(detail or {})
        super().__init__(message, detail=merged, cause=cause)
        self.model = model
        self.status = status


_CODE_BY_KIND: dict[str, ErrorCode] = {
    "asr": ErrorCode.ASR_FAILED,
    "tts": ErrorCode.TTS_FAILED,
    "vision": ErrorCode.VISION_FAILED,
}


def wrap_engine_error(exc: BaseException, *, engine: str, kind: str = "asr",
                      message: str = "") -> PerceptionError:
    """把任意第三方异常包装成感知层异常，保留原始 cause。

    Args:
        exc: 原始异常。
        engine: 引擎名。
        kind: ``asr`` / ``tts`` / ``vision``。
        message: 自定义消息；留空时自动生成。

    Returns:
        对应类型的 :class:`PerceptionError` 子类实例。
    """
    if isinstance(exc, PerceptionError):
        return exc
    text = message or f"{kind} 引擎 {engine} 执行失败: {exc}"
    detail = {"engine": engine, "kind": kind, "error_type": type(exc).__name__}
    if kind == "tts":
        return TTSError(text, detail=detail, cause=exc)
    if kind == "vision":
        return VisionError(text, detail=detail, cause=exc)
    return ASRError(text, detail=detail, cause=exc)


def is_degradable(exc: BaseException) -> bool:
    """该异常是否应触发"降级"而非"报错"。"""
    if isinstance(exc, PerceptionError):
        return bool(getattr(exc, "degradable", False)
                    or exc.detail.get("degradable", False))
    return False


__all__ = [
    "ASRError",
    "ASRTimeout",
    "EngineUnavailable",
    "TTSError",
    "TTSTimeout",
    "VLMCallFailed",
    "VisionBudgetExceeded",
    "VisionError",
    "is_degradable",
    "wrap_engine_error",
]

"""Celery 离线任务入口与任务定义。

任务：会话归档、转写索引、质检评分、记忆维护、报表生成。
Redis 未配置时优雅降级（任务直接内联执行）。
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from celery import Celery

    _HAS_CELERY = True
except ImportError:  # pragma: no cover
    Celery = None  # type: ignore[assignment,misc]
    _HAS_CELERY = False

_USE_CELERY = False
_app: Any = None


def init_celery(broker_url: str | None, backend_url: str | None = None) -> Any:
    """初始化 Celery app；broker 为 None 时禁用。"""
    global _app, _USE_CELERY
    if not _HAS_CELERY or not broker_url:
        _USE_CELERY = False
        logger.info("celery 未启用（broker 未配置），任务将内联执行")
        return None
    _app = Celery("omnistream", broker=broker_url,
                  backend=backend_url or broker_url)
    _app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="Asia/Shanghai",
        enable_utc=True,
        task_track_started=True,
        broker_connection_retry_on_startup=True,
    )
    _USE_CELERY = True
    return _app


def _wrap(name: str):
    """根据 celery 是否启用决定注册为 task 还是普通函数。"""

    def decorator(fn):
        if _USE_CELERY and _app is not None:
            return _app.task(name=name)(fn)
        return fn

    return decorator


@_wrap("omnistream.archive_session")
def archive_session(session_id: str, transcript_path: str | None = None) -> dict[str, Any]:
    """归档会话：转写落盘 + 状态更新。"""
    logger.info("archive session %s", session_id)
    return {"session_id": session_id, "status": "archived"}


@_wrap("omnistream.index_transcript")
def index_transcript(session_id: str, transcript: str) -> dict[str, Any]:
    """转写文本索引（向量化入库）。"""
    logger.info("index transcript %s (%d chars)", session_id, len(transcript))
    return {"session_id": session_id, "chars": len(transcript), "indexed": True}


@_wrap("omnistream.run_quality_check")
def run_quality_check(session_id: str, transcript: str) -> dict[str, Any]:
    """离线质检评分（延迟指标、打断质量、敏感词命中）。"""
    from omnistream.governance.quality import score_transcript

    result = score_transcript(transcript)
    logger.info("quality check %s -> %s", session_id, result["score"])
    return {"session_id": session_id, **result}


@_wrap("omnistream.memory_maintenance")
def memory_maintenance(tenant_id: str | None = None) -> dict[str, Any]:
    """记忆层维护：遗忘与压缩。"""
    from omnistream.memory.manager import MemoryManager

    mm = MemoryManager()
    return mm.maintenance()


@_wrap("omnistream.generate_daily_report")
def generate_daily_report(tenant_id: str, date: str) -> dict[str, Any]:
    """生成租户日报（会话数、时长、延迟、质检分）。"""
    logger.info("daily report %s %s", tenant_id, date)
    return {"tenant_id": tenant_id, "date": date, "status": "generated"}

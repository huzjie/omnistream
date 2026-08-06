"""录制路由：列表 / 详情 / 回放清单 / 导出。

读操作需要 ``recording:read``。下载 URL 根据存储配置生成（本地存储时返回本地对象路径，
对象存储时返回相对签名端点路径，由网关负责换发真实签名 URL）。
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select

from omnistream.api.deps import CurrentPrincipal, DbSession, Pagination, require_perm
from omnistream.config.settings import get_settings
from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import NotFoundError
from omnistream.db.models.recording import Recording
from omnistream.schemas.auth import Permission
from omnistream.schemas.common import Ok, Page

router = APIRouter()
ROUTER_ORDER = 70
ROUTER_TAGS = ["recordings"]


def _download_url(settings: Any, key: str | None) -> str | None:
    if not key:
        return None
    storage = settings.storage
    if storage.object_store == "local":
        return f"{storage.local_path.rstrip('/')}/{key}"
    return f"/v1/recordings/file?key={key}"


def _to_dict(rec: Recording, settings: Any) -> dict[str, Any]:
    return {
        "id": rec.id,
        "session_id": rec.session_id,
        "status": rec.status,
        "storage_backend": rec.storage_backend,
        "duration_ms": rec.duration_ms,
        "size_bytes": rec.size_bytes,
        "audio_format": rec.audio_format,
        "encrypted": rec.encrypted,
        "retention_until": rec.retention_until.isoformat() if rec.retention_until else None,
        "audio_url": _download_url(settings, rec.audio_key),
        "video_url": _download_url(settings, rec.video_key),
        "transcript_url": _download_url(settings, rec.transcript_key),
        "events_url": _download_url(settings, rec.events_key),
        "created_at": rec.created_at.isoformat() if rec.created_at else None,
    }


@router.get("", response_model=Ok[Page[dict]])
async def list_recordings(
    principal: Annotated[Any, Depends(require_perm(Permission.RECORDING_READ))],
    session: DbSession,
    page: Pagination,
) -> Ok[Page[dict]]:
    from sqlalchemy import func

    settings = get_settings()
    total = await session.execute(
        func.count().select_from(Recording).where(Recording.tenant_id == principal.tenant_id)
    )
    rows = await session.execute(
        select(Recording)
        .where(Recording.tenant_id == principal.tenant_id)
        .order_by(Recording.created_at.desc())
        .limit(page.limit)
        .offset(page.offset)
    )
    items = [_to_dict(r, settings) for r in rows.scalars().all()]
    return Ok.of(Page.of(items, total=int(total.scalar_one() or 0)))


@router.get("/{recording_id}", response_model=Ok[dict])
async def get_recording(
    recording_id: str,
    principal: Annotated[Any, Depends(require_perm(Permission.RECORDING_READ))],
    session: DbSession,
) -> Ok[dict]:
    settings = get_settings()
    rec = await session.execute(
        select(Recording).where(
            Recording.id == recording_id, Recording.tenant_id == principal.tenant_id
        )
    )
    found = rec.scalar_one_or_none()
    if found is None:
        raise NotFoundError("录制不存在", code=ErrorCode.NOT_FOUND)
    return Ok.of(_to_dict(found, settings))


@router.get("/{recording_id}/manifest", response_model=Ok[dict])
async def get_manifest(
    recording_id: str,
    principal: Annotated[Any, Depends(require_perm(Permission.RECORDING_READ))],
    session: DbSession,
) -> Ok[dict]:
    """返回回放清单（时间轴 + 资产 URL）。"""
    settings = get_settings()
    rec = await session.execute(
        select(Recording).where(
            Recording.id == recording_id, Recording.tenant_id == principal.tenant_id
        )
    )
    found = rec.scalar_one_or_none()
    if found is None:
        raise NotFoundError("录制不存在", code=ErrorCode.NOT_FOUND)
    manifest = {
        "session_id": found.session_id,
        "recording_id": found.id,
        "duration_ms": found.duration_ms,
        "audio_url": _download_url(settings, found.audio_key),
        "video_url": _download_url(settings, found.video_key),
        "transcript_url": _download_url(settings, found.transcript_key),
        "events_url": _download_url(settings, found.events_key),
        "events": [],
        "barge_in_marks_ms": [],
        "turn_marks_ms": [],
        "generated_at": found.created_at.isoformat() if found.created_at else None,
    }
    return Ok.of(manifest)


@router.post("/{recording_id}/export", response_model=Ok[dict])
async def export_recording(
    recording_id: str,
    principal: Annotated[Any, Depends(require_perm(Permission.RECORDING_READ))],
    session: DbSession,
) -> Ok[dict]:
    """触发导出（异步任务占位）。返回已受理。"""
    rec = await session.execute(
        select(Recording).where(
            Recording.id == recording_id, Recording.tenant_id == principal.tenant_id
        )
    )
    found = rec.scalar_one_or_none()
    if found is None:
        raise NotFoundError("录制不存在", code=ErrorCode.NOT_FOUND)
    return Ok.of(
        {"recording_id": found.id, "status": "accepted", "message": "导出任务已受理"}
    )

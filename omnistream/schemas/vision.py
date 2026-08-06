"""视觉理解领域模型：实体、场景描述、观察事件、视频记忆节点。"""
from __future__ import annotations

from datetime import datetime

from pydantic import Field

from omnistream.core.clock import utcnow
from omnistream.schemas.base import OmniModel


class BoundingBox(OmniModel):
    """归一化包围盒，取值 [0,1]，原点在左上角。"""

    x: float = Field(default=0.0, ge=0.0, le=1.0)
    y: float = Field(default=0.0, ge=0.0, le=1.0)
    w: float = Field(default=0.0, ge=0.0, le=1.0)
    h: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


class VisualEntity(OmniModel):
    """画面中的一个可指代对象。"""

    label: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    bbox: BoundingBox | None = None
    attributes: dict[str, str] = Field(default_factory=dict)
    #: 跨帧跟踪 ID，同一物体在连续帧中保持一致
    track_id: str | None = None
    #: 是否为用户当前指向/注视的对象（"这个是什么" 的所指）
    salient: bool = False


class SceneDescription(OmniModel):
    """VLM 对单帧（或一小段）画面的结构化理解。"""

    summary: str = Field(default="", description="一句话场景描述，直接进对话上下文")
    entities: list[VisualEntity] = Field(default_factory=list)
    ocr_text: str = ""
    scene_tags: list[str] = Field(default_factory=list)
    people_count: int = 0
    #: 相对上一次描述的变化程度，用于决定是否值得主动开口
    change_score: float = Field(default=0.0, ge=0.0, le=1.0)
    engine: str = "mock"
    latency_ms: float = 0.0

    @property
    def salient_entities(self) -> list[VisualEntity]:
        return [e for e in self.entities if e.salient]

    def to_prompt(self, max_entities: int = 6) -> str:
        """渲染成注入模型上下文的一段文本。"""
        parts: list[str] = []
        if self.summary:
            parts.append(self.summary)
        if self.entities:
            names = ", ".join(e.label for e in self.entities[:max_entities])
            parts.append(f"画面中可见：{names}")
        if self.ocr_text:
            snippet = self.ocr_text[:120]
            parts.append(f"画面文字：{snippet}")
        return "；".join(parts)


class VisionObservation(OmniModel):
    """一次落库/下发的视觉观察事件。"""

    id: str = ""
    session_id: str = ""
    pts_ms: int = 0
    description: SceneDescription = Field(default_factory=SceneDescription)
    frame_key: str | None = Field(default=None, description="对象存储中的原帧 key")
    keyframe_score: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utcnow)
    #: 是否已作为上下文注入过模型
    injected: bool = False

    @property
    def summary(self) -> str:
        return self.description.summary


class VideoMemoryNode(OmniModel):
    """视频记忆图谱节点：把一段时间内语义连续的观察压成一个"场景"。

    用于长会话中回答"刚才我给你看的那个东西是什么颜色"这类跨时间引用。
    """

    id: str = ""
    session_id: str = ""
    start_pts_ms: int = 0
    end_pts_ms: int = 0
    title: str = ""
    summary: str = ""
    entities: list[VisualEntity] = Field(default_factory=list)
    observation_ids: list[str] = Field(default_factory=list)
    thumbnail_key: str | None = None
    embedding: list[float] | None = Field(default=None, repr=False)
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_pts_ms - self.start_pts_ms)

    def covers(self, pts_ms: int) -> bool:
        return self.start_pts_ms <= pts_ms <= self.end_pts_ms

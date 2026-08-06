"""视觉理解引擎抽象接口。

冻结契约（§2.21）::

    class BaseVisionEngine:
        async def describe(frame, context) -> SceneDescription

设计要点
--------
1. **视觉是"旁路"而不是"主路"**。全双工对话的主路是音频，
   视觉理解慢一点没关系，但**绝不能阻塞语音链路**。
   因此所有实现都必须是可取消、有超时、超预算就放弃的。
2. **一次描述一帧**。跨帧的语义（"他刚才把杯子拿走了"）由
   :mod:`~omnistream.perception.vision.change_narrator` 在描述之上做差分，
   引擎本身保持无状态，方便并发与缓存。
3. **描述结果直接进上下文**。:class:`SceneDescription` 的
   :meth:`~omnistream.schemas.vision.SceneDescription.to_prompt`
   决定了模型看到什么，所以 ``summary`` 必须是**可直接朗读的自然语言**，
   不能是 ``{"objects": [...]}`` 这种给机器看的结构。
4. **失败必须降级而不是抛异常上浮**。视觉挂了，对话要能继续。
"""
from __future__ import annotations

import abc
import asyncio
import base64
import time
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import structlog

from omnistream.core.ids import new_id
from omnistream.media.video.frame import RawVideoFrame, encode_image
from omnistream.perception.errors import VisionError
from omnistream.protocol.enums import VideoEncoding
from omnistream.schemas.vision import (
    BoundingBox,
    SceneDescription,
    VisionObservation,
    VisualEntity,
)

log = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_VISION_TIMEOUT_S",
    "BaseVisionEngine",
    "VisionCapabilities",
    "VisionRequest",
    "VisionUnavailable",
    "build_entity",
    "build_observation",
    "frame_to_data_url",
    "frame_to_jpeg",
]

#: 单帧描述的默认超时（秒）。超时即放弃，不重试——画面已经过时了。
DEFAULT_VISION_TIMEOUT_S: Final[float] = 8.0

#: 送进 VLM 的图像长边上限（像素）。再大只是烧 token。
DEFAULT_MAX_SIDE: Final[int] = 768

#: 送进 VLM 的 JPEG 质量
DEFAULT_JPEG_QUALITY: Final[int] = 72


class VisionUnavailable(RuntimeError):
    """引擎依赖缺失时由构造函数抛出，供工厂降级捕获。"""


@dataclass(slots=True)
class VisionCapabilities:
    """引擎能力声明，路由与预算据此决策。"""

    #: 能否给出自然语言场景描述
    captioning: bool = True
    #: 能否给出带包围盒的实体
    detection: bool = False
    #: 能否读画面里的文字
    ocr: bool = False
    #: 能否回答关于画面的自由提问（VQA）
    vqa: bool = False
    #: 能否一次吃多帧（视频理解）
    multi_frame: bool = False
    needs_gpu: bool = False
    needs_network: bool = False
    #: 典型单帧延迟（毫秒）
    typical_latency_ms: int = 1200
    #: 单帧相对成本（0=免费本地，1=最贵云端）
    cost: float = 0.0
    #: 支持的最大输入边长
    max_side: int = DEFAULT_MAX_SIDE


@dataclass(slots=True)
class VisionRequest:
    """一次视觉理解请求。"""

    frame: RawVideoFrame
    #: 会话 ID，用于日志与落库
    session_id: str = ""
    #: 用户的具体问题（"这个是什么颜色"），为空则做通用描述
    question: str = ""
    #: 系统提示词覆盖
    system_prompt: str = ""
    #: 上一次的描述，供引擎做增量叙述
    prior: SceneDescription | None = None
    #: 对话上下文摘要，帮助 VLM 理解"这个"指什么
    context: str = ""
    max_tokens: int = 256
    temperature: float = 0.2
    detect_entities: bool = True
    ocr_enabled: bool = True
    timeout_s: float = DEFAULT_VISION_TIMEOUT_S
    #: 语言标签，决定描述用什么语言输出
    language: str = "zh-CN"

    @property
    def pts_ms(self) -> int:
        """该帧的呈现时间戳。"""
        return int(self.frame.pts_ms)

    def cache_key(self) -> str:
        """基于画面内容与问题的缓存键。

        用降采样后的像素哈希，容忍轻微噪声但能区分真实变化。
        """
        import hashlib

        gray = self.frame.to_gray()
        step = max(1, min(gray.shape) // 16)
        thumb = gray[::step, ::step]
        # 量化到 16 级，压掉传感器噪声
        quantized = (np.asarray(thumb, dtype=np.uint8) >> 4).tobytes()
        digest = hashlib.blake2b(quantized, digest_size=12).hexdigest()
        return f"{digest}|{self.question}|{self.language}"


def frame_to_jpeg(
    frame: RawVideoFrame,
    *,
    max_side: int = DEFAULT_MAX_SIDE,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> bytes:
    """把帧降采样并编码成 JPEG 字节。

    Args:
        frame: 原始帧。
        max_side: 长边上限，超过则等比缩小。
        quality: JPEG 质量。

    Returns:
        JPEG 字节；Pillow 缺失时返回原始 RGB 字节（调用方需自行判断）。
    """
    pixels = frame.pixels
    long_side = max(pixels.shape[0], pixels.shape[1])
    if long_side > max_side > 0:
        scale = max_side / long_side
        new_h = max(1, int(pixels.shape[0] * scale))
        new_w = max(1, int(pixels.shape[1] * scale))
        rows = (np.arange(new_h) * (pixels.shape[0] / new_h)).astype(np.int32)
        cols = (np.arange(new_w) * (pixels.shape[1] / new_w)).astype(np.int32)
        pixels = pixels[np.ix_(rows, cols)]
    try:
        return encode_image(pixels, VideoEncoding.JPEG, quality=quality)
    except Exception as exc:  # noqa: BLE001 - Pillow 缺失时退回裸 RGB
        log.debug("vision.jpeg_encode_failed", error=str(exc))
        return encode_image(pixels, VideoEncoding.RGB24)


def frame_to_data_url(
    frame: RawVideoFrame,
    *,
    max_side: int = DEFAULT_MAX_SIDE,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> str:
    """把帧编码成 ``data:image/jpeg;base64,...``，可直接塞进 VLM 请求。"""
    payload = frame_to_jpeg(frame, max_side=max_side, quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")


def build_entity(
    label: str,
    *,
    confidence: float = 0.0,
    bbox: tuple[float, float, float, float] | None = None,
    attributes: dict[str, str] | None = None,
    track_id: str | None = None,
    salient: bool = False,
) -> VisualEntity:
    """构造一个 :class:`VisualEntity`，包围盒自动夹到 [0,1]。

    Args:
        label: 实体名称。
        confidence: 置信度。
        bbox: ``(x, y, w, h)`` 归一化包围盒。
        attributes: 属性字典（颜色、状态等）。
        track_id: 跨帧跟踪 ID。
        salient: 是否为用户所指对象。

    Returns:
        实体对象。
    """
    box: BoundingBox | None = None
    if bbox is not None:
        x, y, w, h = bbox
        x = max(0.0, min(1.0, float(x)))
        y = max(0.0, min(1.0, float(y)))
        w = max(0.0, min(1.0 - x, float(w)))
        h = max(0.0, min(1.0 - y, float(h)))
        box = BoundingBox(x=x, y=y, w=w, h=h)
    return VisualEntity(
        label=label,
        confidence=max(0.0, min(1.0, float(confidence))),
        bbox=box,
        attributes=dict(attributes or {}),
        track_id=track_id,
        salient=bool(salient),
    )


def build_observation(
    description: SceneDescription,
    *,
    session_id: str = "",
    pts_ms: int = 0,
    frame_key: str | None = None,
    keyframe_score: float = 0.0,
    observation_id: str = "",
) -> VisionObservation:
    """把描述包装成可落库/可下发的观察事件。

    Args:
        description: 场景描述。
        session_id: 会话 ID。
        pts_ms: 帧时间戳。
        frame_key: 对象存储中的原帧 key。
        keyframe_score: 关键帧得分。
        observation_id: 显式 ID，为空则自动生成。

    Returns:
        观察事件。
    """
    return VisionObservation(
        id=observation_id or new_id("vob"),
        session_id=session_id,
        pts_ms=int(pts_ms),
        description=description,
        frame_key=frame_key,
        keyframe_score=max(0.0, min(1.0, float(keyframe_score))),
    )


class BaseVisionEngine(abc.ABC):
    """视觉理解引擎抽象基类。"""

    #: 注册名，用于 ``create_vision("vlm")``
    name: str = "base"

    #: 能力声明，子类覆盖
    capabilities: VisionCapabilities = VisionCapabilities()

    def __init__(
        self,
        *,
        language: str = "zh-CN",
        max_tokens: int = 256,
        temperature: float = 0.2,
        timeout_s: float = DEFAULT_VISION_TIMEOUT_S,
        concurrency: int = 2,
        session_id: str = "",
    ) -> None:
        """初始化。

        Args:
            language: 描述输出语言。
            max_tokens: 单次描述的最大长度。
            temperature: 采样温度。
            timeout_s: 单帧超时。
            concurrency: 并发上限，避免把 GPU/额度打满。
            session_id: 会话 ID，仅用于日志。
        """
        self.language = language
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.timeout_s = max(0.1, float(timeout_s))
        self.session_id = session_id
        self._semaphore = asyncio.Semaphore(max(1, int(concurrency)))
        self._closed = False

        self.frames_described = 0
        self.frames_failed = 0
        self.total_latency_ms = 0.0
        self.last_latency_ms = 0.0

    # ------------------------------------------------------------------
    # 子类实现
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def _describe(self, request: VisionRequest) -> SceneDescription:
        """真正干活的方法，由子类实现。

        实现约定：

        * 不要自己做超时/并发控制，基类已经包好了；
        * 失败时抛 :class:`~omnistream.perception.errors.VisionError`
          或其子类，基类会转成降级描述；
        * ``summary`` 必须是自然语言，且用 ``request.language`` 指定的语言。
        """

    # ------------------------------------------------------------------
    # 通用能力
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """引擎依赖是否就绪。"""
        return True

    @property
    def closed(self) -> bool:
        """是否已关闭。"""
        return self._closed

    @property
    def avg_latency_ms(self) -> float:
        """平均单帧延迟。"""
        return (self.total_latency_ms / self.frames_described
                if self.frames_described else 0.0)

    async def describe(self, request: VisionRequest) -> SceneDescription:
        """描述一帧画面（带并发控制、超时与降级）。

        Args:
            request: 视觉请求。

        Returns:
            场景描述；失败时返回一个 ``summary`` 为空、``engine`` 标注失败原因
            的降级描述，**不会抛异常**。
        """
        if self._closed:
            return self._degraded("engine_closed", 0.0)

        started = time.monotonic()
        try:
            async with self._semaphore:
                if self._closed:
                    return self._degraded("engine_closed", 0.0)
                async with asyncio.timeout(request.timeout_s or self.timeout_s):
                    description = await self._describe(request)
        except TimeoutError:
            self.frames_failed += 1
            elapsed = (time.monotonic() - started) * 1000.0
            log.warning("vision.describe_timeout", engine=self.name,
                        pts_ms=request.pts_ms, elapsed_ms=round(elapsed, 1))
            return self._degraded("timeout", elapsed)
        except asyncio.CancelledError:
            raise
        except VisionError as exc:
            self.frames_failed += 1
            elapsed = (time.monotonic() - started) * 1000.0
            log.warning("vision.describe_failed", engine=self.name, error=str(exc))
            return self._degraded(f"error:{type(exc).__name__}", elapsed)
        except Exception as exc:  # noqa: BLE001 - 视觉失败绝不能打断对话
            self.frames_failed += 1
            elapsed = (time.monotonic() - started) * 1000.0
            log.warning("vision.describe_crashed", engine=self.name, error=str(exc))
            return self._degraded(f"crash:{type(exc).__name__}", elapsed)

        elapsed = (time.monotonic() - started) * 1000.0
        self.frames_described += 1
        self.total_latency_ms += elapsed
        self.last_latency_ms = elapsed
        description.engine = description.engine or self.name
        description.latency_ms = round(elapsed, 2)
        return description

    async def describe_batch(
        self, requests: list[VisionRequest]
    ) -> list[SceneDescription]:
        """并发描述多帧，顺序与输入一致。

        Args:
            requests: 请求列表。

        Returns:
            描述列表，长度与输入相同。
        """
        if not requests:
            return []
        results = await asyncio.gather(
            *(self.describe(r) for r in requests), return_exceptions=True
        )
        output: list[SceneDescription] = []
        for item in results:
            if isinstance(item, SceneDescription):
                output.append(item)
            else:
                output.append(self._degraded("batch_error", 0.0))
        return output

    async def aclose(self) -> None:
        """释放资源，幂等。"""
        self._closed = True

    def _degraded(self, reason: str, latency_ms: float) -> SceneDescription:
        """构造一个降级描述：内容为空但结构合法，下游可安全消费。"""
        return SceneDescription(
            summary="",
            entities=[],
            ocr_text="",
            scene_tags=["degraded"],
            people_count=0,
            change_score=0.0,
            engine=f"{self.name}:{reason}",
            latency_ms=round(latency_ms, 2),
        )

    def stats(self) -> dict[str, Any]:
        """统计快照。"""
        return {
            "engine": self.name,
            "described": self.frames_described,
            "failed": self.frames_failed,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "last_latency_ms": round(self.last_latency_ms, 1),
            "closed": self._closed,
        }

    def __repr__(self) -> str:
        return (f"<{type(self).__name__} name={self.name} "
                f"described={self.frames_described}>")


@dataclass(slots=True)
class VisionEngineState:
    """引擎的可变运行状态（供路由与健康检查读取）。"""

    engine: str = ""
    healthy: bool = True
    consecutive_failures: int = 0
    last_error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

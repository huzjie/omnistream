"""speexdsp AEC 可选实现（服务端混音场景）。

依赖 ``speexdsp-python``（``pip install speexdsp``）。缺失时构造函数抛
:class:`~omnistream.duplex.aec.base.AECUnavailable`，工厂会自动降级到
:class:`~omnistream.duplex.aec.passthrough.PassthroughAEC`。

speexdsp 的 MDF 自适应滤波器要求：

* 输入必须是 **16-bit 小端 PCM 裸字节**；
* 麦克风帧与参考帧**长度必须完全一致**且等于创建时的 ``frame_size``；
* 参考帧必须是"即将/正在播放"的信号，因此需要
  :class:`~omnistream.duplex.aec.reference.ReferenceSignal` 做延迟对齐。

滤波器 tail length 默认取 200ms，覆盖典型房间混响 + 播放缓冲；
tail 越长收敛越慢、CPU 越高，可通过 ``filter_length_ms`` 调整。

即便原生库存在，本实现也做了完整兜底：任何一帧处理异常都会退化为直通，
只累加 ``stats.errors``，绝不让音频主链路中断。
"""
from __future__ import annotations

from typing import Any, Final

import numpy as np
import structlog

from omnistream.duplex.aec.base import (
    DEFAULT_FRAME_MS,
    DEFAULT_SAMPLE_RATE,
    AECUnavailable,
    BaseAEC,
)
from omnistream.duplex.aec.reference import ReferenceSignal
from omnistream.schemas.audio import AudioChunk

log = structlog.get_logger(__name__)

try:  # pragma: no cover - 取决于运行环境是否安装了 speexdsp
    from speexdsp import EchoCanceller as _EchoCanceller  # type: ignore[import-not-found]

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 - 任何导入失败都视为不可用
    _EchoCanceller = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc

#: 默认滤波器长度（毫秒），即可消除的最大回声拖尾。
DEFAULT_FILTER_LENGTH_MS: Final[int] = 200


def is_available() -> bool:
    """speexdsp 是否可用。"""
    return _EchoCanceller is not None


def import_error() -> Exception | None:
    """返回导入失败的原始异常（可用时为 ``None``）。"""
    return _IMPORT_ERROR


class SpeexAEC(BaseAEC):
    """基于 speexdsp MDF 的回声消除。"""

    name = "speex"

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_ms: int = DEFAULT_FRAME_MS,
        channels: int = 1,
        filter_length_ms: int = DEFAULT_FILTER_LENGTH_MS,
        track_reference: bool = True,
        initial_delay_ms: int = 120,
    ) -> None:
        super().__init__(sample_rate=sample_rate, frame_ms=frame_ms, channels=channels)
        if _EchoCanceller is None:
            raise AECUnavailable(
                f"speexdsp 未安装，无法使用 SpeexAEC：{_IMPORT_ERROR}"
            )

        self.filter_length_ms = int(filter_length_ms)
        self.frame_size = self.frame_samples
        self.filter_length = max(
            self.frame_size,
            int(self.sample_rate * self.filter_length_ms / 1000),
        )
        self.reference = (
            ReferenceSignal(sample_rate=self.sample_rate, initial_delay_ms=initial_delay_ms)
            if track_reference else None
        )
        self._canceller: Any = self._create_canceller()
        self._degraded = False

    # ------------------------------------------------------------------
    # 构造原生对象
    # ------------------------------------------------------------------

    def _create_canceller(self) -> Any:
        """兼容 speexdsp-python 的两种构造风格。"""
        assert _EchoCanceller is not None
        factory = getattr(_EchoCanceller, "create", None)
        try:
            if callable(factory):
                return factory(self.frame_size, self.filter_length, self.sample_rate)
            return _EchoCanceller(self.frame_size, self.filter_length, self.sample_rate)
        except Exception as exc:  # noqa: BLE001
            raise AECUnavailable(f"speexdsp EchoCanceller 创建失败: {exc}") from exc

    # ------------------------------------------------------------------
    # 参考信号
    # ------------------------------------------------------------------

    def push_reference(self, chunk: AudioChunk) -> None:
        super().push_reference(chunk)
        if self.reference is not None:
            self.reference.push(chunk)

    # ------------------------------------------------------------------
    # 主处理
    # ------------------------------------------------------------------

    def process(self, mic: AudioChunk, ref: AudioChunk | None = None) -> AudioChunk:
        """执行回声消除；任何异常都退化为直通。"""
        self._ensure_pcm(mic)
        mic_samples = self.to_mono_float(mic)
        n = mic_samples.size
        if n == 0:
            return mic

        ref_samples = self._resolve_reference(n, ref)
        if ref_samples.size == 0 or not np.any(ref_samples):
            # 没有参考信号时消除毫无意义，直通即可（也避免滤波器被静音带偏）。
            self._account(mic_samples, mic_samples)
            return mic

        mic_frame, ref_frame = self._align(mic_samples, ref_samples)

        try:
            out_bytes = self._canceller.process(
                _to_pcm16_bytes(mic_frame), _to_pcm16_bytes(ref_frame)
            )
            cleaned = _from_pcm16_bytes(out_bytes)
        except Exception as exc:  # noqa: BLE001 - 单帧失败不得中断链路
            self.stats.errors += 1
            if not self._degraded:
                self._degraded = True
                log.warning("aec.speex_frame_failed", error=str(exc),
                            frame_size=self.frame_size)
            self._account(mic_samples, mic_samples)
            return mic

        if cleaned.size < n:
            cleaned = np.pad(cleaned, (0, n - cleaned.size))
        cleaned = cleaned[:n]
        self._account(mic_samples, cleaned)
        return self._rebuild(mic, cleaned)

    def _align(self, mic: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """把两路信号裁剪/补零到 ``frame_size``。"""
        size = self.frame_size
        mic_frame = _fit(mic, size)
        ref_frame = _fit(ref, size)
        return mic_frame, ref_frame

    def _resolve_reference(self, n: int, ref: AudioChunk | None) -> np.ndarray:
        if ref is not None:
            return self.to_mono_float(ref)
        if self.reference is None:
            return np.zeros(0, dtype=np.float32)
        samples = self.reference.take(n)
        if not np.any(samples):
            self.stats.reference_underruns += 1
        return samples

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        return is_available()

    def reset(self) -> None:
        super().reset()
        self._degraded = False
        if self.reference is not None:
            self.reference.reset()
        try:
            self._canceller = self._create_canceller()
        except AECUnavailable as exc:
            log.warning("aec.speex_reset_failed", error=str(exc))

    def close(self) -> None:
        closer = getattr(self._canceller, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception as exc:  # noqa: BLE001
                log.debug("aec.speex_close_failed", error=str(exc))
        self._canceller = None


def _fit(samples: np.ndarray, size: int) -> np.ndarray:
    """裁剪或补零到指定长度。"""
    if samples.size == size:
        return samples
    if samples.size > size:
        return samples[:size]
    return np.pad(samples, (0, size - samples.size))


def _to_pcm16_bytes(samples: np.ndarray) -> bytes:
    """float32 [-1,1] → 16-bit 小端 PCM 字节。"""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _from_pcm16_bytes(data: bytes) -> np.ndarray:
    """16-bit 小端 PCM 字节 → float32 [-1,1]。"""
    if not data:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


__all__ = [
    "DEFAULT_FILTER_LENGTH_MS",
    "SpeexAEC",
    "import_error",
    "is_available",
]

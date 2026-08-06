"""流式重采样器：soxr 优先，audioop 次选，纯 numpy 兜底。

三级后端
--------
1. ``soxr``（VHQ 质量、SIMD 加速、原生流式 ``ResampleStream``）—— 首选。
2. ``audioop.ratecv``（CPython 标准库，Python < 3.13 可用，自带状态）—— 次选。
3. 纯 numpy 多相插值：Hamming 窗 sinc 抗混叠低通 + 线性插值，**跨块保持相位连续** —— 永远可用。

三种后端都是**流式**的：内部保留滤波器状态与小数相位，连续调用 :meth:`Resampler.process`
不会在块边界产生咔哒声。这是实时链路（20ms 一帧）的硬要求。
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import structlog

from omnistream.media import metrics
from omnistream.media.audio import pcm
from omnistream.media.errors import ResampleError
from omnistream.protocol.enums import AudioEncoding
from omnistream.schemas.audio import AudioChunk

log = structlog.get_logger(__name__)

__all__ = ["AUDIOOP_AVAILABLE", "Resampler", "SOXR_AVAILABLE", "available_backends", "resample_once"]

try:  # pragma: no cover - 环境相关
    import soxr

    SOXR_AVAILABLE = True
except ImportError:  # pragma: no cover
    soxr = None  # type: ignore[assignment]
    SOXR_AVAILABLE = False

try:  # pragma: no cover - Python 3.13 起被移除
    import audioop  # type: ignore[import-not-found]

    AUDIOOP_AVAILABLE = True
except ImportError:  # pragma: no cover
    audioop = None  # type: ignore[assignment]
    AUDIOOP_AVAILABLE = False


def available_backends() -> list[str]:
    """返回当前环境可用的后端名，按优先级排序。"""
    backends = []
    if SOXR_AVAILABLE:
        backends.append("soxr")
    if AUDIOOP_AVAILABLE:
        backends.append("audioop")
    backends.append("numpy")
    return backends


# --------------------------------------------------------------------------
# 纯 numpy 后端
# --------------------------------------------------------------------------


def _design_lowpass(cutoff_ratio: float, num_taps: int = 63) -> np.ndarray:
    """设计 Hamming 窗 sinc 低通 FIR。

    Args:
        cutoff_ratio: 归一化截止频率（相对采样率，0~0.5）。
        num_taps: 抽头数，取奇数保证线性相位且群延迟为整数。

    Returns:
        归一化后的一维 ``float64`` 系数。
    """
    if num_taps % 2 == 0:
        num_taps += 1
    n = np.arange(num_taps, dtype=np.float64) - (num_taps - 1) / 2.0
    cutoff = float(np.clip(cutoff_ratio, 1e-4, 0.499))
    taps = 2.0 * cutoff * np.sinc(2.0 * cutoff * n)
    taps *= np.hamming(num_taps)
    total = float(np.sum(taps))
    if abs(total) > 1e-9:
        taps /= total
    return taps


class _NumpyResampler:
    """纯 numpy 的流式有理重采样。

    算法：
        1. 降采样时先过 Hamming 窗 sinc 低通（截止 = 0.45 * 目标奈奎斯特），抑制混叠。
        2. 以 ``step = src / dst`` 的小数步长在输入上线性插值取样。
        3. 跨块保留 ``_tail``（未消费的样本）与 ``_pos``（小数相位），保证连续。
    """

    __slots__ = ("_channels", "_dst", "_fir_state", "_pos", "_src", "_step", "_tail", "_taps")

    def __init__(self, src_rate: int, dst_rate: int, channels: int) -> None:
        self._src = float(src_rate)
        self._dst = float(dst_rate)
        self._channels = max(1, channels)
        self._step = self._src / self._dst
        ratio = self._dst / self._src
        if ratio < 1.0:
            # 降采样：截止频率取目标奈奎斯特的 90%，留过渡带
            self._taps: np.ndarray | None = _design_lowpass(0.45 * ratio, 63)
            self._fir_state = np.zeros((len(self._taps) - 1, self._channels), dtype=np.float64)
        else:
            self._taps = None
            self._fir_state = np.zeros((0, self._channels), dtype=np.float64)
        self._tail = np.zeros((1, self._channels), dtype=np.float64)
        self._pos = 0.0

    def reset(self) -> None:
        """清空内部状态（会话重置 / seek 时调用）。"""
        if self._taps is not None:
            self._fir_state = np.zeros((len(self._taps) - 1, self._channels), dtype=np.float64)
        self._tail = np.zeros((1, self._channels), dtype=np.float64)
        self._pos = 0.0

    def _filter(self, planar: np.ndarray) -> np.ndarray:
        assert self._taps is not None
        buf = np.concatenate((self._fir_state, planar), axis=0)
        out = np.empty((planar.shape[0], self._channels), dtype=np.float64)
        for ch in range(self._channels):
            out[:, ch] = np.convolve(buf[:, ch], self._taps, mode="valid")
        keep = len(self._taps) - 1
        self._fir_state = buf[buf.shape[0] - keep :] if keep else buf[0:0]
        return out

    def process(self, planar: np.ndarray) -> np.ndarray:
        """处理 ``(frames, channels)`` 的 float64 输入，返回同布局输出。"""
        if planar.shape[0] == 0:
            return np.zeros((0, self._channels), dtype=np.float64)
        if self._taps is not None:
            planar = self._filter(planar)
        buf = np.concatenate((self._tail, planar), axis=0)
        max_index = buf.shape[0] - 1
        if max_index < 1:
            self._tail = buf
            return np.zeros((0, self._channels), dtype=np.float64)
        n_out = int(np.floor((max_index - self._pos) / self._step)) + 1
        if n_out <= 0:
            self._tail = buf
            return np.zeros((0, self._channels), dtype=np.float64)
        idx = self._pos + self._step * np.arange(n_out, dtype=np.float64)
        grid = np.arange(buf.shape[0], dtype=np.float64)
        out = np.empty((n_out, self._channels), dtype=np.float64)
        for ch in range(self._channels):
            out[:, ch] = np.interp(idx, grid, buf[:, ch])
        next_pos = idx[-1] + self._step
        keep_from = int(np.floor(next_pos))
        keep_from = max(0, min(keep_from, buf.shape[0] - 1))
        self._tail = buf[keep_from:]
        self._pos = next_pos - keep_from
        return out


# --------------------------------------------------------------------------
# 统一入口
# --------------------------------------------------------------------------


class Resampler:
    """流式采样率转换器。

    Example::

        rs = Resampler(48000, 16000, channels=1)
        for pcm16_bytes in stream:
            out = rs.process(pcm16_bytes)     # bytes -> bytes，pcm16 小端

    线程/协程安全性：**非线程安全**，每条流独占一个实例。
    """

    def __init__(
        self,
        src_rate: int,
        dst_rate: int,
        *,
        channels: int = 1,
        quality: str = "VHQ",
        backend: str | None = None,
    ) -> None:
        """
        Args:
            src_rate: 源采样率。
            dst_rate: 目标采样率。
            channels: 声道数（interleaved）。
            quality: soxr 质量档（``QQ``/``LQ``/``MQ``/``HQ``/``VHQ``）。
            backend: 强制指定后端（``soxr``/``audioop``/``numpy``），``None`` 表示自动。

        Raises:
            ResampleError: 采样率或声道数非法。
        """
        if src_rate <= 0 or dst_rate <= 0:
            raise ResampleError(
                "sample rates must be positive", detail={"src": src_rate, "dst": dst_rate}
            )
        if channels < 1:
            raise ResampleError("channels must be >= 1", detail={"channels": channels})
        self.src_rate = int(src_rate)
        self.dst_rate = int(dst_rate)
        self.channels = int(channels)
        self.quality = quality
        self._passthrough = self.src_rate == self.dst_rate
        self._impl: Any = None
        self._audioop_state: Any = None
        self._numpy: _NumpyResampler | None = None
        self.backend = "passthrough" if self._passthrough else (backend or self._pick_backend())
        if not self._passthrough:
            self._init_backend()
        log.debug(
            "resampler_created",
            src_rate=self.src_rate,
            dst_rate=self.dst_rate,
            channels=self.channels,
            backend=self.backend,
        )

    # ---------------- 构造 ----------------

    def _pick_backend(self) -> str:
        if SOXR_AVAILABLE:
            return "soxr"
        if AUDIOOP_AVAILABLE and self.channels in (1, 2):
            return "audioop"
        return "numpy"

    def _init_backend(self) -> None:
        if self.backend == "soxr":
            if not SOXR_AVAILABLE:
                self.backend = "numpy"
            else:
                try:
                    self._impl = soxr.ResampleStream(  # type: ignore[union-attr]
                        self.src_rate,
                        self.dst_rate,
                        self.channels,
                        dtype="float32",
                        quality=self.quality,
                    )
                except Exception as exc:  # soxr 参数不被支持时降级
                    log.warning("soxr_init_failed", error=str(exc))
                    self.backend = "numpy"
        if self.backend == "audioop" and not (AUDIOOP_AVAILABLE and self.channels in (1, 2)):
            self.backend = "numpy"
        if self.backend == "numpy":
            self._numpy = _NumpyResampler(self.src_rate, self.dst_rate, self.channels)

    # ---------------- 处理 ----------------

    def reset(self) -> None:
        """重置内部相位与滤波状态。"""
        self._audioop_state = None
        if self.backend == "soxr" and SOXR_AVAILABLE:
            try:
                self._impl = soxr.ResampleStream(  # type: ignore[union-attr]
                    self.src_rate, self.dst_rate, self.channels,
                    dtype="float32", quality=self.quality,
                )
            except Exception:
                self.backend = "numpy"
                self._numpy = _NumpyResampler(self.src_rate, self.dst_rate, self.channels)
        elif self._numpy is not None:
            self._numpy.reset()

    def process(self, data: bytes, *, last: bool = False) -> bytes:
        """转换一段 pcm16 小端字节。

        Args:
            data: 输入 pcm16 字节（interleaved）。
            last: 是否为流的最后一块（soxr 需要据此冲刷内部延迟）。

        Returns:
            目标采样率的 pcm16 字节。
        """
        if not data:
            return b"" if not last else self.flush()
        if self._passthrough:
            return data
        started = time.perf_counter()
        try:
            if self.backend == "audioop":
                out = self._process_audioop(data)
            elif self.backend == "soxr":
                out = self._process_soxr(data, last=last)
            else:
                out = self._process_numpy(data)
        except ResampleError:
            raise
        except Exception as exc:
            raise ResampleError(
                f"resample failed on backend {self.backend}: {exc}",
                detail={"backend": self.backend, "src": self.src_rate, "dst": self.dst_rate},
                cause=exc,
            ) from exc
        metrics.observe_ms(metrics.resample_ms, (time.perf_counter() - started) * 1000.0,
                           backend=self.backend)
        return out

    def process_array(self, arr: np.ndarray) -> np.ndarray:
        """``float32`` interleaved 数组进 / 出（避免多余的字节往返）。"""
        data = pcm.float32_to_bytes(arr, AudioEncoding.PCM16)
        return pcm.bytes_to_float32(self.process(data), AudioEncoding.PCM16)

    def process_chunk(self, chunk: AudioChunk) -> AudioChunk:
        """转换一个 :class:`AudioChunk`，pts 与采集时刻原样保留。"""
        if chunk.encoding != AudioEncoding.PCM16:
            raise ResampleError(
                f"resampler only accepts pcm16, got {chunk.encoding}",
                detail={"encoding": str(chunk.encoding)},
            )
        return AudioChunk(
            pts_ms=chunk.pts_ms,
            data=self.process(chunk.data),
            sample_rate=self.dst_rate,
            channels=self.channels,
            encoding=AudioEncoding.PCM16,
            capture_ts_ns=chunk.capture_ts_ns,
            speaker=chunk.speaker,
        )

    def flush(self) -> bytes:
        """冲刷内部残留（流结束时调用）。"""
        if self._passthrough:
            return b""
        if self.backend == "soxr" and self._impl is not None:
            try:
                tail = self._impl.resample_chunk(
                    np.zeros((0, self.channels), dtype=np.float32), last=True
                )
                return pcm.float32_to_bytes(np.asarray(tail).reshape(-1), AudioEncoding.PCM16)
            except Exception:
                return b""
        return b""

    # ---------------- 各后端实现 ----------------

    def _process_audioop(self, data: bytes) -> bytes:
        converted, self._audioop_state = audioop.ratecv(  # type: ignore[union-attr]
            data, 2, self.channels, self.src_rate, self.dst_rate, self._audioop_state
        )
        return converted

    def _process_soxr(self, data: bytes, *, last: bool) -> bytes:
        flat = pcm.bytes_to_float32(data, AudioEncoding.PCM16)
        planar = flat.reshape(-1, self.channels) if self.channels > 1 else flat.reshape(-1, 1)
        out = self._impl.resample_chunk(planar.astype(np.float32), last=last)
        return pcm.float32_to_bytes(np.asarray(out, dtype=np.float32).reshape(-1),
                                    AudioEncoding.PCM16)

    def _process_numpy(self, data: bytes) -> bytes:
        assert self._numpy is not None
        flat = pcm.bytes_to_float32(data, AudioEncoding.PCM16).astype(np.float64)
        planar = flat.reshape(-1, self.channels) if self.channels > 1 else flat.reshape(-1, 1)
        out = self._numpy.process(planar)
        return pcm.float32_to_bytes(out.reshape(-1).astype(np.float32), AudioEncoding.PCM16)

    def __repr__(self) -> str:
        return (
            f"<Resampler {self.src_rate}->{self.dst_rate} ch={self.channels} "
            f"backend={self.backend}>"
        )


def resample_once(
    data: bytes, src_rate: int, dst_rate: int, *, channels: int = 1
) -> bytes:
    """一次性重采样（非流式场景：加载 WAV、生成示例音频）。"""
    if src_rate == dst_rate:
        return data
    rs = Resampler(src_rate, dst_rate, channels=channels)
    return rs.process(data, last=True) + rs.flush()

"""参考信号管理：把 TTS 输出对齐延迟后作为回声参考。

服务端拿到的"播放信号"和"麦克风信号"之间存在一段未知延迟
（网络 + 客户端播放缓冲 + 声学传播 + 采集缓冲），通常在 60~300ms。
AEC 自适应滤波器只能覆盖有限的 tail length，所以必须先做**粗对齐**，
再把对齐后的参考帧喂给滤波器。

本模块提供：

* :class:`ReferenceSignal` —— 有界环形缓冲 + 读写指针 + 欠载补零；
* 基于 FFT 归一化互相关的 :meth:`ReferenceSignal.estimate_delay_ms`，
  在双讲（double-talk）较少的窗口上自动校准延迟；
* 纯 numpy 实现，零额外依赖。
"""
from __future__ import annotations

import math
from typing import Final

import numpy as np
import structlog

from omnistream.duplex.aec.base import DEFAULT_SAMPLE_RATE, EPS
from omnistream.schemas.audio import AudioChunk

log = structlog.get_logger(__name__)

#: 环形缓冲默认容量（毫秒）。要能同时容纳最大延迟与一次相关分析窗口。
DEFAULT_CAPACITY_MS: Final[int] = 4000

#: 默认初始延迟猜测（毫秒）。浏览器 WebRTC 链路的典型值。
DEFAULT_INITIAL_DELAY_MS: Final[int] = 120

#: 允许估计的最大延迟（毫秒）。
DEFAULT_MAX_DELAY_MS: Final[int] = 400

#: 互相关分析窗口（毫秒）。
DEFAULT_ANALYSIS_MS: Final[int] = 320

#: 相关峰值低于该阈值时认为估计不可信（多半正处于双讲或静音）。
MIN_CORRELATION: Final[float] = 0.28


class ReferenceSignal:
    """播放参考信号的环形缓冲与延迟对齐器。"""

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        capacity_ms: int = DEFAULT_CAPACITY_MS,
        initial_delay_ms: int = DEFAULT_INITIAL_DELAY_MS,
        max_delay_ms: int = DEFAULT_MAX_DELAY_MS,
        auto_align: bool = True,
        analysis_ms: int = DEFAULT_ANALYSIS_MS,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.capacity = max(1, int(self.sample_rate * capacity_ms / 1000))
        self.max_delay_ms = int(max_delay_ms)
        self.auto_align = bool(auto_align)
        self.analysis_samples = max(64, int(self.sample_rate * analysis_ms / 1000))

        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._write_total = 0          # 已写入的累计样本数
        self._read_total = -1          # 已读出的累计样本数；-1 表示尚未初始化
        self._delay_ms = float(initial_delay_ms)
        self._last_estimate_ms: float | None = None
        self._estimate_confidence = 0.0
        self._underruns = 0
        self._resyncs = 0

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def delay_ms(self) -> float:
        """当前使用的对齐延迟（毫秒）。"""
        return self._delay_ms

    @property
    def delay_samples(self) -> int:
        return int(self._delay_ms * self.sample_rate / 1000.0)

    @property
    def buffered_ms(self) -> float:
        """缓冲中尚未被消费的参考时长（毫秒）。"""
        if self._read_total < 0:
            return min(self._write_total, self.capacity) * 1000.0 / self.sample_rate
        return max(0, self._write_total - self._read_total) * 1000.0 / self.sample_rate

    @property
    def underruns(self) -> int:
        return self._underruns

    @property
    def confidence(self) -> float:
        """最近一次延迟估计的归一化相关峰值 [0,1]。"""
        return self._estimate_confidence

    def set_delay_ms(self, value: float) -> None:
        """手动设置延迟（例如从客户端上报的 playoutDelay 得到）。"""
        self._delay_ms = float(min(max(0.0, value), float(self.max_delay_ms)))
        self._read_total = -1  # 强制下次 take 重新定位

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def push(self, chunk: AudioChunk) -> None:
        """写入一帧播放音频（会自动转单声道 float32）。"""
        arr = chunk.to_ndarray()
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        samples = np.asarray(arr, dtype=np.float32).reshape(-1)
        if chunk.sample_rate != self.sample_rate and samples.size > 0:
            samples = _linear_resample(samples, chunk.sample_rate, self.sample_rate)
        self.push_array(samples)

    def push_array(self, samples: np.ndarray) -> None:
        """写入一段裸样本（一维 float32，范围 [-1,1]）。"""
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        n = data.size
        if n == 0:
            return
        if n >= self.capacity:
            data = data[-self.capacity:]
            n = data.size
        start = self._write_total % self.capacity
        end = start + n
        if end <= self.capacity:
            self._buf[start:end] = data
        else:
            head = self.capacity - start
            self._buf[start:] = data[:head]
            self._buf[: n - head] = data[head:]
        self._write_total += n

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def take(self, num_samples: int) -> np.ndarray:
        """取出与当前麦克风帧对齐的参考帧；不足部分补零。"""
        n = int(num_samples)
        if n <= 0:
            return np.zeros(0, dtype=np.float32)

        if self._read_total < 0:
            self._read_total = max(0, self._write_total - self.delay_samples - n)

        # 漂移保护：读指针落后太多（说明消费侧卡过），直接重新定位。
        lag = self._write_total - self._read_total
        if lag > self.delay_samples + self.capacity // 2:
            self._read_total = max(0, self._write_total - self.delay_samples - n)
            self._resyncs += 1

        out = self._read_at(self._read_total, n)
        available = self._write_total - self._read_total
        if available < n:
            self._underruns += 1
            missing = n - max(0, available)
            if missing > 0:
                out[max(0, available):] = 0.0
        self._read_total += n
        return out

    def _read_at(self, absolute_index: int, n: int) -> np.ndarray:
        """从累计索引处读取 n 个样本（越界部分返回 0）。"""
        out = np.zeros(n, dtype=np.float32)
        oldest = max(0, self._write_total - self.capacity)
        start = max(int(absolute_index), oldest)
        end = min(int(absolute_index) + n, self._write_total)
        if end <= start:
            return out
        idx = np.arange(start, end, dtype=np.int64) % self.capacity
        out[start - int(absolute_index): end - int(absolute_index)] = self._buf[idx]
        return out

    def recent(self, num_samples: int) -> np.ndarray:
        """取最近写入的 n 个样本（用于互相关分析）。"""
        n = min(int(num_samples), self.capacity, self._write_total)
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        return self._read_at(self._write_total - n, n)

    # ------------------------------------------------------------------
    # 延迟估计
    # ------------------------------------------------------------------

    def estimate_delay_ms(self, mic: np.ndarray) -> float | None:
        """用归一化互相关估计麦克风相对参考的延迟。

        Args:
            mic: 最近一段麦克风样本（一维 float32）。

        Returns:
            估计延迟（毫秒）；不可信时返回 ``None``。
        """
        mic_arr = np.asarray(mic, dtype=np.float32).reshape(-1)
        if mic_arr.size < 64:
            return None

        span = self.analysis_samples + int(self.max_delay_ms * self.sample_rate / 1000)
        ref = self.recent(span)
        if ref.size < mic_arr.size + 8:
            return None

        seg = mic_arr[-self.analysis_samples:] if mic_arr.size > self.analysis_samples else mic_arr
        seg = seg - float(np.mean(seg))
        ref = ref - float(np.mean(ref))
        seg_norm = float(np.linalg.norm(seg))
        if seg_norm < 1e-4:
            return None

        n_fft = 1 << int(math.ceil(math.log2(ref.size + seg.size)))
        fft_ref = np.fft.rfft(ref, n=n_fft)
        fft_seg = np.fft.rfft(seg[::-1], n=n_fft)
        corr = np.fft.irfft(fft_ref * fft_seg, n=n_fft)[: ref.size]

        # 每个偏移位置的参考能量，用于归一化
        window = np.ones(seg.size, dtype=np.float64)
        energy = np.convolve(np.square(ref.astype(np.float64)), window, mode="valid")
        usable = min(corr.size - seg.size + 1, energy.size)
        if usable <= 1:
            return None
        denom = np.sqrt(energy[:usable]) * seg_norm + EPS
        scores = corr[seg.size - 1: seg.size - 1 + usable] / denom

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        self._estimate_confidence = max(0.0, min(1.0, best_score))
        if best_score < MIN_CORRELATION:
            return None

        # best_idx 是"参考窗口内的起点"，末尾对应最新样本
        lag_samples = usable - 1 - best_idx
        delay_ms = lag_samples * 1000.0 / self.sample_rate
        if delay_ms < 0 or delay_ms > self.max_delay_ms:
            return None
        self._last_estimate_ms = delay_ms
        return delay_ms

    def observe(self, mic: np.ndarray, *, smoothing: float = 0.2) -> float:
        """周期性校准延迟；返回校准后的延迟（毫秒）。"""
        if not self.auto_align:
            return self._delay_ms
        est = self.estimate_delay_ms(mic)
        if est is None:
            return self._delay_ms
        old = self._delay_ms
        self._delay_ms = (1.0 - smoothing) * old + smoothing * est
        if abs(self._delay_ms - old) > 20.0:
            self._read_total = -1  # 大幅变化时重新定位读指针
            log.info("aec.reference_realigned", old_ms=round(old, 1),
                     new_ms=round(self._delay_ms, 1), confidence=round(self.confidence, 3))
        return self._delay_ms

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """清空缓冲（换会话时调用），保留延迟估计。"""
        self._buf.fill(0.0)
        self._write_total = 0
        self._read_total = -1
        self._underruns = 0
        self._resyncs = 0

    def stats(self) -> dict[str, float]:
        return {
            "delay_ms": round(self._delay_ms, 2),
            "buffered_ms": round(self.buffered_ms, 2),
            "underruns": float(self._underruns),
            "resyncs": float(self._resyncs),
            "confidence": round(self._estimate_confidence, 3),
            "last_estimate_ms": round(self._last_estimate_ms, 2)
            if self._last_estimate_ms is not None else -1.0,
        }

    def __repr__(self) -> str:
        return (f"<ReferenceSignal delay={self._delay_ms:.0f}ms "
                f"buffered={self.buffered_ms:.0f}ms underruns={self._underruns}>")


def _linear_resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """线性插值重采样。参考信号对相位精度要求不高，够用且零依赖。"""
    if src_rate == dst_rate or samples.size == 0:
        return samples
    dst_len = max(1, int(round(samples.size * dst_rate / float(src_rate))))
    src_idx = np.linspace(0.0, samples.size - 1, num=dst_len, dtype=np.float64)
    return np.interp(src_idx, np.arange(samples.size, dtype=np.float64),
                     samples.astype(np.float64)).astype(np.float32)


__all__ = [
    "DEFAULT_ANALYSIS_MS",
    "DEFAULT_CAPACITY_MS",
    "DEFAULT_INITIAL_DELAY_MS",
    "DEFAULT_MAX_DELAY_MS",
    "MIN_CORRELATION",
    "ReferenceSignal",
]

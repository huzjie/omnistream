"""定长 int16 音频环形缓冲。

音频入口的原则是**永不丢帧**（丢音频=丢语义），但内存不能无界增长，因此用固定
容量的环形缓冲承接：写入超过容量时覆盖最旧数据并累加 ``overwrites`` 计数，由
观测层告警。读取支持"消费式读取"与"回看式窥视"（barge-in 需要回看前缀音频）。
"""
from __future__ import annotations

import threading

import numpy as np

from omnistream.core.constants import (
    BYTES_PER_PCM16_SAMPLE,
    DEFAULT_INPUT_SAMPLE_RATE,
    RING_BUFFER_SECONDS,
)

__all__ = ["AudioRingBuffer"]


class AudioRingBuffer:
    """线程安全的 int16 单声道环形缓冲。

    ::

        rb = AudioRingBuffer(sample_rate=16000, capacity_s=30)
        rb.write_bytes(pcm16_frame)
        frame = rb.read_bytes(320)      # 消费 20ms
        prefix = rb.peek_last_ms(300)   # 回看最近 300ms（不消费）
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_INPUT_SAMPLE_RATE,
        capacity_s: float = RING_BUFFER_SECONDS,
        *,
        channels: int = 1,
        name: str = "audio_ring",
    ) -> None:
        if sample_rate <= 0:
            raise ValueError(f"非法采样率: {sample_rate}")
        if channels < 1:
            raise ValueError(f"非法声道数: {channels}")
        self.name = name
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.capacity = max(1, int(sample_rate * capacity_s)) * self.channels
        self._buf = np.zeros(self.capacity, dtype=np.int16)
        self._write = 0
        self._available = 0
        self._lock = threading.Lock()
        self.overwrites = 0
        self.total_written = 0
        self.total_read = 0

    # ------------------------------------------------------------- 属性
    @property
    def available(self) -> int:
        """当前可读取的样本数（跨声道计数）。"""
        with self._lock:
            return self._available

    @property
    def available_ms(self) -> float:
        """当前可读取的时长（毫秒）。"""
        return self.available * 1000.0 / (self.sample_rate * self.channels)

    @property
    def capacity_ms(self) -> float:
        return self.capacity * 1000.0 / (self.sample_rate * self.channels)

    @property
    def fill_ratio(self) -> float:
        """填充率 0.0~1.0，超过 0.8 应触发背压告警。"""
        return self.available / self.capacity if self.capacity else 0.0

    @property
    def is_empty(self) -> bool:
        return self.available == 0

    # ------------------------------------------------------------- 写
    def write(self, samples: np.ndarray) -> int:
        """写入 int16 样本数组，返回被覆盖（丢弃）的旧样本数。"""
        data = np.ascontiguousarray(samples.reshape(-1), dtype=np.int16)
        n = int(data.size)
        if n == 0:
            return 0
        with self._lock:
            if n >= self.capacity:
                # 单次写入超过总容量：只保留最后 capacity 个样本。
                self._buf[:] = data[-self.capacity:]
                dropped = n - self.capacity + self._available
                self._write = 0
                self._available = self.capacity
                self.overwrites += dropped
                self.total_written += n
                return int(dropped)

            end = self._write + n
            if end <= self.capacity:
                self._buf[self._write:end] = data
            else:
                head = self.capacity - self._write
                self._buf[self._write:] = data[:head]
                self._buf[: n - head] = data[head:]
            self._write = end % self.capacity

            dropped = 0
            if self._available + n > self.capacity:
                dropped = self._available + n - self.capacity
                self._available = self.capacity
                self.overwrites += dropped
            else:
                self._available += n
            self.total_written += n
            return int(dropped)

    def write_bytes(self, raw: bytes) -> int:
        """写入小端 int16 PCM 字节流，返回被覆盖的样本数。"""
        if len(raw) % BYTES_PER_PCM16_SAMPLE:
            raw = raw[: len(raw) - (len(raw) % BYTES_PER_PCM16_SAMPLE)]
        return self.write(np.frombuffer(raw, dtype="<i2"))

    def write_float(self, samples: np.ndarray) -> int:
        """写入 [-1, 1] 的 float32 样本，内部转 int16。"""
        clipped = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
        return self.write((clipped * 32767.0).astype(np.int16))

    # ------------------------------------------------------------- 读
    def read(self, n: int) -> np.ndarray:
        """消费式读取最多 ``n`` 个样本，不足则返回全部可用样本。"""
        with self._lock:
            count = min(int(n), self._available)
            if count <= 0:
                return np.zeros(0, dtype=np.int16)
            start = (self._write - self._available) % self.capacity
            end = start + count
            if end <= self.capacity:
                out = self._buf[start:end].copy()
            else:
                head = self.capacity - start
                out = np.concatenate((self._buf[start:], self._buf[: count - head]))
            self._available -= count
            self.total_read += count
            return out

    def read_bytes(self, n_samples: int) -> bytes:
        """消费式读取并直接返回小端 int16 字节。"""
        return self.read(n_samples).astype("<i2").tobytes()

    def read_ms(self, duration_ms: float) -> np.ndarray:
        """按毫秒消费式读取。"""
        return self.read(int(self.sample_rate * self.channels * duration_ms / 1000.0))

    def peek_last(self, n: int) -> np.ndarray:
        """窥视最近写入的 ``n`` 个样本（不消费，也不影响可读游标）。"""
        with self._lock:
            count = min(int(n), self._available)
            if count <= 0:
                return np.zeros(0, dtype=np.int16)
            start = (self._write - count) % self.capacity
            end = start + count
            if end <= self.capacity:
                return self._buf[start:end].copy()
            head = self.capacity - start
            return np.concatenate((self._buf[start:], self._buf[: count - head]))

    def peek_last_ms(self, duration_ms: float) -> np.ndarray:
        """窥视最近 ``duration_ms`` 毫秒的音频（VAD 前缀 padding 用）。"""
        return self.peek_last(int(self.sample_rate * self.channels * duration_ms / 1000.0))

    def drain(self) -> np.ndarray:
        """一次性取走全部可读样本。"""
        return self.read(self.available)

    def clear(self) -> int:
        """丢弃全部内容，返回被丢弃的样本数。"""
        with self._lock:
            dropped = self._available
            self._available = 0
            self._write = 0
            self._buf.fill(0)
            return int(dropped)

    def stats(self) -> dict[str, float | int | str]:
        """可观测快照。"""
        return {
            "name": self.name,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "capacity_samples": self.capacity,
            "capacity_ms": round(self.capacity_ms, 2),
            "available_samples": self.available,
            "available_ms": round(self.available_ms, 2),
            "fill_ratio": round(self.fill_ratio, 4),
            "overwrites": self.overwrites,
            "total_written": self.total_written,
            "total_read": self.total_read,
        }

    def __len__(self) -> int:
        return self.available

    def __repr__(self) -> str:
        return (
            f"<AudioRingBuffer {self.name} {self.available}/{self.capacity} "
            f"samples overwrites={self.overwrites}>"
        )

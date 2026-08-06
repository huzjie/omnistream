"""轻量说话人切换检测（能量 + 谱特征在线聚类，非精确声纹）。

**定位**：不是 speaker verification，也不是离线 diarization。
目标只有一个 —— 在多人围坐同一支麦克风的会议/客服场景里，
**及时发现"换人说话了"**，让轮次归属、转写标注和记忆写入不至于串台。

方法（全部纯 numpy，零模型零依赖）：

1. 对每个语音帧提取 24 维 mel 能量 + 4 维韵律/音色特征
   （谱质心、谱滚降、谱平坦度、基频代理），L2 归一化成 28 维嵌入；
2. 用 ``window_ms`` 的滑动窗口做时间平滑，抑制单帧噪声；
3. 在线聚类：与已有说话人中心求余弦距离，
   最近中心距离 < ``assign_threshold`` 则归属之并做增量更新，
   否则（且已积累足够语音）新建一个说话人；
4. 当平滑嵌入与"当前说话人中心"的距离连续多帧超过 ``change_threshold``，
   判定为切换，输出 :class:`DiarizationEvent`。

精度定位是"够用"：同性别相近音色可能混淆，但对"男/女/远近场"这类
显著差异的切换检出率很高，且延迟在 300~600ms 量级。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

import numpy as np
import structlog

from omnistream.schemas.audio import AudioChunk

log = structlog.get_logger(__name__)

EPS: Final[float] = 1e-10

#: mel 滤波器组带数
DEFAULT_NUM_BANDS: Final[int] = 24

#: 韵律特征维数（质心/滚降/平坦度/基频代理）
PROSODY_DIMS: Final[int] = 4

#: 基频搜索范围（Hz）
F0_MIN_HZ: Final[float] = 60.0
F0_MAX_HZ: Final[float] = 400.0

#: 说话人标签前缀
SPEAKER_PREFIX: Final[str] = "spk_"

#: 未知/静音时的占位标签
UNKNOWN_SPEAKER: Final[str] = "spk_unknown"

_FILTERBANK_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


class DiarizationEventKind(StrEnum):
    """说话人事件类型。"""

    SPEAKER_STARTED = "speaker_started"
    SPEAKER_CHANGED = "speaker_changed"
    SPEAKER_ENDED = "speaker_ended"


@dataclass(slots=True, frozen=True)
class DiarizationEvent:
    """一次说话人事件。"""

    kind: DiarizationEventKind
    speaker: str
    pts_ms: int
    #: 与当前说话人中心的余弦距离，越大越"不像"
    distance: float = 0.0
    #: 判定置信度 [0,1]
    confidence: float = 0.0
    previous: str | None = None


@dataclass(slots=True)
class SpeakerSegment:
    """一段连续归属同一说话人的语音。"""

    speaker: str
    start_ms: int
    end_ms: int
    frames: int = 0
    confidence: float = 0.0

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


@dataclass(slots=True)
class _Profile:
    """一个说话人的在线中心。"""

    label: str
    centroid: np.ndarray
    frames: int = 0
    total_ms: float = 0.0
    last_pts_ms: int = 0
    distances: list[float] = field(default_factory=list)

    def update(self, vec: np.ndarray, frame_ms: float, pts_ms: int) -> None:
        n = float(self.frames)
        merged = (self.centroid * n + vec) / (n + 1.0)
        norm = float(np.linalg.norm(merged))
        self.centroid = merged / norm if norm > EPS else merged
        self.frames += 1
        self.total_ms += frame_ms
        self.last_pts_ms = pts_ms


def mel_filterbank(sample_rate: int, n_fft: int, num_bands: int) -> np.ndarray:
    """构造（并缓存）三角 mel 滤波器组，形状 ``(num_bands, n_fft // 2 + 1)``。"""
    key = (sample_rate, n_fft, num_bands)
    cached = _FILTERBANK_CACHE.get(key)
    if cached is not None:
        return cached

    def hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    low_hz, high_hz = 60.0, min(7600.0, sample_rate / 2.0 - 100.0)
    mel_points = np.linspace(hz_to_mel(low_hz), hz_to_mel(high_hz), num_bands + 2)
    hz_points = np.array([mel_to_hz(float(m)) for m in mel_points], dtype=np.float64)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((num_bands, n_fft // 2 + 1), dtype=np.float32)
    for i in range(num_bands):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        if center == left:
            center = min(left + 1, n_fft // 2)
        if right == center:
            right = min(center + 1, n_fft // 2)
        for k in range(left, center):
            fb[i, k] = (k - left) / max(1, center - left)
        for k in range(center, right):
            fb[i, k] = (right - k) / max(1, right - center)
    _FILTERBANK_CACHE[key] = fb
    return fb


def extract_embedding(samples: np.ndarray, sample_rate: int,
                      num_bands: int = DEFAULT_NUM_BANDS) -> np.ndarray:
    """从一帧单声道 float32 样本提取归一化音色嵌入。"""
    n = samples.size
    if n < 32:
        return np.zeros(num_bands + PROSODY_DIMS, dtype=np.float32)

    n_fft = 1 << int(math.ceil(math.log2(n)))
    windowed = samples * np.hanning(n).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft)).astype(np.float64) + EPS

    fb = mel_filterbank(sample_rate, n_fft, num_bands)
    band_energy = fb @ spectrum[: fb.shape[1]]
    log_bands = np.log(band_energy + EPS)
    # 倒谱均值归一化：抵消通道/音量差异，只保留音色形状
    log_bands = log_bands - float(np.mean(log_bands))
    scale = float(np.std(log_bands))
    if scale > EPS:
        log_bands = log_bands / scale

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)[: spectrum.size]
    total = float(np.sum(spectrum))
    centroid = float(np.sum(freqs * spectrum) / (total + EPS))
    cumulative = np.cumsum(spectrum)
    rolloff_idx = int(np.searchsorted(cumulative, 0.85 * cumulative[-1]))
    rolloff = float(freqs[min(rolloff_idx, freqs.size - 1)])
    geo = float(np.exp(np.mean(np.log(spectrum))))
    flatness = float(min(1.0, geo / (float(np.mean(spectrum)) + EPS)))
    f0 = estimate_f0(samples, sample_rate)

    prosody = np.array(
        [
            centroid / max(1.0, sample_rate / 2.0),
            rolloff / max(1.0, sample_rate / 2.0),
            flatness,
            f0 / F0_MAX_HZ,
        ],
        dtype=np.float32,
    )
    vec = np.concatenate([log_bands.astype(np.float32), prosody])
    norm = float(np.linalg.norm(vec))
    return (vec / norm).astype(np.float32) if norm > EPS else vec.astype(np.float32)


def estimate_f0(samples: np.ndarray, sample_rate: int) -> float:
    """自相关基频估计（Hz）；无明显周期性时返回 0。"""
    if samples.size < 64:
        return 0.0
    x = samples.astype(np.float64) - float(np.mean(samples))
    energy = float(np.dot(x, x))
    if energy < 1e-8:
        return 0.0
    corr = np.correlate(x, x, mode="full")[x.size - 1:]
    min_lag = max(1, int(sample_rate / F0_MAX_HZ))
    max_lag = min(x.size - 1, int(sample_rate / F0_MIN_HZ))
    if max_lag <= min_lag:
        return 0.0
    window = corr[min_lag:max_lag]
    if window.size == 0:
        return 0.0
    peak_idx = int(np.argmax(window))
    peak_value = float(window[peak_idx]) / (corr[0] + EPS)
    if peak_value < 0.3:
        return 0.0
    return float(sample_rate / (min_lag + peak_idx))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """两个归一化向量的余弦距离 [0,2]，实际取值多在 [0,1]。"""
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 1.0
    return float(max(0.0, 1.0 - float(np.dot(a, b))))


class LightweightDiarizer:
    """在线说话人切换检测器。"""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        num_bands: int = DEFAULT_NUM_BANDS,
        max_speakers: int = 4,
        assign_threshold: float = 0.30,
        change_threshold: float = 0.42,
        change_frames: int = 6,
        window_ms: int = 480,
        min_segment_ms: int = 400,
        warmup_ms: int = 600,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.num_bands = int(num_bands)
        self.max_speakers = max(1, int(max_speakers))
        self.assign_threshold = float(assign_threshold)
        self.change_threshold = float(change_threshold)
        self.change_frames = max(1, int(change_frames))
        self.window_ms = int(window_ms)
        self.min_segment_ms = int(min_segment_ms)
        self.warmup_ms = int(warmup_ms)

        self._profiles: dict[str, _Profile] = {}
        self._window: list[np.ndarray] = []
        self._window_ms_acc = 0.0
        self._current: str | None = None
        self._segment: SpeakerSegment | None = None
        self._mismatch_frames = 0
        self._speech_ms = 0.0
        self._next_index = 1
        self._last_distance = 0.0

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def current_speaker(self) -> str:
        """当前说话人标签；尚未判定时返回 ``spk_unknown``。"""
        return self._current or UNKNOWN_SPEAKER

    @property
    def speaker_count(self) -> int:
        return len(self._profiles)

    @property
    def last_distance(self) -> float:
        return self._last_distance

    def profiles(self) -> dict[str, dict[str, float]]:
        """各说话人的累计画像，供控制台展示。"""
        return {
            label: {
                "frames": float(p.frames),
                "total_ms": round(p.total_ms, 1),
                "last_pts_ms": float(p.last_pts_ms),
            }
            for label, p in self._profiles.items()
        }

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def observe(self, chunk: AudioChunk, *, is_speech: bool) -> DiarizationEvent | None:
        """喂入一帧音频与 VAD 结论，必要时返回说话人事件。"""
        pts_ms = int(chunk.pts_ms)
        frame_ms = float(chunk.duration_ms)

        if not is_speech:
            return self._on_silence(pts_ms)

        arr = chunk.to_ndarray()
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        samples = np.asarray(arr, dtype=np.float32).reshape(-1)
        vec = extract_embedding(samples, chunk.sample_rate, self.num_bands)
        if not np.any(vec):
            return None

        self._speech_ms += frame_ms
        self._push_window(vec, frame_ms)
        smoothed = self._smoothed()

        if self._speech_ms < self.warmup_ms and self._current is None:
            return None

        label, distance = self._nearest(smoothed)
        self._last_distance = distance

        if self._current is None:
            chosen = label if (label is not None and distance <= self.assign_threshold) \
                else self._new_profile(smoothed)
            self._enter_speaker(chosen, pts_ms, distance)
            self._profiles[chosen].update(smoothed, frame_ms, pts_ms)
            return DiarizationEvent(
                kind=DiarizationEventKind.SPEAKER_STARTED,
                speaker=chosen,
                pts_ms=pts_ms,
                distance=distance,
                confidence=self._confidence(distance),
            )

        own_distance = cosine_distance(smoothed, self._profiles[self.current_speaker].centroid)
        if own_distance <= self.change_threshold:
            self._mismatch_frames = 0
            self._profiles[self.current_speaker].update(smoothed, frame_ms, pts_ms)
            if self._segment is not None:
                self._segment.end_ms = pts_ms + int(frame_ms)
                self._segment.frames += 1
            return None

        self._mismatch_frames += 1
        if self._mismatch_frames < self.change_frames:
            return None

        # 确认切换：优先归入已有说话人，否则新建
        target = label if (label is not None and distance <= self.assign_threshold
                           and label != self._current) else None
        if target is None:
            target = self._new_profile(smoothed) if len(self._profiles) < self.max_speakers \
                else (label or self.current_speaker)
        previous = self._current
        self._enter_speaker(target, pts_ms, distance)
        self._profiles[target].update(smoothed, frame_ms, pts_ms)
        log.info("diarization.speaker_changed", previous=previous, speaker=target,
                 pts_ms=pts_ms, distance=round(distance, 3))
        return DiarizationEvent(
            kind=DiarizationEventKind.SPEAKER_CHANGED,
            speaker=target,
            pts_ms=pts_ms,
            distance=distance,
            confidence=self._confidence(distance),
            previous=previous,
        )

    def _on_silence(self, pts_ms: int) -> DiarizationEvent | None:
        """静音帧：不更新画像，只在段足够长时收尾。"""
        if self._segment is None or self._current is None:
            return None
        if self._segment.duration_ms < self.min_segment_ms:
            return None
        speaker = self._current
        self._segment.end_ms = max(self._segment.end_ms, pts_ms)
        finished = self._segment
        self._segment = None
        self._current = None
        self._window.clear()
        self._window_ms_acc = 0.0
        self._mismatch_frames = 0
        return DiarizationEvent(
            kind=DiarizationEventKind.SPEAKER_ENDED,
            speaker=speaker,
            pts_ms=finished.end_ms,
            distance=self._last_distance,
            confidence=finished.confidence,
        )

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _push_window(self, vec: np.ndarray, frame_ms: float) -> None:
        self._window.append(vec)
        self._window_ms_acc += frame_ms
        max_frames = max(1, int(self.window_ms / max(1.0, frame_ms)))
        while len(self._window) > max_frames:
            self._window.pop(0)

    def _smoothed(self) -> np.ndarray:
        if not self._window:
            return np.zeros(self.num_bands + PROSODY_DIMS, dtype=np.float32)
        stacked = np.mean(np.stack(self._window, axis=0), axis=0)
        norm = float(np.linalg.norm(stacked))
        return (stacked / norm).astype(np.float32) if norm > EPS else stacked.astype(np.float32)

    def _nearest(self, vec: np.ndarray) -> tuple[str | None, float]:
        best_label: str | None = None
        best_dist = 1.0
        for label, profile in self._profiles.items():
            dist = cosine_distance(vec, profile.centroid)
            if dist < best_dist:
                best_dist = dist
                best_label = label
        return best_label, best_dist

    def _new_profile(self, vec: np.ndarray) -> str:
        label = f"{SPEAKER_PREFIX}{self._next_index}"
        self._next_index += 1
        self._profiles[label] = _Profile(label=label, centroid=vec.copy())
        return label

    def _enter_speaker(self, label: str, pts_ms: int, distance: float) -> None:
        self._current = label
        self._mismatch_frames = 0
        self._segment = SpeakerSegment(
            speaker=label,
            start_ms=pts_ms,
            end_ms=pts_ms,
            frames=1,
            confidence=self._confidence(distance),
        )

    @staticmethod
    def _confidence(distance: float) -> float:
        return float(max(0.0, min(1.0, 1.0 - distance)))

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def flush(self) -> SpeakerSegment | None:
        """结束当前段并返回（会话结束/强制切轮时调用）。"""
        segment = self._segment
        self._segment = None
        self._current = None
        self._window.clear()
        self._window_ms_acc = 0.0
        self._mismatch_frames = 0
        return segment

    def reset(self, *, keep_profiles: bool = True) -> None:
        """复位。默认保留说话人画像，使跨轮次标签保持稳定。"""
        self.flush()
        self._speech_ms = 0.0
        self._last_distance = 0.0
        if not keep_profiles:
            self._profiles.clear()
            self._next_index = 1

    def __repr__(self) -> str:
        return (f"<LightweightDiarizer speakers={len(self._profiles)} "
                f"current={self.current_speaker}>")


__all__ = [
    "DEFAULT_NUM_BANDS",
    "F0_MAX_HZ",
    "F0_MIN_HZ",
    "PROSODY_DIMS",
    "SPEAKER_PREFIX",
    "UNKNOWN_SPEAKER",
    "DiarizationEvent",
    "DiarizationEventKind",
    "LightweightDiarizer",
    "SpeakerSegment",
    "cosine_distance",
    "estimate_f0",
    "extract_embedding",
    "mel_filterbank",
]

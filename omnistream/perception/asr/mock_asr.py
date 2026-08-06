"""确定性 Mock ASR —— 零模型、零网络、可离线跑通全链路。

它不是"假装返回一句固定话"的桩，而是一个**真实工作的音频驱动模拟器**：

1. 从 PCM 计算短时能量包络，用能量峰谷切出"音节"（syllable onset detection）；
2. 每检测到一个音节，就从确定性词库中按 ``(session_id, 音节序号)`` 的哈希
   取一个词吐出去 —— 同样的音频永远得到同样的文本，CI 可断言；
3. 输出节奏与真实引擎一致：``asr_chunk_ms`` 窗口出 partial、
   静音超过阈值或输入流结束出 final；
4. 支持词级时间戳（由真实的音节位置推出，不是编的）。

因此它能真实地驱动稳定器、标点恢复、端点检测、打断等下游逻辑，
是默认引擎（``OMS_PERCEPTION_ASR_ENGINE=mock``）与所有降级路径的落点。
"""
from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from typing import Any, Final, Iterable

import numpy as np
import structlog

from omnistream.perception.asr.base import (
    ASR_SAMPLE_RATE,
    ASRCapabilities,
    BaseStreamingASR,
    build_delta,
)
from omnistream.perception.asr.punctuation import PauseMark, PunctuationRestorer
from omnistream.perception.asr.stabilizer import LocalAgreementStabilizer
from omnistream.perception.asr.word_timing import WordSpan, interpolate_words
from omnistream.schemas.audio import AudioChunk
from omnistream.schemas.transcript import TranscriptDelta

log = structlog.get_logger(__name__)

__all__ = ["MockStreamingASR", "SYLLABLE_LEXICON", "LEXICON_EN"]

#: 中文音节词库。挑选的都是高频字，拼起来读得通、看起来像真的转写。
SYLLABLE_LEXICON: Final[tuple[str, ...]] = (
    "我", "你", "他", "们", "这", "那", "个", "是", "的", "了",
    "在", "有", "和", "就", "不", "人", "都", "一", "上", "也",
    "很", "到", "说", "要", "去", "会", "着", "没", "看", "好",
    "自", "己", "还", "把", "做", "想", "能", "下", "过", "来",
    "现", "在", "时", "候", "问", "题", "系", "统", "语", "音",
    "实", "时", "交", "互", "模", "型", "识", "别", "内", "容",
)

#: 英文词库，用于 ``language`` 以 ``en`` 开头时。
LEXICON_EN: Final[tuple[str, ...]] = (
    "the", "and", "you", "that", "this", "with", "have", "from", "they", "will",
    "would", "there", "their", "what", "about", "which", "when", "make", "like",
    "time", "just", "know", "take", "into", "year", "your", "good", "some",
    "could", "them", "than", "then", "look", "only", "come", "over", "think",
    "also", "back", "after", "work", "first", "well", "even", "want", "because",
)

#: 能量高于该门限视为有声（dBFS）
_SPEECH_DBFS: Final[float] = -42.0

#: 两个音节之间的最小间隔（毫秒），防止一个音被切成两个
_MIN_SYLLABLE_GAP_MS: Final[float] = 90.0

#: 连续静音超过该值即认为一句话说完
_UTTERANCE_SILENCE_MS: Final[float] = 620.0

#: 分析窗（毫秒），能量包络的时间分辨率
_ANALYSIS_MS: Final[float] = 20.0


def _dbfs(samples: np.ndarray) -> float:
    """计算一段样本的 RMS dBFS。"""
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return 20.0 * float(np.log10(rms)) if rms > 1e-9 else -120.0


def _pick(lexicon: tuple[str, ...], seed: str, index: int) -> str:
    """从词库中确定性地取词。"""
    digest = hashlib.blake2b(f"{seed}:{index}".encode("utf-8"), digest_size=8).digest()
    return lexicon[int.from_bytes(digest, "big") % len(lexicon)]


class MockStreamingASR(BaseStreamingASR):
    """音频驱动的确定性流式识别器。

    Example::

        asr = MockStreamingASR(session_id="sess_1", turn_id="turn_1")
        async for delta in asr.stream(audio_iter):
            print(delta.text, delta.is_final)
    """

    name = "mock"
    capabilities = ASRCapabilities(
        streaming=True,
        word_timestamps=True,
        language_detection=False,
        hotwords=True,
        punctuation=True,
        needs_gpu=False,
        needs_network=False,
        typical_partial_ms=120,
        languages=("zh", "en", "ja", "ko", "ru"),
    )

    def __init__(
        self,
        *,
        language: str = "zh-CN",
        sample_rate: int = ASR_SAMPLE_RATE,
        turn_id: str = "",
        session_id: str = "",
        hotwords: Iterable[str] | None = None,
        enable_punctuation: bool = True,
        enable_word_timestamps: bool = True,
        partial_interval_ms: int = 240,
        seed: str = "",
    ) -> None:
        """
        Args:
            partial_interval_ms: 两次 partial 之间的最小间隔。
            seed: 词库采样种子，留空则用 ``session_id``。
        """
        super().__init__(
            language=language,
            sample_rate=sample_rate,
            turn_id=turn_id,
            session_id=session_id,
            hotwords=hotwords,
            enable_punctuation=enable_punctuation,
            enable_word_timestamps=enable_word_timestamps,
        )
        self.partial_interval_ms = int(partial_interval_ms)
        self.seed = seed or session_id or "omnistream"
        self._stabilizer = LocalAgreementStabilizer(agreement=2)
        self._punct = PunctuationRestorer(language=language)

        # 解码状态
        self._tokens: list[str] = []
        self._spans: list[WordSpan] = []
        self._pauses: list[PauseMark] = []
        self._elapsed_ms = 0.0
        self._segment_start_ms = 0.0
        self._silence_ms = 0.0
        self._last_syllable_ms = -1e9
        self._last_partial_ms = -1e9
        self._syllable_index = 0
        self._prev_voiced = False
        self._prev_db = -120.0
        self._delta_index = 0
        self._residual = np.zeros(0, dtype=np.float32)

    # ------------------------------------------------------------------
    # 核心：音频 -> 音节 -> 文本
    # ------------------------------------------------------------------

    @property
    def _lexicon(self) -> tuple[str, ...]:
        return LEXICON_EN if self.language.lower().startswith("en") else SYLLABLE_LEXICON

    def _window_samples(self) -> int:
        return max(1, int(self.sample_rate * _ANALYSIS_MS / 1000.0))

    def _ingest(self, chunk: AudioChunk) -> int:
        """吸收一帧音频，返回本帧新识别出的音节数。"""
        try:
            samples = np.asarray(chunk.to_ndarray(), dtype=np.float32).reshape(-1)
        except ValueError:
            # 非 PCM 编码（opus/mulaw）在到达 ASR 前应已解码；此处按静音处理
            samples = np.zeros(0, dtype=np.float32)
        if samples.size:
            self._residual = np.concatenate([self._residual, samples])

        window = self._window_samples()
        new_syllables = 0
        while self._residual.size >= window:
            frame = self._residual[:window]
            self._residual = self._residual[window:]
            db = _dbfs(frame)
            voiced = db > _SPEECH_DBFS
            if voiced:
                if self._silence_ms >= 120.0 and self._tokens:
                    self._pauses.append(PauseMark(
                        char_index=len(self._token_text()),
                        pause_ms=int(self._silence_ms),
                    ))
                self._silence_ms = 0.0
                # onset：从静音进入有声，或能量显著上跳（连读中的新音节）
                rising = db - self._prev_db > 6.0
                gap_ok = self._elapsed_ms - self._last_syllable_ms >= _MIN_SYLLABLE_GAP_MS
                if (not self._prev_voiced or rising) and gap_ok:
                    self._emit_syllable(db)
                    new_syllables += 1
            else:
                self._silence_ms += _ANALYSIS_MS
            self._prev_voiced = voiced
            self._prev_db = db
            self._elapsed_ms += _ANALYSIS_MS
        return new_syllables

    def _emit_syllable(self, db: float) -> None:
        """记录一个音节及其时间戳。"""
        if not self._tokens:
            self._segment_start_ms = self._elapsed_ms
        token = _pick(self._lexicon, self.seed, self._syllable_index)
        self._syllable_index += 1
        self._tokens.append(token)
        start = int(self._elapsed_ms)
        # 音节时长按能量给个合理估计：越响的音节读得越久（150~320ms）
        loudness = min(1.0, max(0.0, (db + 60.0) / 40.0))
        duration = 150.0 + loudness * 170.0
        confidence = 0.55 + loudness * 0.4
        self._spans.append(WordSpan(word=token, start_ms=start,
                                    end_ms=start + int(duration),
                                    confidence=round(confidence, 3)))
        self._last_syllable_ms = self._elapsed_ms

    def _token_text(self) -> str:
        """当前累计的原始文本（未加标点）。"""
        if self.language.lower().startswith("en"):
            return " ".join(self._tokens)
        return "".join(self._tokens)

    def _render(self, *, is_final: bool) -> str:
        """渲染成带标点的文本。"""
        raw = self._token_text()
        if not raw or not self.enable_punctuation:
            return raw
        return self._punct.restore(raw, pauses=self._pauses, is_final=is_final)

    def _make_delta(self, *, is_final: bool, stability: float) -> TranscriptDelta:
        text = self._render(is_final=is_final)
        words = []
        if self.enable_word_timestamps and self._spans:
            words = [s.to_model() for s in self._spans]
        elif self.enable_word_timestamps and text:
            words = [s.to_model() for s in interpolate_words(
                text, int(self._segment_start_ms), int(self._elapsed_ms))]
        delta = build_delta(
            turn_id=self.turn_id,
            text=text,
            is_final=is_final,
            stability=stability,
            start_ms=int(self._segment_start_ms),
            end_ms=int(self._elapsed_ms),
            speaker="user",
            words=words,
            language=self.language,
            engine=self.name,
        )
        try:
            delta.index = self._delta_index
        except (AttributeError, ValueError):
            pass
        self._delta_index += 1
        return delta

    def _reset_segment(self) -> None:
        """一句话结束后清空段状态（保留全局时间轴）。"""
        self._tokens.clear()
        self._spans.clear()
        self._pauses.clear()
        self._stabilizer.reset()
        self._segment_start_ms = self._elapsed_ms
        self._delta_index = 0

    # ------------------------------------------------------------------
    # BaseStreamingASR 接口
    # ------------------------------------------------------------------

    async def stream(self, chunks: AsyncIterator[AudioChunk]) -> AsyncIterator[TranscriptDelta]:
        """消费音频流并产出增量转写。"""
        emitted_final = False
        async for chunk in chunks:
            self._ingest(chunk)
            if not self._tokens:
                continue

            # 一句话说完：静音够长 -> final
            if self._silence_ms >= _UTTERANCE_SILENCE_MS:
                yield self._make_delta(is_final=True, stability=1.0)
                emitted_final = True
                self._reset_segment()
                self._silence_ms = 0.0
                continue

            # 否则按节奏出 partial
            if self._elapsed_ms - self._last_partial_ms >= self.partial_interval_ms:
                self._last_partial_ms = self._elapsed_ms
                stabilized = self._stabilizer.update(self._token_text())
                yield self._make_delta(is_final=False,
                                       stability=stabilized.stability)
                emitted_final = False

        # 输入结束：补一条 final（契约要求）
        final = self._make_delta(is_final=True, stability=1.0)
        if self._tokens or not emitted_final:
            yield final
        self._reset_segment()

    async def reset(self) -> None:
        """清空全部解码状态。"""
        self._reset_segment()
        self._elapsed_ms = 0.0
        self._silence_ms = 0.0
        self._last_syllable_ms = -1e9
        self._last_partial_ms = -1e9
        self._syllable_index = 0
        self._prev_voiced = False
        self._prev_db = -120.0
        self._residual = np.zeros(0, dtype=np.float32)
        self._punct.reset()

    async def finalize(self) -> TranscriptDelta | None:
        """强制收尾，返回最终 delta。"""
        if not self._tokens:
            return None
        delta = self._make_delta(is_final=True, stability=1.0)
        self._reset_segment()
        return delta

    def stats(self) -> dict[str, Any]:
        data = super().stats()
        data.update({
            "syllables": self._syllable_index,
            "elapsed_ms": round(self._elapsed_ms, 1),
            "pending_tokens": len(self._tokens),
        })
        return data

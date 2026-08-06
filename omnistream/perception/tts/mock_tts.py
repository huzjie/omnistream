"""零依赖的确定性 TTS 引擎：共振峰合成器。

为什么不是"输出一段静音"
------------------------
Mock TTS 是整个全双工链路的**默认引擎**，CI、本地开发、云端依赖缺失时都会用到。
如果它只吐静音，那么下游的一切都无法被真实验证：

* 播放队列的 ``cancel_from(pts)`` 到底截断在哪一帧？
* barge-in 的回声消除参考信号有没有能量？
* AEC / 噪声门 / 增益链路会不会把有效语音吃掉？
* 前端波形图、音量条是不是活的？

所以这里实现了一个**真正的源-滤波器（source-filter）语音合成器**：
声门脉冲谐波源 + 共振峰滤波 + 辅音噪声 + 声调轮廓 + 句末降调。
输出的音频听起来像"外星人说中文"——不可懂，但**声学特征与真实语音同构**
（基频 80~260Hz、共振峰 F1/F2/F3、音节节奏 5Hz 左右、句末 F0 下倾），
足以驱动 VAD、AEC、能量检测、波形可视化的全部逻辑。

确定性
------
同样的文本 + 同样的音色 + 同样的韵律，永远产出**逐样点相同**的音频。
所有随机量都来自 ``blake2b(text)`` 派生的种子，不使用全局随机源。
这让缓存、回归对比、录制回放都可复现。
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import AsyncIterator
from typing import Final

import numpy as np
import structlog

from omnistream.perception.tts.base import (
    DEFAULT_TTS_SAMPLE_RATE,
    TTS_FRAME_MS,
    BaseStreamingTTS,
    SynthesisRequest,
    TTSCapabilities,
)
from omnistream.schemas.audio import AudioChunk

log = structlog.get_logger(__name__)

__all__ = [
    "CONSONANTS",
    "MockStreamingTTS",
    "PAUSE_MS",
    "TONE_CONTOURS",
    "VOWEL_FORMANTS",
    "estimate_duration_ms",
]

#: 元音共振峰表 ``(名称, F1, F2, F3)``，取自汉语普通话单元音的典型测量值（男声）。
VOWEL_FORMANTS: Final[tuple[tuple[str, float, float, float], ...]] = (
    ("a", 730.0, 1090.0, 2440.0),
    ("o", 570.0, 840.0, 2410.0),
    ("e", 530.0, 1840.0, 2480.0),
    ("i", 270.0, 2290.0, 3010.0),
    ("u", 300.0, 870.0, 2240.0),
    ("v", 290.0, 2100.0, 2600.0),
    ("ai", 660.0, 1700.0, 2500.0),
    ("ou", 450.0, 1030.0, 2380.0),
)

#: 声调轮廓：5 个采样点上的基频倍率（普通话四声 + 轻声）。
TONE_CONTOURS: Final[tuple[tuple[float, ...], ...]] = (
    (1.10, 1.11, 1.10, 1.10, 1.09),   # 阴平：高平
    (0.92, 0.95, 1.02, 1.12, 1.21),   # 阳平：中升
    (0.96, 0.86, 0.79, 0.88, 1.03),   # 上声：降升
    (1.22, 1.10, 0.97, 0.85, 0.76),   # 去声：全降
    (1.00, 0.98, 0.96, 0.95, 0.93),   # 轻声：短促微降
)

#: 辅音类型：``(名称, 时长毫秒, 噪声高通强度, 是否浊音, 是否爆破)``
CONSONANTS: Final[tuple[tuple[str, int, float, bool, bool], ...]] = (
    ("none", 0, 0.0, False, False),
    ("fricative", 62, 0.85, False, False),   # s / sh / f / x
    ("plosive", 14, 0.60, False, True),      # p / t / k / b / d / g
    ("nasal", 46, 0.00, True, False),        # m / n / l
)

#: 标点对应的停顿时长（毫秒）
PAUSE_MS: Final[dict[str, int]] = {
    "，": 130, ",": 130,
    "。": 270, ".": 240,
    "！": 270, "!": 250,
    "？": 280, "?": 260,
    "；": 190, ";": 180,
    "：": 150, ":": 140,
    "、": 95,
    "…": 300, "—": 200,
    "\n": 240, "\t": 60, " ": 45,
    "（": 60, "）": 60, "(": 60, ")": 60,
    "“": 40, "”": 40, "\"": 40, "'": 30,
    "《": 50, "》": 50,
}

#: 单个汉字音节的基准时长（毫秒，speed=1.0）
_SYLLABLE_MS: Final[int] = 196

#: 拉丁字母音节的基准时长（毫秒）
_LATIN_SYLLABLE_MS: Final[int] = 168

#: 数字逐位朗读的时长（毫秒）
_DIGIT_MS: Final[int] = 210

#: 音色 -> ``(基频 Hz, 共振峰缩放, 气声比例)``。
#: 女声共振峰整体上移约 17%，基频约为男声的 1.7 倍。
_VOICE_PRESETS: Final[dict[str, tuple[float, float, float]]] = {
    "default": (196.0, 1.15, 0.030),
    "xiaoyan": (208.0, 1.18, 0.034),
    "narrator": (188.0, 1.12, 0.026),
    "news_male": (116.0, 1.00, 0.022),
    "en_female": (202.0, 1.16, 0.032),
    "en_male": (112.0, 0.99, 0.020),
}

#: 未知音色的兜底预设
_FALLBACK_PRESET: Final[tuple[float, float, float]] = (170.0, 1.08, 0.028)

#: 情绪 -> ``(基频倍率, 语速倍率, 音高抖动倍率)``
_EMOTION_PRESETS: Final[dict[str, tuple[float, float, float]]] = {
    "": (1.00, 1.00, 1.00),
    "neutral": (1.00, 1.00, 1.00),
    "cheerful": (1.10, 1.06, 1.5),
    "excited": (1.14, 1.12, 1.8),
    "friendly": (1.04, 1.00, 1.1),
    "sad": (0.90, 0.90, 0.7),
    "serious": (0.96, 0.94, 0.6),
    "angry": (1.08, 1.10, 2.0),
    "calm": (0.97, 0.95, 0.7),
    "gentle": (1.00, 0.96, 0.9),
    "whisper": (0.98, 0.98, 1.0),
}

#: 拉丁词的元音字母，用于估算音节数
_LATIN_VOWELS: Final[frozenset[str]] = frozenset("aeiouyAEIOUY")

#: 分词正则：汉字 / 拉丁词 / 数字 / 其他单字符
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf]|[A-Za-z]+|[0-9]+|\s+|.",
    re.UNICODE,
)


def _is_cjk(ch: str) -> bool:
    """是否为 CJK 表意文字。"""
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF


def _latin_syllables(word: str) -> int:
    """粗略估算英文单词的音节数（元音簇计数法）。"""
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in _LATIN_VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith(("e", "E")) and count > 1:
        count -= 1
    return max(1, count)


def estimate_duration_ms(text: str, *, speed: float = 1.0) -> int:
    """估算一段文本的合成时长（毫秒）。

    路由器用它做 TTFA 预算判断，缓存用它做容量核算。

    Args:
        text: 待合成文本。
        speed: 语速倍率。

    Returns:
        预估毫秒数，至少为 0。
    """
    if not text:
        return 0
    total = 0.0
    for token in _TOKEN_RE.findall(text):
        if not token:
            continue
        head = token[0]
        if _is_cjk(head):
            total += _SYLLABLE_MS
        elif head.isascii() and head.isalpha():
            total += _LATIN_SYLLABLE_MS * _latin_syllables(token)
        elif head.isdigit():
            total += _DIGIT_MS * len(token)
        else:
            total += PAUSE_MS.get(head, 55)
    rate = max(0.1, float(speed))
    return int(total / rate)


class MockStreamingTTS(BaseStreamingTTS):
    """确定性共振峰合成器，零外部依赖。

    Example::

        tts = MockStreamingTTS(voice="xiaoyan")
        async for chunk in tts.synthesize(text_stream):
            await playback.enqueue(chunk)
    """

    name = "mock"

    capabilities = TTSCapabilities(
        streaming=True,
        ssml=False,
        voice_cloning=False,
        emotion=True,
        voices=len(_VOICE_PRESETS),
        needs_gpu=False,
        needs_network=False,
        typical_ttfa_ms=25,
        languages=("zh", "en", "ja", "ko", "*"),
        sample_rates=(8000, 16000, 22050, 24000, 48000),
    )

    def __init__(
        self,
        *,
        voice: str = "default",
        language: str = "zh-CN",
        sample_rate: int = DEFAULT_TTS_SAMPLE_RATE,
        speed: float = 1.0,
        pitch: float = 0.0,
        volume_db: float = 0.0,
        frame_ms: int = TTS_FRAME_MS,
        session_id: str = "",
        simulate_latency_ms: int = 0,
        target_peak: float = 0.62,
        first_chunk_chars: int = 8,
    ) -> None:
        """初始化。

        Args:
            voice: 逻辑音色名，决定基频与共振峰缩放。
            language: 语言标签（仅影响节奏微调）。
            sample_rate: 输出采样率。
            speed: 语速倍率。
            pitch: 音高偏移（半音）。
            volume_db: 音量增益（dB）。
            frame_ms: 输出帧长。
            session_id: 会话 ID，仅用于日志。
            simulate_latency_ms: 人为注入的首包延迟，用于压测 TTFA 预算。
            target_peak: 归一化峰值，避免削顶。
            first_chunk_chars: 首句最小字符数，越小 TTFA 越低。
        """
        super().__init__(
            voice=voice,
            language=language,
            sample_rate=sample_rate,
            speed=speed,
            pitch=pitch,
            volume_db=volume_db,
            frame_ms=frame_ms,
            session_id=session_id,
        )
        self.simulate_latency_ms = max(0, int(simulate_latency_ms))
        self.target_peak = max(0.05, min(0.95, float(target_peak)))
        self.first_chunk_chars = max(1, int(first_chunk_chars))
        self._served = 0

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """永远可用（只依赖 numpy）。"""
        return True

    def available_voices(self) -> list[str]:
        """内置音色列表。"""
        return sorted(_VOICE_PRESETS)

    async def synthesize(self, text_stream: AsyncIterator[str]) -> AsyncIterator[AudioChunk]:
        """消费文本流并产出音频帧。"""
        async for chunk in self._stream_sentences(
            text_stream, first_chunk_chars=self.first_chunk_chars
        ):
            yield chunk

    async def synthesize_text(self, request: SynthesisRequest) -> np.ndarray:
        """合成一小句，返回单声道 float32 PCM。"""
        if self.simulate_latency_ms > 0:
            await asyncio.sleep(self.simulate_latency_ms / 1000.0)
        if self.cancelled:
            return np.zeros(0, dtype=np.float32)
        self._served += 1
        return self._render(request)

    def stats(self) -> dict[str, object]:
        """统计快照。"""
        base = super().stats()
        base["served"] = self._served
        base["simulate_latency_ms"] = self.simulate_latency_ms
        return base

    # ------------------------------------------------------------------
    # 合成实现
    # ------------------------------------------------------------------

    def _preset(self, voice: str) -> tuple[float, float, float]:
        """取音色预设，未知音色按名字哈希稳定地落到一个预设上。"""
        if voice in _VOICE_PRESETS:
            return _VOICE_PRESETS[voice]
        if not voice:
            return _FALLBACK_PRESET
        digest = hashlib.blake2b(voice.encode("utf-8"), digest_size=4).digest()
        keys = sorted(_VOICE_PRESETS)
        return _VOICE_PRESETS[keys[digest[0] % len(keys)]]

    def _render(self, request: SynthesisRequest) -> np.ndarray:
        """把一小句渲染成 PCM。"""
        text = request.text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)

        sr = self.sample_rate
        f0_base, formant_scale, breath = self._preset(request.voice)
        emo_f0, emo_rate, emo_jitter = _EMOTION_PRESETS.get(
            request.emotion, _EMOTION_PRESETS[""]
        )
        # 音高偏移：半音 -> 频率倍率
        f0_base *= emo_f0 * (2.0 ** (request.pitch / 12.0))
        rate = max(0.1, request.speed * emo_rate)

        seed = int.from_bytes(
            hashlib.blake2b(
                f"{request.voice}|{text}".encode("utf-8"), digest_size=8
            ).digest(),
            "big",
        )
        rng = np.random.default_rng(seed % (2**63))

        tokens = [t for t in _TOKEN_RE.findall(text) if t]
        # 预扫描音节总数，用于计算句内基频下倾（declination）
        syllable_total = max(1, sum(self._token_syllables(t) for t in tokens))

        parts: list[np.ndarray] = []
        syllable_index = 0
        for token in tokens:
            head = token[0]
            if head.isspace():
                parts.append(self._silence(PAUSE_MS.get(" ", 45) / rate))
                continue
            if _is_cjk(head):
                progress = syllable_index / syllable_total
                parts.append(
                    self._syllable(token, f0_base, formant_scale, breath,
                                   rate, progress, emo_jitter, rng)
                )
                syllable_index += 1
                continue
            if head.isascii() and head.isalpha():
                count = _latin_syllables(token)
                for i in range(count):
                    progress = syllable_index / syllable_total
                    parts.append(
                        self._syllable(f"{token}#{i}", f0_base, formant_scale, breath,
                                       rate, progress, emo_jitter, rng,
                                       base_ms=_LATIN_SYLLABLE_MS)
                    )
                    syllable_index += 1
                parts.append(self._silence(28.0 / rate))
                continue
            if head.isdigit():
                for i, digit in enumerate(token):
                    progress = syllable_index / syllable_total
                    parts.append(
                        self._syllable(f"num{digit}", f0_base, formant_scale, breath,
                                       rate, progress, emo_jitter, rng,
                                       base_ms=_DIGIT_MS)
                    )
                    syllable_index += 1
                continue
            pause = PAUSE_MS.get(head)
            if pause is None:
                progress = syllable_index / syllable_total
                parts.append(
                    self._syllable(head, f0_base, formant_scale, breath,
                                   rate, progress, emo_jitter, rng,
                                   base_ms=_LATIN_SYLLABLE_MS)
                )
                syllable_index += 1
            else:
                parts.append(self._silence(pause / rate))

        parts = [p for p in parts if p.size > 0]
        if not parts:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(parts).astype(np.float32, copy=False)

        # 句首/句尾各 8ms 淡入淡出，消除拼接爆音
        audio = self._fade_edges(audio, sr, 8)
        # 归一化到目标峰值后再叠加用户音量
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-6:
            audio *= self.target_peak / peak
        gain = 10.0 ** (request.volume_db / 20.0)
        audio *= gain
        np.clip(audio, -1.0, 1.0, out=audio)
        return audio

    def _token_syllables(self, token: str) -> int:
        """一个 token 折算成几个音节（供下倾曲线用）。"""
        head = token[0]
        if _is_cjk(head):
            return 1
        if head.isascii() and head.isalpha():
            return _latin_syllables(token)
        if head.isdigit():
            return len(token)
        if head.isspace() or head in PAUSE_MS:
            return 0
        return 1

    def _silence(self, duration_ms: float) -> np.ndarray:
        """生成静音段。"""
        n = int(self.sample_rate * max(0.0, duration_ms) / 1000.0)
        return np.zeros(n, dtype=np.float32)

    def _syllable(
        self,
        key: str,
        f0_base: float,
        formant_scale: float,
        breath: float,
        rate: float,
        progress: float,
        jitter_scale: float,
        rng: np.random.Generator,
        *,
        base_ms: int = _SYLLABLE_MS,
    ) -> np.ndarray:
        """合成一个音节：辅音 + 浊元音。

        Args:
            key: 决定音色参数的确定性键（通常是那个字）。
            f0_base: 基频。
            formant_scale: 共振峰缩放（女声 >1）。
            breath: 气声噪声比例。
            rate: 语速倍率。
            progress: 该音节在句中的相对位置 0~1，用于基频下倾。
            jitter_scale: 音高抖动强度倍率。
            rng: 确定性随机源。
            base_ms: 基准时长。

        Returns:
            单声道 float32 PCM。
        """
        sr = self.sample_rate
        h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()

        vowel = VOWEL_FORMANTS[h[0] % len(VOWEL_FORMANTS)]
        tone = TONE_CONTOURS[h[1] % len(TONE_CONTOURS)]
        cons_name, cons_ms, cons_hp, cons_voiced, cons_burst = CONSONANTS[
            h[2] % len(CONSONANTS)
        ]

        # 时长抖动 ±15%，句末音节延长 22%（final lengthening）
        length_jitter = 0.85 + (h[3] / 255.0) * 0.30
        final_stretch = 1.22 if progress > 0.88 else 1.0
        vowel_ms = base_ms * length_jitter * final_stretch / rate
        cons_len_ms = cons_ms / rate

        segments: list[np.ndarray] = []
        if cons_len_ms > 0.5:
            segments.append(
                self._consonant(cons_len_ms, cons_hp, cons_voiced, cons_burst,
                                f0_base, rng)
            )

        n = max(1, int(sr * vowel_ms / 1000.0))
        t_norm = np.linspace(0.0, 1.0, n, dtype=np.float32)

        # 基频轮廓：声调 x 句内下倾 x 微抖动
        contour = np.interp(
            t_norm,
            np.linspace(0.0, 1.0, len(tone), dtype=np.float32),
            np.asarray(tone, dtype=np.float32),
        ).astype(np.float32)
        declination = 1.0 - 0.14 * float(progress)
        jitter = 1.0 + (rng.standard_normal(n).astype(np.float32) * 0.0035 * jitter_scale)
        # 抖动做一次滑动平均，避免变成高频噪声
        if n >= 9:
            kernel = np.ones(9, dtype=np.float32) / 9.0
            jitter = np.convolve(jitter, kernel, mode="same").astype(np.float32)
        f0 = np.clip(f0_base * contour * declination * jitter, 55.0, 500.0)

        # 连续相位积分，保证无相位跳变
        phase = np.cumsum(2.0 * np.pi * f0 / sr, dtype=np.float64).astype(np.float32)

        formants = (
            (vowel[1] * formant_scale, 90.0, 1.00),
            (vowel[2] * formant_scale, 120.0, 0.62),
            (vowel[3] * formant_scale, 180.0, 0.30),
            (3600.0 * formant_scale, 260.0, 0.10),
        )
        mean_f0 = float(np.mean(f0))
        max_harmonic = max(2, min(80, int((sr * 0.45) / max(60.0, mean_f0))))

        wave = np.zeros(n, dtype=np.float32)
        for k in range(1, max_harmonic + 1):
            freq = mean_f0 * k
            gain = 0.0
            for center, bandwidth, amp in formants:
                gain += amp / (1.0 + ((freq - center) / (bandwidth * 0.5)) ** 2)
            # 声门源频谱 -12dB/oct 衰减
            gain *= 1.0 / (k ** 1.15)
            if gain < 1e-4:
                continue
            wave += gain * np.sin(phase * k, dtype=np.float32)

        if breath > 0.0:
            noise = rng.standard_normal(n).astype(np.float32)
            # 一阶高通，让气声集中在高频
            noise = np.diff(noise, prepend=np.float32(0.0)).astype(np.float32)
            wave += noise * breath * float(np.max(np.abs(wave)) or 1.0)

        wave *= self._envelope(n, sr)
        segments.append(wave)
        return np.concatenate(segments) if len(segments) > 1 else segments[0]

    def _consonant(
        self,
        duration_ms: float,
        highpass: float,
        voiced: bool,
        burst: bool,
        f0_base: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """生成辅音段：擦音噪声 / 爆破冲激 / 鼻音低频。"""
        sr = self.sample_rate
        n = max(1, int(sr * duration_ms / 1000.0))
        if voiced:
            # 鼻音：低频正弦 + 强阻尼
            t = np.arange(n, dtype=np.float32) / sr
            sig = (np.sin(2.0 * np.pi * f0_base * t, dtype=np.float32) * 0.55
                   + np.sin(2.0 * np.pi * 260.0 * t, dtype=np.float32) * 0.25)
            sig *= np.exp(-3.0 * np.linspace(0.0, 1.0, n, dtype=np.float32))
            return (sig * 0.30).astype(np.float32)

        noise = rng.standard_normal(n).astype(np.float32)
        if highpass > 0.0:
            # 一阶差分高通，highpass 控制混合比例
            hp = np.diff(noise, prepend=np.float32(0.0)).astype(np.float32)
            noise = noise * (1.0 - highpass) + hp * highpass
        if burst:
            # 爆破音：极短冲激包络 + 后随静默
            env = np.exp(-12.0 * np.linspace(0.0, 1.0, n, dtype=np.float32))
            noise = noise * env * 0.55
            tail = np.zeros(max(1, int(sr * 0.012)), dtype=np.float32)
            return np.concatenate([noise, tail]).astype(np.float32)
        env = np.hanning(n).astype(np.float32) if n > 2 else np.ones(n, dtype=np.float32)
        return (noise * env * 0.22).astype(np.float32)

    def _envelope(self, n: int, sr: int) -> np.ndarray:
        """元音幅度包络：快起音、平稳段、缓释放。"""
        attack = min(n // 3, max(1, int(sr * 0.014)))
        release = min(n - attack, max(1, int(sr * 0.030)))
        env = np.ones(n, dtype=np.float32)
        if attack > 1:
            env[:attack] = 0.5 - 0.5 * np.cos(
                np.linspace(0.0, np.pi, attack, dtype=np.float32)
            )
        if release > 1:
            env[n - release:] = 0.5 + 0.5 * np.cos(
                np.linspace(0.0, np.pi, release, dtype=np.float32)
            )
        return env

    def _fade_edges(self, audio: np.ndarray, sr: int, fade_ms: int) -> np.ndarray:
        """首尾淡入淡出，消除拼接处的爆音。"""
        n = audio.size
        fade = min(n // 2, max(1, int(sr * fade_ms / 1000.0)))
        if fade <= 1 or n < 4:
            return audio
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        audio[:fade] *= ramp
        audio[n - fade:] *= ramp[::-1]
        return audio

"""零依赖确定性语音合成（纯 numpy 共振峰合成）。

它不是"能听懂内容"的 TTS，而是**真实可听、韵律自然、口型清晰**的伪语音：
- 按字符映射到元音/辅音的共振峰参数，逐音素做源-滤波器合成；
- 浊音用带抖动的脉冲串作激励，清音用带通噪声；
- 叠加句子级基频轮廓（陈述句下降、疑问句上扬）与音节包络。

结果完全确定：相同文本 → 相同波形，便于演示、回归与延迟看板。
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import AsyncIterator, Iterable

import numpy as np

__all__ = [
    "PHONEME_TABLE",
    "estimate_duration_ms",
    "synthesize",
    "synthesize_stream",
    "text_to_phonemes",
]

#: 音素 → (F1, F2, F3, 浊音标志, 相对时长, 相对音量)
PHONEME_TABLE: dict[str, tuple[float, float, float, bool, float, float]] = {
    "a": (730.0, 1090.0, 2440.0, True, 1.00, 1.00),
    "e": (530.0, 1840.0, 2480.0, True, 0.95, 0.92),
    "i": (270.0, 2290.0, 3010.0, True, 0.85, 0.85),
    "o": (570.0, 840.0, 2410.0, True, 1.00, 0.95),
    "u": (300.0, 870.0, 2240.0, True, 0.90, 0.88),
    "v": (290.0, 1900.0, 2200.0, True, 0.90, 0.85),
    "m": (280.0, 900.0, 2200.0, True, 0.55, 0.60),
    "n": (320.0, 1400.0, 2600.0, True, 0.55, 0.60),
    "l": (360.0, 1300.0, 2700.0, True, 0.55, 0.65),
    "r": (420.0, 1300.0, 1600.0, True, 0.55, 0.65),
    "s": (0.0, 5500.0, 7000.0, False, 0.65, 0.35),
    "sh": (0.0, 3000.0, 5000.0, False, 0.70, 0.38),
    "f": (0.0, 4000.0, 6500.0, False, 0.60, 0.28),
    "h": (0.0, 1800.0, 3200.0, False, 0.50, 0.25),
    "t": (0.0, 3800.0, 5200.0, False, 0.30, 0.30),
    "k": (0.0, 2200.0, 3500.0, False, 0.30, 0.30),
    "p": (0.0, 1200.0, 2400.0, False, 0.28, 0.28),
    "z": (250.0, 4200.0, 6000.0, True, 0.55, 0.40),
    "j": (280.0, 2600.0, 3400.0, True, 0.55, 0.45),
    "w": (330.0, 700.0, 2200.0, True, 0.50, 0.60),
    "y": (280.0, 2200.0, 3000.0, True, 0.50, 0.60),
    "-": (0.0, 0.0, 0.0, False, 0.45, 0.00),   # 停顿
}

_VOWELS = ("a", "e", "i", "o", "u", "v")
_CONSONANTS = ("m", "n", "l", "r", "s", "sh", "f", "h", "t", "k", "p", "z", "j", "w", "y")
_PAUSE_CHARS = set(" \t\n，,。.！!？?；;：:、…—“”\"'()（）")
_QUESTION_CHARS = set("?？")


def _stable_index(ch: str, modulo: int) -> int:
    """字符 → 稳定索引。用哈希保证跨进程、跨平台一致。"""
    digest = hashlib.blake2b(ch.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % modulo


def text_to_phonemes(text: str) -> list[str]:
    """把任意文本确定性地映射为音素序列。

    ASCII 字母直接按发音近似归类；中文/其他字符按稳定哈希映射为"辅音+元音"音节。
    """
    out: list[str] = []
    for ch in text:
        if ch in _PAUSE_CHARS:
            if out and out[-1] != "-":
                out.append("-")
            continue
        low = ch.lower()
        if low in PHONEME_TABLE and low != "-":
            out.append(low)
            if low in _CONSONANTS:
                out.append(_VOWELS[_stable_index(ch, len(_VOWELS))])
            continue
        if low.isascii() and low.isalpha():
            out.append(_CONSONANTS[_stable_index(low, len(_CONSONANTS))])
            out.append(_VOWELS[_stable_index(low + "v", len(_VOWELS))])
            continue
        if low.isdigit():
            out.append(_CONSONANTS[_stable_index(low, len(_CONSONANTS))])
            out.append(_VOWELS[_stable_index(low, len(_VOWELS))])
            continue
        # 中文与其他 Unicode：一字一音节（声母 + 韵母）
        out.append(_CONSONANTS[_stable_index(ch, len(_CONSONANTS))])
        out.append(_VOWELS[_stable_index(ch + "#", len(_VOWELS))])
    if not out:
        out = ["a"]
    while out and out[-1] == "-":
        out.pop()
    return out or ["a"]


def estimate_duration_ms(text: str, *, speed: float = 1.0) -> int:
    """预估合成时长（毫秒），供延迟看板与预算计算使用。"""
    phonemes = text_to_phonemes(text)
    total = sum(PHONEME_TABLE[p][4] for p in phonemes) * 95.0
    return int(total / max(0.1, speed))


def _resonator(excitation: np.ndarray, freq: float, bandwidth: float,
               sample_rate: int) -> np.ndarray:
    """二阶谐振器（IIR）。经典 Klatt 共振峰滤波器实现。"""
    if freq <= 0.0:
        return excitation
    r = math.exp(-math.pi * bandwidth / sample_rate)
    theta = 2.0 * math.pi * freq / sample_rate
    a1 = 2.0 * r * math.cos(theta)
    a2 = -(r * r)
    b0 = 1.0 - a1 - a2
    out = np.zeros_like(excitation)
    y1 = 0.0
    y2 = 0.0
    for i in range(excitation.size):
        y = b0 * float(excitation[i]) + a1 * y1 + a2 * y2
        out[i] = y
        y2 = y1
        y1 = y
    return out


def _glottal_pulse(n_samples: int, f0_curve: np.ndarray, sample_rate: int,
                   rng: np.random.Generator) -> np.ndarray:
    """声门脉冲串激励，带轻微频率抖动（jitter）与幅度抖动（shimmer）。"""
    out = np.zeros(n_samples, dtype=np.float32)
    phase = 0.0
    for i in range(n_samples):
        f0 = float(f0_curve[i]) * (1.0 + 0.008 * float(rng.standard_normal()))
        phase += f0 / sample_rate
        if phase >= 1.0:
            phase -= 1.0
            amp = 1.0 + 0.05 * float(rng.standard_normal())
            out[i] = amp
    # 脉冲整形：单极点低通，模拟声门波的斜坡
    alpha = 0.94
    prev = 0.0
    for i in range(n_samples):
        prev = out[i] + alpha * prev
        out[i] = prev
    return out


def _noise_source(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n_samples).astype(np.float32) * 0.35


def _envelope(n_samples: int, attack_ratio: float = 0.18,
              release_ratio: float = 0.28) -> np.ndarray:
    """音素包络，避免逐音素拼接产生咔哒声。"""
    env = np.ones(n_samples, dtype=np.float32)
    attack = max(1, int(n_samples * attack_ratio))
    release = max(1, int(n_samples * release_ratio))
    env[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
    env[-release:] = np.linspace(1.0, 0.0, release, dtype=np.float32)
    return env


def _f0_contour(n_samples: int, base_f0: float, question: bool,
                position: float) -> np.ndarray:
    """句子级基频轮廓：陈述句整体下降，疑问句结尾上扬。"""
    t = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    if question:
        curve = base_f0 * (1.0 + 0.22 * position * t)
    else:
        curve = base_f0 * (1.0 - 0.16 * (position * 0.7 + t * 0.3))
    # 音节内轻微起伏，让语音不呆板
    curve = curve * (1.0 + 0.03 * np.sin(2.0 * math.pi * t))
    return curve.astype(np.float32)


def synthesize(text: str, *, sample_rate: int = 24000, speed: float = 1.0,
               base_f0: float = 165.0, seed: int | None = None,
               gain: float = 0.62) -> np.ndarray:
    """合成整段语音，返回 float32 单声道波形，取值 [-1, 1]。

    Args:
        text: 待合成文本，支持中英文混排。
        sample_rate: 输出采样率。
        speed: 语速倍率，>1 更快。
        base_f0: 基频（Hz）。男声约 110，女声约 200。
        seed: 随机种子。默认由文本内容派生，保证确定性。
        gain: 输出增益。

    Returns:
        ``np.ndarray[float32]`` 波形。
    """
    if not text or not text.strip():
        return np.zeros(0, dtype=np.float32)

    if seed is None:
        seed = int.from_bytes(
            hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest(), "big")
    rng = np.random.default_rng(seed)

    phonemes = text_to_phonemes(text)
    question = bool(_QUESTION_CHARS & set(text))
    segments: list[np.ndarray] = []
    total = len(phonemes)

    for idx, ph in enumerate(phonemes):
        f1, f2, f3, voiced, rel_dur, rel_amp = PHONEME_TABLE[ph]
        dur_ms = max(12.0, rel_dur * 95.0 / max(0.1, speed))
        n = max(8, int(sample_rate * dur_ms / 1000.0))
        position = idx / max(1, total - 1) if total > 1 else 1.0

        if rel_amp <= 0.0:                      # 停顿
            segments.append(np.zeros(n, dtype=np.float32))
            continue

        if voiced:
            f0_curve = _f0_contour(n, base_f0, question, position)
            excitation = _glottal_pulse(n, f0_curve, sample_rate, rng)
        else:
            excitation = _noise_source(n, rng)

        wave = _resonator(excitation, f1, 80.0, sample_rate)
        wave = wave + 0.60 * _resonator(excitation, f2, 110.0, sample_rate)
        wave = wave + 0.28 * _resonator(excitation, f3, 160.0, sample_rate)
        wave = wave * _envelope(n) * rel_amp
        segments.append(wave.astype(np.float32))

    audio = np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)

    # 归一化 + 软限幅，保证音量稳定且不削顶
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-6:
        audio = audio / peak * gain
    audio = np.tanh(audio * 1.15).astype(np.float32)

    # 首尾各 10ms 淡入淡出
    fade = min(int(sample_rate * 0.01), audio.size // 2)
    if fade > 0:
        audio[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return audio


def to_pcm16(wave: np.ndarray) -> bytes:
    """float32 [-1,1] → 小端 int16 字节串。"""
    clipped = np.clip(wave, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def iter_frames(wave: np.ndarray, *, sample_rate: int = 24000,
                frame_ms: int = 20) -> Iterable[bytes]:
    """把波形切成定长 PCM16 帧（不足部分补零）。"""
    frame_samples = max(1, sample_rate * frame_ms // 1000)
    total = int(math.ceil(wave.size / frame_samples)) if wave.size else 0
    for i in range(total):
        chunk = wave[i * frame_samples:(i + 1) * frame_samples]
        if chunk.size < frame_samples:
            chunk = np.pad(chunk, (0, frame_samples - chunk.size))
        yield to_pcm16(chunk)


async def synthesize_stream(text: str, *, sample_rate: int = 24000,
                            frame_ms: int = 20, speed: float = 1.0,
                            base_f0: float = 165.0,
                            seed: int | None = None) -> AsyncIterator[bytes]:
    """流式产出 PCM16 帧。

    这是 :class:`~omnistream.providers.mock.mock_provider.MockRealtimeProvider`
    依赖的接口：每次 ``yield`` 一个 ``frame_ms`` 长度的小端 int16 字节串。
    """
    wave = synthesize(text, sample_rate=sample_rate, speed=speed,
                      base_f0=base_f0, seed=seed)
    for frame in iter_frames(wave, sample_rate=sample_rate, frame_ms=frame_ms):
        yield frame

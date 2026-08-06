"""音频格式描述与转换矩阵。

覆盖 ``pcm16 / pcm_f32 / opus / mulaw`` × ``8k / 16k / 22.05k / 24k / 44.1k / 48k``。

这里只做**描述与可行性判定**，实际转换由 :mod:`omnistream.media.audio.pcm`
（位深/声道）、:mod:`omnistream.media.audio.resampler`（采样率）、
:mod:`omnistream.media.audio.opus_codec` 与 :mod:`omnistream.media.audio.mulaw`（编解码）完成。
"""
from __future__ import annotations

from dataclasses import dataclass

from omnistream.media.constants import (
    DEFAULT_CHANNELS,
    DEFAULT_FRAME_MS,
    DEFAULT_INPUT_SAMPLE_RATE,
    SUPPORTED_SAMPLE_RATES,
)
from omnistream.media.errors import ResampleError
from omnistream.protocol.enums import AudioEncoding
from omnistream.schemas.audio import AudioFormat

__all__ = [
    "COMPRESSED_ENCODINGS",
    "ConversionPlan",
    "OPUS_SAMPLE_RATES",
    "RAW_ENCODINGS",
    "bytes_per_frame",
    "bytes_per_sample",
    "can_convert",
    "describe",
    "duration_ms_of",
    "format_from_header",
    "header_of",
    "is_raw_pcm",
    "make_format",
    "plan_conversion",
    "samples_for_ms",
    "validate_format",
]

#: 未压缩（可直接 ndarray 化）的编码。
RAW_ENCODINGS: frozenset[AudioEncoding] = frozenset({AudioEncoding.PCM16, AudioEncoding.PCM_F32})

#: 需要先解码的编码。
COMPRESSED_ENCODINGS: frozenset[AudioEncoding] = frozenset({AudioEncoding.OPUS, AudioEncoding.MULAW})

#: libopus 原生支持的采样率。
OPUS_SAMPLE_RATES: tuple[int, ...] = (8000, 12000, 16000, 24000, 48000)

#: μ-law 在电话网中固定 8k 单声道，但本实现允许任意采样率（仅做位压缩）。
MULAW_CANONICAL_RATE: int = 8000


def bytes_per_sample(encoding: AudioEncoding) -> int:
    """单声道单样本的字节数（压缩编码返回 0，表示不定长）。"""
    if encoding == AudioEncoding.PCM_F32:
        return 4
    if encoding == AudioEncoding.PCM16:
        return 2
    if encoding == AudioEncoding.MULAW:
        return 1
    return 0


def is_raw_pcm(encoding: AudioEncoding) -> bool:
    """是否为可直接解释为样本数组的裸 PCM。"""
    return encoding in RAW_ENCODINGS


def samples_for_ms(sample_rate: int, duration_ms: float) -> int:
    """给定采样率与时长，返回样本数（向下取整）。"""
    return int(sample_rate * duration_ms / 1000.0)


def bytes_per_frame(fmt: AudioFormat, frame_ms: int = DEFAULT_FRAME_MS) -> int:
    """一帧（默认 20ms）的字节数。压缩编码返回 0。"""
    width = bytes_per_sample(fmt.encoding)
    if width == 0:
        return 0
    return samples_for_ms(fmt.sample_rate, frame_ms) * width * fmt.channels


def duration_ms_of(data: bytes, fmt: AudioFormat) -> float:
    """根据字节长度推算时长（毫秒）。压缩编码无法推算，返回 0.0。"""
    width = bytes_per_sample(fmt.encoding)
    if width == 0 or fmt.channels <= 0 or fmt.sample_rate <= 0:
        return 0.0
    samples = len(data) // (width * fmt.channels)
    return samples * 1000.0 / fmt.sample_rate


def make_format(
    encoding: AudioEncoding = AudioEncoding.PCM16,
    sample_rate: int = DEFAULT_INPUT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
) -> AudioFormat:
    """构造并校验一个 :class:`AudioFormat`。"""
    fmt = AudioFormat(encoding=encoding, sample_rate=sample_rate, channels=channels)
    validate_format(fmt)
    return fmt


def validate_format(fmt: AudioFormat) -> None:
    """校验格式组合是否可被本层处理。

    Raises:
        ResampleError: 采样率/声道/编码组合非法。
    """
    if fmt.sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise ResampleError(
            f"unsupported sample rate: {fmt.sample_rate}",
            detail={"supported": list(SUPPORTED_SAMPLE_RATES)},
        )
    if fmt.channels not in (1, 2):
        raise ResampleError(
            f"unsupported channel count: {fmt.channels}", detail={"supported": [1, 2]}
        )
    if fmt.encoding == AudioEncoding.OPUS and fmt.sample_rate not in OPUS_SAMPLE_RATES:
        raise ResampleError(
            f"opus does not support sample rate {fmt.sample_rate}",
            detail={"supported": list(OPUS_SAMPLE_RATES)},
        )


def can_convert(src: AudioFormat, dst: AudioFormat) -> bool:
    """判断 ``src → dst`` 是否可行（不考虑可选依赖是否安装）。"""
    try:
        validate_format(src)
        validate_format(dst)
    except ResampleError:
        return False
    return True


@dataclass(slots=True, frozen=True)
class ConversionPlan:
    """一次格式转换需要执行的步骤序列。

    Attributes:
        decode: 是否需要先解码为 PCM16。
        resample: 是否需要采样率转换。
        remix: 是否需要声道重排。
        requantize: 是否需要位深转换（pcm16 ↔ pcm_f32）。
        encode: 是否需要目标编码。
        src: 源格式。
        dst: 目标格式。
    """

    src: AudioFormat
    dst: AudioFormat
    decode: bool = False
    resample: bool = False
    remix: bool = False
    requantize: bool = False
    encode: bool = False

    @property
    def is_passthrough(self) -> bool:
        """完全无需处理时为 True（直通，零拷贝路径）。"""
        return not (self.decode or self.resample or self.remix or self.requantize or self.encode)

    @property
    def steps(self) -> list[str]:
        """人类可读的步骤列表，用于日志与文档。"""
        out: list[str] = []
        if self.decode:
            out.append(f"decode:{self.src.encoding}")
        if self.remix:
            out.append(f"remix:{self.src.channels}->{self.dst.channels}")
        if self.resample:
            out.append(f"resample:{self.src.sample_rate}->{self.dst.sample_rate}")
        if self.requantize:
            out.append(f"requantize:->{self.dst.encoding}")
        if self.encode:
            out.append(f"encode:{self.dst.encoding}")
        return out


def plan_conversion(src: AudioFormat, dst: AudioFormat) -> ConversionPlan:
    """计算 ``src → dst`` 的转换计划。

    Raises:
        ResampleError: 格式非法。
    """
    validate_format(src)
    validate_format(dst)
    decode = src.encoding in COMPRESSED_ENCODINGS
    encode = dst.encoding in COMPRESSED_ENCODINGS
    remix = src.channels != dst.channels
    resample = src.sample_rate != dst.sample_rate
    requantize = (
        not decode
        and not encode
        and src.encoding != dst.encoding
    )
    return ConversionPlan(
        src=src,
        dst=dst,
        decode=decode,
        resample=resample,
        remix=remix,
        requantize=requantize,
        encode=encode,
    )


def header_of(fmt: AudioFormat) -> dict[str, object]:
    """转成二进制帧头字段（与 ``protocol/binary.py`` 的 audio header 对齐）。"""
    return {"sr": fmt.sample_rate, "ch": fmt.channels, "enc": str(fmt.encoding)}


def format_from_header(header: dict[str, object]) -> AudioFormat:
    """从二进制帧头还原 :class:`AudioFormat`。

    未携带的字段回落到默认值，非法编码回落到 pcm16（宽容解析，避免单帧毁掉整条流）。
    """
    raw_enc = str(header.get("enc", AudioEncoding.PCM16.value))
    try:
        encoding = AudioEncoding(raw_enc)
    except ValueError:
        encoding = AudioEncoding.PCM16
    try:
        sample_rate = int(header.get("sr", DEFAULT_INPUT_SAMPLE_RATE))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        sample_rate = DEFAULT_INPUT_SAMPLE_RATE
    try:
        channels = int(header.get("ch", DEFAULT_CHANNELS))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        channels = DEFAULT_CHANNELS
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        sample_rate = DEFAULT_INPUT_SAMPLE_RATE
    if channels not in (1, 2):
        channels = DEFAULT_CHANNELS
    return AudioFormat(encoding=encoding, sample_rate=sample_rate, channels=channels)


def describe(fmt: AudioFormat) -> str:
    """紧凑的可读描述，如 ``pcm16/16000Hz/mono``。"""
    layout = {1: "mono", 2: "stereo"}.get(fmt.channels, f"{fmt.channels}ch")
    return f"{fmt.encoding}/{fmt.sample_rate}Hz/{layout}"

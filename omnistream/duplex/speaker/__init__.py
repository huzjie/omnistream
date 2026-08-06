"""说话人子包：轻量切换检测 + 标签维护 + 轮次归属。

典型装配::

    diarizer = LightweightDiarizer(sample_rate=16000)
    tracker = SpeakerTracker(session_id=sid)

    ev = diarizer.observe(chunk, is_speech=vad_result.is_speech)
    if ev is not None:
        switched = tracker.on_event(ev)
        if ev.kind is DiarizationEventKind.SPEAKER_ENDED:
            seg = diarizer.flush()
            if seg is not None:
                tracker.on_segment(seg)

轮次结束时 ``tracker.attribute_turn(turn_id)`` 给出该轮的说话人归属，
``is_primary=False`` 的轮次不会被写入长期记忆（防止旁人污染画像）。
"""
from __future__ import annotations

from omnistream.duplex.speaker.diarization import (
    DEFAULT_NUM_BANDS,
    SPEAKER_PREFIX,
    UNKNOWN_SPEAKER,
    DiarizationEvent,
    DiarizationEventKind,
    LightweightDiarizer,
    SpeakerSegment,
    cosine_distance,
    estimate_f0,
    extract_embedding,
)
from omnistream.duplex.speaker.tracker import (
    DEFAULT_MIN_SWITCH_MS,
    DEFAULT_PRIMARY_MIN_MS,
    SpeakerProfile,
    SpeakerTracker,
    TurnAttribution,
)

__all__ = [
    "DEFAULT_MIN_SWITCH_MS",
    "DEFAULT_NUM_BANDS",
    "DEFAULT_PRIMARY_MIN_MS",
    "SPEAKER_PREFIX",
    "UNKNOWN_SPEAKER",
    "DiarizationEvent",
    "DiarizationEventKind",
    "LightweightDiarizer",
    "SpeakerProfile",
    "SpeakerSegment",
    "SpeakerTracker",
    "TurnAttribution",
    "cosine_distance",
    "estimate_f0",
    "extract_embedding",
]

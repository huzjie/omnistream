"""实时事件协议枚举。客户端与服务端严格按此表通信；TS SDK 类型由此生成。"""
from __future__ import annotations

from enum import StrEnum

PROTOCOL_VERSION = "oms.v1"


class Modality(StrEnum):
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"


class AudioEncoding(StrEnum):
    PCM16 = "pcm16"          # 小端 int16
    PCM_F32 = "pcm_f32"
    OPUS = "opus"
    MULAW = "mulaw"


class VideoEncoding(StrEnum):
    JPEG = "jpeg"
    WEBP = "webp"
    RGB24 = "rgb24"


class SessionState(StrEnum):
    CREATED = "created"
    ACTIVE = "active"
    ENDING = "ending"
    ENDED = "ended"
    FAILED = "failed"


class TurnState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


class TurnRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class StopReason(StrEnum):
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    ERROR = "error"
    MAX_DURATION = "max_duration"
    TOOL_PENDING = "tool_pending"


class ToolCallState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class MemoryLayer(StrEnum):
    SESSION = "session"
    PRIVATE_STATE = "private_state"
    STABLE_KNOWLEDGE = "stable_knowledge"
    PROCEDURAL = "procedural"


class ClientEventType(StrEnum):
    """上行（客户端 → 服务端）事件。"""
    SESSION_UPDATE = "session.update"
    INPUT_AUDIO_APPEND = "input_audio.append"
    INPUT_AUDIO_COMMIT = "input_audio.commit"
    INPUT_AUDIO_CLEAR = "input_audio.clear"
    INPUT_VIDEO_FRAME = "input_video.frame"
    INPUT_TEXT = "input_text.send"
    RESPONSE_CREATE = "response.create"
    RESPONSE_CANCEL = "response.cancel"
    TOOL_RESULT = "tool.result"
    MEMORY_UPSERT = "memory.upsert"
    MEMORY_DELETE = "memory.delete"
    PING = "ping"
    SESSION_CLOSE = "session.close"


class ServerEventType(StrEnum):
    """下行（服务端 → 客户端）事件。"""
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    SESSION_CLOSED = "session.closed"

    TURN_STARTED = "turn.started"
    TURN_STATE = "turn.state"
    TURN_COMPLETED = "turn.completed"

    SPEECH_STARTED = "input_audio.speech_started"
    SPEECH_STOPPED = "input_audio.speech_stopped"

    TRANSCRIPT_DELTA = "transcript.delta"
    TRANSCRIPT_DONE = "transcript.done"

    RESPONSE_TEXT_DELTA = "response.text.delta"
    RESPONSE_TEXT_DONE = "response.text.done"
    RESPONSE_AUDIO_DELTA = "response.audio.delta"
    RESPONSE_AUDIO_DONE = "response.audio.done"
    RESPONSE_DONE = "response.done"
    RESPONSE_CANCELLED = "response.cancelled"

    BARGE_IN = "barge_in.detected"
    PLAYBACK_TRUNCATED = "playback.truncated"

    VISION_OBSERVATION = "vision.observation"
    VISION_FRAME_ACCEPTED = "vision.frame_accepted"

    TOOL_CALL = "tool.call"
    TOOL_CALL_DONE = "tool.call_done"

    MEMORY_INJECTED = "memory.injected"
    MEMORY_SILENT = "memory.silent"

    METRICS_TICK = "metrics.tick"
    RATE_LIMIT = "rate_limit.updated"
    PONG = "pong"
    ERROR = "error"

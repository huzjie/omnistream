"""说话人标签维护与轮次归属。

:class:`~omnistream.duplex.speaker.diarization.LightweightDiarizer` 只回答
"声音换人了吗"，:class:`SpeakerTracker` 负责把这个信号变成**业务可用的身份**：

* **主说话人（primary）**：累计说话时长最长的那位，默认被视为"用户本人"，
  记忆写入、偏好学习只认他，避免旁人一句话污染长期记忆；
* **稳定标签**：对外暴露 ``spk_1`` / ``spk_2``…（可绑定真实 ``user_id``），
  即使 diarizer 内部重新聚类，对外标签也保持连续；
* **轮次归属**：一轮里可能有多人说话，按有效语音时长投票决定归属；
* **切换节流**：短于 ``min_switch_ms`` 的插入不触发对外的"说话人已切换"事件，
  防止咳嗽/附和造成标签抖动。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import structlog

from omnistream.duplex.speaker.diarization import (
    UNKNOWN_SPEAKER,
    DiarizationEvent,
    DiarizationEventKind,
    SpeakerSegment,
)

log = structlog.get_logger(__name__)

#: 低于该时长的说话人切换不对外通告
DEFAULT_MIN_SWITCH_MS: Final[int] = 350

#: 主说话人判定所需的最小累计时长
DEFAULT_PRIMARY_MIN_MS: Final[int] = 1500


@dataclass(slots=True)
class SpeakerProfile:
    """对外可见的说话人档案。"""

    label: str
    total_ms: float = 0.0
    segments: int = 0
    turns: int = 0
    first_seen_ms: int = 0
    last_seen_ms: int = 0
    #: 业务侧绑定的真实身份（登录用户 id / 座席工号等）
    identity: str | None = None
    display_name: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "identity": self.identity,
            "display_name": self.display_name,
            "total_ms": round(self.total_ms, 1),
            "segments": self.segments,
            "turns": self.turns,
            "first_seen_ms": self.first_seen_ms,
            "last_seen_ms": self.last_seen_ms,
        }


@dataclass(slots=True)
class TurnAttribution:
    """一轮的说话人归属结果。"""

    turn_id: str
    speaker: str
    #: 该说话人在本轮占的语音时长
    speaker_ms: float = 0.0
    #: 本轮总语音时长
    total_ms: float = 0.0
    #: 本轮出现过的所有说话人及其时长
    participants: dict[str, float] = field(default_factory=dict)
    is_primary: bool = True
    switched: bool = False

    @property
    def dominance(self) -> float:
        """归属说话人的占比 [0,1]。低于 0.6 说明本轮存在明显交叉说话。"""
        if self.total_ms <= 0.0:
            return 1.0
        return float(min(1.0, self.speaker_ms / self.total_ms))

    def as_dict(self) -> dict[str, object]:
        return {
            "turn_id": self.turn_id,
            "speaker": self.speaker,
            "speaker_ms": round(self.speaker_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "dominance": round(self.dominance, 3),
            "participants": {k: round(v, 1) for k, v in self.participants.items()},
            "is_primary": self.is_primary,
            "switched": self.switched,
        }


class SpeakerTracker:
    """说话人标签维护器。"""

    def __init__(
        self,
        *,
        session_id: str = "",
        min_switch_ms: int = DEFAULT_MIN_SWITCH_MS,
        primary_min_ms: int = DEFAULT_PRIMARY_MIN_MS,
        single_speaker_mode: bool = False,
    ) -> None:
        self.session_id = session_id
        self.min_switch_ms = int(min_switch_ms)
        self.primary_min_ms = int(primary_min_ms)
        #: 单人模式：所有声音都归为 ``spk_1``（一对一场景可省掉抖动风险）
        self.single_speaker_mode = bool(single_speaker_mode)

        self._profiles: dict[str, SpeakerProfile] = {}
        self._active: str | None = None
        self._primary: str | None = None
        self._turn_participants: dict[str, float] = {}
        self._turn_switched = False
        self._pending_switch: tuple[str, int] | None = None
        self._switch_count = 0

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def active_speaker(self) -> str:
        """当前正在说话的人（无人说话时返回上一位或 unknown）。"""
        if self.single_speaker_mode:
            return "spk_1"
        return self._active or self._primary or UNKNOWN_SPEAKER

    @property
    def primary_speaker(self) -> str:
        """主说话人（累计时长最长者）。"""
        if self.single_speaker_mode:
            return "spk_1"
        return self._primary or UNKNOWN_SPEAKER

    @property
    def switch_count(self) -> int:
        return self._switch_count

    def profiles(self) -> dict[str, dict[str, object]]:
        return {label: p.as_dict() for label, p in self._profiles.items()}

    def is_primary(self, label: str) -> bool:
        """该标签是否为主说话人（决定能否写入长期记忆）。"""
        if self.single_speaker_mode:
            return True
        return label == self._primary

    # ------------------------------------------------------------------
    # 身份绑定
    # ------------------------------------------------------------------

    def bind_identity(self, label: str, identity: str, *, display_name: str | None = None) -> None:
        """把 diarizer 标签绑定到业务身份。"""
        profile = self._ensure(label)
        profile.identity = identity
        profile.display_name = display_name or identity
        log.info("speaker.identity_bound", session_id=self.session_id,
                 label=label, identity=identity)

    def identity_of(self, label: str) -> str:
        """返回可展示的身份名，未绑定时回落到标签本身。"""
        profile = self._profiles.get(label)
        if profile is None:
            return label
        return profile.display_name or profile.identity or label

    # ------------------------------------------------------------------
    # 事件消费
    # ------------------------------------------------------------------

    def on_event(self, event: DiarizationEvent) -> bool:
        """消费 diarizer 事件；返回是否发生了**对外可见**的说话人切换。"""
        if self.single_speaker_mode:
            self._ensure("spk_1")
            self._active = "spk_1"
            return False

        profile = self._ensure(event.speaker)
        if profile.first_seen_ms == 0:
            profile.first_seen_ms = event.pts_ms
        profile.last_seen_ms = max(profile.last_seen_ms, event.pts_ms)

        if event.kind is DiarizationEventKind.SPEAKER_ENDED:
            self._pending_switch = None
            return False

        if event.speaker == self._active:
            self._pending_switch = None
            return False

        # 切换节流：先记 pending，等到累计足够时长再对外通告
        if self._pending_switch is None or self._pending_switch[0] != event.speaker:
            self._pending_switch = (event.speaker, event.pts_ms)
            if self._active is None:
                return self._commit_switch(event.speaker, event.pts_ms)
            return False

        started_ms = self._pending_switch[1]
        if event.pts_ms - started_ms < self.min_switch_ms:
            return False
        return self._commit_switch(event.speaker, event.pts_ms)

    def _commit_switch(self, label: str, pts_ms: int) -> bool:
        previous = self._active
        self._active = label
        self._pending_switch = None
        if previous is not None and previous != label:
            self._switch_count += 1
            log.info("speaker.switched", session_id=self.session_id,
                     previous=previous, current=label, pts_ms=pts_ms)
            self._turn_switched = True
            return True
        return False

    def on_segment(self, segment: SpeakerSegment) -> None:
        """消费一段完整语音，累计时长并更新主说话人。"""
        label = "spk_1" if self.single_speaker_mode else segment.speaker
        profile = self._ensure(label)
        profile.total_ms += float(segment.duration_ms)
        profile.segments += 1
        profile.last_seen_ms = max(profile.last_seen_ms, segment.end_ms)
        self._turn_participants[label] = (
            self._turn_participants.get(label, 0.0) + float(segment.duration_ms)
        )
        self._recompute_primary()

    def observe_speech(self, label: str, duration_ms: float) -> None:
        """逐帧累计语音时长（不产生 segment 时的轻量入口）。"""
        target = "spk_1" if self.single_speaker_mode else label
        if target == UNKNOWN_SPEAKER:
            return
        profile = self._ensure(target)
        profile.total_ms += float(duration_ms)
        self._turn_participants[target] = (
            self._turn_participants.get(target, 0.0) + float(duration_ms)
        )
        self._recompute_primary()

    def _recompute_primary(self) -> None:
        if not self._profiles:
            return
        best = max(self._profiles.values(), key=lambda p: p.total_ms)
        if best.total_ms >= self.primary_min_ms or self._primary is None:
            if self._primary != best.label:
                log.debug("speaker.primary_updated", session_id=self.session_id,
                          primary=best.label, total_ms=round(best.total_ms, 1))
            self._primary = best.label

    # ------------------------------------------------------------------
    # 轮次归属
    # ------------------------------------------------------------------

    def begin_turn(self) -> None:
        """开启新轮次的归属统计。"""
        self._turn_participants = {}
        self._turn_switched = False

    def attribute_turn(self, turn_id: str) -> TurnAttribution:
        """按本轮各说话人的语音时长投票，产出归属结果。"""
        participants = dict(self._turn_participants)
        total = float(sum(participants.values()))
        if not participants:
            speaker = self.active_speaker
            speaker_ms = 0.0
        else:
            speaker, speaker_ms = max(participants.items(), key=lambda kv: kv[1])

        profile = self._profiles.get(speaker)
        if profile is not None:
            profile.turns += 1

        attribution = TurnAttribution(
            turn_id=turn_id,
            speaker=speaker,
            speaker_ms=speaker_ms,
            total_ms=total,
            participants=participants,
            is_primary=self.is_primary(speaker),
            switched=self._turn_switched,
        )
        self._turn_participants = {}
        self._turn_switched = False
        log.debug("speaker.turn_attributed", session_id=self.session_id,
                  **attribution.as_dict())
        return attribution

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _ensure(self, label: str) -> SpeakerProfile:
        profile = self._profiles.get(label)
        if profile is None:
            profile = SpeakerProfile(label=label)
            self._profiles[label] = profile
        return profile

    def reset(self, *, keep_profiles: bool = False) -> None:
        """会话复位。"""
        self._active = None
        self._pending_switch = None
        self._turn_participants = {}
        self._turn_switched = False
        if not keep_profiles:
            self._profiles.clear()
            self._primary = None
            self._switch_count = 0

    def stats(self) -> dict[str, object]:
        return {
            "active": self.active_speaker,
            "primary": self.primary_speaker,
            "switch_count": self._switch_count,
            "profiles": self.profiles(),
        }

    def __repr__(self) -> str:
        return (f"<SpeakerTracker active={self.active_speaker} "
                f"primary={self.primary_speaker} switches={self._switch_count}>")


__all__ = [
    "DEFAULT_MIN_SWITCH_MS",
    "DEFAULT_PRIMARY_MIN_MS",
    "SpeakerProfile",
    "SpeakerTracker",
    "TurnAttribution",
]

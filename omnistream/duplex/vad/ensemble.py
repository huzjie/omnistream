"""VADEnsemble —— 多后端加权投票 + 迟滞（hangover）平滑。

为什么需要 Ensemble
-------------------
* Silero 准但偶尔在爆破音上有 1~2 帧空洞
* webrtcvad 灵敏但对键盘声/呼吸声误报高
* Energy 稳定但对低音量说话不敏感

加权投票 + hangover 能同时压掉"空洞"与"毛刺"：

* **hangover**：一旦判定为语音，后续 N 帧即使概率下滑也保持语音态（补空洞）
* **onset**：需要连续 M 帧超阈值才认定语音起点（压毛刺）

Ensemble 自身也是一个 :class:`BaseVAD`，可以被 EndpointDetector / BargeInDetector 直接消费。
"""
from __future__ import annotations

from dataclasses import dataclass

import structlog

from omnistream.duplex.vad.base import DEFAULT_FRAME_MS, DEFAULT_SAMPLE_RATE, BaseVAD, VADResult
from omnistream.schemas.audio import AudioChunk

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class WeightedVAD:
    """带权重的后端条目。"""

    vad: BaseVAD
    weight: float = 1.0


class VADEnsemble(BaseVAD):
    """多后端投票 VAD。"""

    name = "ensemble"

    def __init__(
        self,
        members: list[WeightedVAD],
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_ms: int = DEFAULT_FRAME_MS,
        threshold: float = 0.5,
        onset_frames: int = 2,
        hangover_frames: int = 8,
    ) -> None:
        if not members:
            raise ValueError("VADEnsemble 至少需要一个后端")
        super().__init__(sample_rate=sample_rate, frame_ms=frame_ms, threshold=threshold)
        self.members = members
        self.onset_frames = max(1, int(onset_frames))
        self.hangover_frames = max(0, int(hangover_frames))
        self._total_weight = sum(max(0.0, m.weight) for m in members) or 1.0
        self._consecutive_above = 0
        self._hangover_left = 0
        self._state_speech = False

    # ------------------------------------------------------------------

    def reset(self) -> None:
        super().reset()
        for member in self.members:
            member.vad.reset()
        self._consecutive_above = 0
        self._hangover_left = 0
        self._state_speech = False

    def close(self) -> None:
        for member in self.members:
            member.vad.close()

    def process_frame(self, chunk: AudioChunk) -> VADResult:
        weighted_prob = 0.0
        energy_db = -120.0
        noise_db = -60.0
        details: dict[str, float] = {}

        for member in self.members:
            result = member.vad.process_frame(chunk)
            weight = max(0.0, member.weight)
            weighted_prob += result.prob * weight
            energy_db = max(energy_db, result.energy_db)
            noise_db = min(noise_db, result.noise_floor_db)
            details[f"p_{result.backend}"] = round(result.prob, 4)

        prob = weighted_prob / self._total_weight
        above = prob >= self.threshold

        if above:
            self._consecutive_above += 1
        else:
            self._consecutive_above = 0

        if not self._state_speech:
            if self._consecutive_above >= self.onset_frames:
                self._state_speech = True
                self._hangover_left = self.hangover_frames
        else:
            if above:
                self._hangover_left = self.hangover_frames
            elif self._hangover_left > 0:
                self._hangover_left -= 1
            else:
                self._state_speech = False

        details["raw_prob"] = round(prob, 4)
        details["hangover_left"] = float(self._hangover_left)
        self._account(self._state_speech)

        return VADResult(
            is_speech=self._state_speech,
            prob=float(prob),
            pts_ms=int(chunk.pts_ms),
            energy_db=energy_db,
            backend=self.name,
            duration_ms=float(chunk.duration_ms) or float(self.frame_ms),
            noise_floor_db=noise_db,
            details=details,
        )

    @property
    def backend_names(self) -> list[str]:
        return [m.vad.name for m in self.members]

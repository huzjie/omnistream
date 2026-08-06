"""全双工配置：VAD 后端与阈值、端点检测、打断灵敏度、轮次时长。

这些参数直接决定"抢话体验"：
- ``min_silence_ms`` 太大 → 助手反应迟钝；太小 → 用户还没说完就被抢答。
- ``barge_in_energy_db`` 太敏感 → 环境噪声误打断；太钝 → 打断不及时。
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from omnistream.core.constants import (
    BARGE_IN_BUDGET_MS,
    CANCEL_BUDGET_MS,
    DEFAULT_IDLE_TIMEOUT_MS,
    DEFAULT_MAX_TURN_MS,
    DEFAULT_MIN_SILENCE_MS,
    DEFAULT_MIN_SPEECH_MS,
    DEFAULT_PREFIX_PADDING_MS,
)


class DuplexSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OMS_DUPLEX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------ VAD
    vad_backend: str = Field(default="auto",
                             description="auto|silero|webrtc|energy|ensemble")
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    vad_frame_ms: int = Field(default=20, ge=10, le=30)
    #: energy 后端的绝对能量门限
    vad_energy_dbfs: float = Field(default=-45.0, le=0.0)
    #: silero 模型路径，留空则用包内自带/自动下载
    silero_model_path: str | None = None
    #: ensemble 模式下各后端权重
    ensemble_weights: dict[str, float] = Field(
        default_factory=lambda: {"silero": 0.6, "webrtc": 0.25, "energy": 0.15})

    # ------------------------------------------------------------ 端点检测
    min_speech_ms: int = Field(default=DEFAULT_MIN_SPEECH_MS, ge=20)
    min_silence_ms: int = Field(default=DEFAULT_MIN_SILENCE_MS, ge=100)
    prefix_padding_ms: int = Field(default=DEFAULT_PREFIX_PADDING_MS, ge=0)
    #: 自适应静音：根据用户语速与历史停顿动态调整 min_silence_ms
    adaptive_silence: bool = True
    adaptive_silence_min_ms: int = Field(default=280, ge=100)
    adaptive_silence_max_ms: int = Field(default=900, ge=200)
    #: 语义端点检测：用轻量模型判断"这句话说完了吗"
    semantic_endpointing: bool = False
    semantic_confidence: float = Field(default=0.7, ge=0.0, le=1.0)

    # ------------------------------------------------------------ 打断
    barge_in_enabled: bool = True
    barge_in_mode: str = Field(default="immediate",
                               description="immediate|finish_sentence|off")
    barge_in_energy_db: float = Field(default=-38.0, le=0.0)
    barge_in_min_duration_ms: int = Field(default=160, ge=40)
    barge_in_cooldown_ms: int = Field(default=700, ge=0)
    #: 从检测到打断到播放停止的预算，超出记为 SLO 违约
    barge_in_budget_ms: int = Field(default=BARGE_IN_BUDGET_MS, ge=50)
    #: provider.cancel_response() 的超时预算
    cancel_budget_ms: int = Field(default=CANCEL_BUDGET_MS, ge=20)
    #: 打断后是否把已播放部分回填给模型（让模型知道用户听到哪了）
    truncate_context_on_barge_in: bool = True
    #: 回声消除不可靠时，用于避免助手自己的声音触发打断
    echo_guard_ms: int = Field(default=120, ge=0)

    # ------------------------------------------------------------ 轮次
    max_turn_ms: int = Field(default=DEFAULT_MAX_TURN_MS, ge=1000)
    idle_timeout_ms: int = Field(default=DEFAULT_IDLE_TIMEOUT_MS, ge=1000)
    #: 主动开口：静默超过该时长后模型可主动发起
    proactive_enabled: bool = False
    proactive_silence_ms: int = Field(default=8_000, ge=1000)
    proactive_max_per_session: int = Field(default=3, ge=0)
    #: 一轮内最多允许的连续工具调用轮数，防止死循环
    max_tool_rounds: int = Field(default=3, ge=1)

    @property
    def effective_min_silence_ms(self) -> int:
        """未开启自适应时即为固定值；开启时作为初始值。"""
        return self.min_silence_ms

    def clamp_silence(self, value_ms: int) -> int:
        """把自适应算法的输出夹到允许区间内。"""
        return max(self.adaptive_silence_min_ms,
                   min(self.adaptive_silence_max_ms, value_ms))

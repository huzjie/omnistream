"""打断（barge-in）子包。

三段式职责拆分，方便单独调参与替换：

``detector`` → ``policy`` → ``executor``

典型装配::

    detector = BargeInDetector(cfg.barge_in, session_id=sid)
    policy = BargeInPolicy(cfg.barge_in, session_id=sid)
    executor = BargeInExecutor(machine, turns, playback=pq, provider=prv, policy=policy, emit=emit)

    signal = detector.observe(vad_result, state=machine.state)
    if signal is not None:
        decision = policy.decide(signal, state=machine.state,
                                 played_ms=pq.played_ms, pending_ms=pq.pending_ms)
        outcome = await executor.execute(signal, decision)
"""
from __future__ import annotations

from omnistream.duplex.bargein.detector import (
    DEFAULT_ACTIVE_STATES,
    DEFAULT_GAP_TOLERANCE_MS,
    HIGH_CONFIDENCE_PROB,
    BargeInDetector,
    BargeInSignal,
    BargeInStats,
)
from omnistream.duplex.bargein.executor import (
    BARGE_IN_SLO_MS,
    DEFAULT_PROVIDER_TIMEOUT_MS,
    BargeInExecutor,
    BargeInOutcome,
    EventEmitter,
)
from omnistream.duplex.bargein.policy import (
    DEFAULT_GUARD_MS,
    DEFAULT_MAX_GRACE_MS,
    BargeInAction,
    BargeInPolicy,
    PolicyDecision,
)

__all__ = [
    "BARGE_IN_SLO_MS",
    "DEFAULT_ACTIVE_STATES",
    "DEFAULT_GAP_TOLERANCE_MS",
    "DEFAULT_GUARD_MS",
    "DEFAULT_MAX_GRACE_MS",
    "DEFAULT_PROVIDER_TIMEOUT_MS",
    "HIGH_CONFIDENCE_PROB",
    "BargeInAction",
    "BargeInDetector",
    "BargeInExecutor",
    "BargeInOutcome",
    "BargeInPolicy",
    "BargeInSignal",
    "BargeInStats",
    "EventEmitter",
    "PolicyDecision",
]

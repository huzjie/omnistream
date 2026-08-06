"""Silero VAD（ONNX Runtime）后端 —— 默认高精度后端。

特点：

* 状态化 RNN，逐帧输出 [0,1] 语音概率，中文/嘈杂环境显著优于 webrtcvad
* CPU 单帧推理 < 1ms（16kHz / 512 样本窗口）
* 自动适配 v4（h/c 双状态）与 v5（合并 state）两种 ONNX 输入签名
* 权重缺失时可按配置自动下载到本地缓存目录；下载被禁用或失败则抛
  :class:`VADUnavailable`，由工厂降级到 webrtc / energy 后端

依赖 ``onnxruntime``（可选）。未安装时模块仍可正常 import。
"""
from __future__ import annotations

import os
import threading
import urllib.request
from pathlib import Path
from typing import Any, Final

import numpy as np
import structlog

from omnistream.duplex.vad.base import (
    DEFAULT_FRAME_MS,
    DEFAULT_SAMPLE_RATE,
    BaseVAD,
    VADResult,
    VADUnavailable,
    rms_dbfs,
)
from omnistream.schemas.audio import AudioChunk

log = structlog.get_logger(__name__)

_ort: Any = None
_IMPORT_ERROR: str = ""

try:  # pragma: no cover - 依赖可选
    import onnxruntime as _onnxruntime

    _ort = _onnxruntime
except Exception as exc:  # noqa: BLE001
    _IMPORT_ERROR = str(exc)

#: 官方 silero-vad v5 ONNX 权重（MIT License）
DEFAULT_MODEL_URL: Final[str] = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1/src/silero_vad/data/silero_vad.onnx"
)

#: Silero 只支持 8k / 16k；其它采样率内部线性重采样到 16k
_NATIVE_RATES: Final[tuple[int, ...]] = (8000, 16000)

#: 各采样率对应的窗口样本数（v5 固定值）
_WINDOW_SAMPLES: Final[dict[int, int]] = {8000: 256, 16000: 512}

_download_lock = threading.Lock()


def is_available() -> bool:
    """onnxruntime 是否可用。"""
    return _ort is not None


def default_model_path() -> Path:
    """默认权重缓存路径：``$OMS_MODEL_DIR`` 或 ``~/.cache/omnistream/models``。"""
    root = os.environ.get("OMS_MODEL_DIR") or str(Path.home() / ".cache" / "omnistream" / "models")
    return Path(root) / "silero_vad.onnx"


def ensure_model(model_path: Path, *, url: str = DEFAULT_MODEL_URL, allow_download: bool = True,
                 timeout_s: float = 30.0) -> Path:
    """确保权重文件存在，必要时下载。

    Args:
        model_path: 目标路径。
        url: 下载地址。
        allow_download: 为 False 且文件不存在时抛 :class:`VADUnavailable`。
        timeout_s: 下载超时。

    Returns:
        权重文件路径。
    """
    if model_path.is_file() and model_path.stat().st_size > 1024:
        return model_path
    if not allow_download:
        raise VADUnavailable(f"Silero 权重不存在且已禁用自动下载: {model_path}")
    with _download_lock:
        if model_path.is_file() and model_path.stat().st_size > 1024:
            return model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = model_path.with_suffix(".onnx.part")
        try:
            log.info("silero_model_download_started", url=url, target=str(model_path))
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310 - 固定可信 URL
                payload = resp.read()
            if len(payload) < 1024:
                raise VADUnavailable(f"Silero 权重下载内容异常，仅 {len(payload)} 字节")
            tmp.write_bytes(payload)
            tmp.replace(model_path)
            log.info("silero_model_download_finished", bytes=len(payload), target=str(model_path))
        except VADUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise VADUnavailable(f"Silero 权重下载失败: {exc}") from exc
    return model_path


class SileroVAD(BaseVAD):
    """Silero VAD ONNX 推理封装（状态化，逐帧概率）。"""

    name = "silero"

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_ms: int = DEFAULT_FRAME_MS,
        threshold: float = 0.5,
        model_path: str | os.PathLike[str] | None = None,
        allow_download: bool = True,
        num_threads: int = 1,
    ) -> None:
        if _ort is None:
            raise VADUnavailable(f"onnxruntime 未安装（pip install onnxruntime）: {_IMPORT_ERROR}")
        super().__init__(sample_rate=sample_rate, frame_ms=frame_ms, threshold=threshold)

        path = Path(model_path) if model_path else default_model_path()
        self.model_path = ensure_model(path, allow_download=allow_download)

        self._native_rate = 16000 if sample_rate not in _NATIVE_RATES else sample_rate
        self._window = _WINDOW_SAMPLES[self._native_rate]

        opts = _ort.SessionOptions()
        opts.inter_op_num_threads = max(1, int(num_threads))
        opts.intra_op_num_threads = max(1, int(num_threads))
        opts.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        try:
            self._session = _ort.InferenceSession(
                str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # noqa: BLE001
            raise VADUnavailable(f"Silero ONNX 会话创建失败: {exc}") from exc

        self._input_names = {i.name for i in self._session.get_inputs()}
        self._output_names = [o.name for o in self._session.get_outputs()]
        #: v5 用单一 state[2,1,128]；v4 用 h[2,1,64] + c[2,1,64]
        self._is_v5 = "state" in self._input_names
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._last_prob = 0.0
        log.info("silero_vad_ready", model=str(self.model_path), version="v5" if self._is_v5 else "v4",
                 rate=self._native_rate, window=self._window)

    # ------------------------------------------------------------------

    def reset(self) -> None:
        super().reset()
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._last_prob = 0.0

    def close(self) -> None:
        self._session = None  # type: ignore[assignment]

    def process_frame(self, chunk: AudioChunk) -> VADResult:
        samples = self.to_mono_float(chunk)
        energy_db = rms_dbfs(samples)
        resampled = self._resample(samples, int(chunk.sample_rate), self._native_rate)
        self._buffer = np.concatenate([self._buffer, resampled]) if self._buffer.size else resampled

        probs: list[float] = []
        while self._buffer.size >= self._window:
            block = self._buffer[: self._window]
            self._buffer = self._buffer[self._window :]
            probs.append(self._infer(block))

        prob = max(probs) if probs else self._last_prob
        self._last_prob = prob
        is_speech = prob >= self.threshold
        self._account(is_speech)
        return VADResult(
            is_speech=is_speech,
            prob=float(prob),
            pts_ms=int(chunk.pts_ms),
            energy_db=energy_db,
            backend=self.name,
            duration_ms=float(chunk.duration_ms) or float(self.frame_ms),
            noise_floor_db=-60.0,
            details={"windows": float(len(probs))},
        )

    # ------------------------------------------------------------------

    def _infer(self, block: np.ndarray) -> float:
        payload: dict[str, Any] = {"input": block.reshape(1, -1).astype(np.float32)}
        if "sr" in self._input_names:
            payload["sr"] = np.array(self._native_rate, dtype=np.int64)
        if self._is_v5:
            payload["state"] = self._state
        else:
            payload["h"] = self._h
            payload["c"] = self._c
        try:
            outputs = self._session.run(self._output_names, payload)
        except Exception as exc:  # noqa: BLE001 - 推理失败不得中断实时链路
            log.warning("silero_infer_failed", error=str(exc))
            return self._last_prob
        prob = float(np.asarray(outputs[0]).reshape(-1)[0])
        if self._is_v5 and len(outputs) > 1:
            self._state = np.asarray(outputs[1], dtype=np.float32)
        elif not self._is_v5 and len(outputs) > 2:
            self._h = np.asarray(outputs[1], dtype=np.float32)
            self._c = np.asarray(outputs[2], dtype=np.float32)
        return float(min(1.0, max(0.0, prob)))

    @staticmethod
    def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate == dst_rate or samples.size == 0:
            return samples.astype(np.float32, copy=False)
        dst_len = max(1, int(round(samples.size * dst_rate / float(src_rate))))
        src_idx = np.linspace(0.0, samples.size - 1.0, num=dst_len, dtype=np.float64)
        return np.interp(src_idx, np.arange(samples.size, dtype=np.float64), samples).astype(np.float32)

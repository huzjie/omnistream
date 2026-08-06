"""通用工具集合。

分层纪律：``utils`` **只能**依赖 ``core``（以及 numpy 这类基础第三方库），
不得反向导入 ``api`` / ``duplex`` / ``providers`` 等上层包，否则会形成环。

子模块速查：

===================  ==========================================================
模块                  职责
===================  ==========================================================
``audio_math``        电平/增益/包络/分帧等纯数值音频运算
``text_norm``         展示、TTS、匹配三套文本归一化
``hashing``           BLAKE2b 指纹、内容寻址、Merkle 根
``crypto``            scrypt 口令、HKDF、AES-GCM 凭证加密、API Key 签发
``asyncio_utils``     后台任务、可确认取消、并发限流、周期任务
``validators``        采样率/ID/slug/URL 等语义校验（SSRF 防护）
``fileio``            原子写、安全路径拼接、JSON/JSONL、保留期清理
===================  ==========================================================

为避免 ``import omnistream.utils`` 就把 numpy 拉进来，本包**不做**符号再导出，
请直接从子模块导入::

    from omnistream.utils.audio_math import dbfs
    from omnistream.utils.hashing import stable_hash
"""
from __future__ import annotations

__all__ = [
    "asyncio_utils",
    "audio_math",
    "crypto",
    "fileio",
    "hashing",
    "text_norm",
    "validators",
]

"""mock 模式端到端冒烟脚本（无 API Key / 无 GPU）。

验证链路：配置加载 → Provider 注册表 → mock provider 连接 → 文本输入
→ 事件流（connected → text_delta/audio_delta → response_done）→ 关闭。

用法：
    OMS_PROVIDERS__DEFAULT=mock python scripts/smoke_mock.py
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omnistream.config.settings import get_settings  # noqa: E402
from omnistream.providers import get_registry  # noqa: E402
from omnistream.schemas.session import SessionConfig  # noqa: E402


async def main() -> int:
    print("== OmniStream mock 冒烟测试 ==")
    settings = get_settings()
    settings.providers.default = "mock"
    reg = get_registry(settings.providers)

    print(f"  provider 注册表: {reg.names()}")

    provider = reg.create_or_mock("mock")
    print(f"  创建 provider: {provider!r}")

    config = SessionConfig(
        provider="mock",
        modalities=["text", "audio"],
        instructions="你是一个简洁的语音助手。",
    )
    await provider.connect(config)
    print("  已连接")

    await provider.send_text("你好，请介绍一下你自己")
    await provider.request_response()

    kinds: list[str] = []
    texts: list[str] = []
    ok = False
    async for ev in provider.events():
        kinds.append(ev.kind)
        if ev.text:
            texts.append(ev.text)
        if ev.kind == "response_done":
            ok = True
            break
        if ev.kind == "error":
            print(f"  错误事件: {ev.error}")
            break
        if ev.kind == "closed":
            break

    await provider.close()
    print(f"  事件序列: {kinds}")
    print(f"  文本输出: {''.join(texts)[:120]!r}")
    print(f"  结果: {'✅ 通过' if ok else '❌ 失败'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

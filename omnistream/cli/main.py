"""oms CLI 入口。

用法：
    oms version
    oms providers list
    oms sessions create --provider mock
    oms sessions list
    oms health
    oms smoke              # 端到端冒烟（mock 模式）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any


def _print(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def cmd_version(_: argparse.Namespace) -> None:
    from omnistream import __version__

    print(f"OmniStream v{__version__}")


def cmd_providers(_: argparse.Namespace) -> None:
    from omnistream.providers import get_registry

    reg = get_registry()
    _print({"providers": [p.model_dump(mode="json") for p in reg.list_info()]})


def cmd_sessions(args: argparse.Namespace) -> None:
    import asyncio as _asyncio

    from omnistream.config.settings import get_settings
    from omnistream.db.repositories.session_repo import SessionRepository
    from omnistream.db.session import reset_session_factory, session_scope

    reset_session_factory()

    async def _run() -> None:
        repo = SessionRepository()
        if args.action == "create":
            async with session_scope() as session:
                created = await repo.create_session(
                    session, provider=args.provider or "mock",
                    end_user_id="cli", record_enabled=False)
                _print({"session_id": created.id,
                        "provider": created.provider,
                        "status": created.state})
        elif args.action == "list":
            async with session_scope() as session:
                page = await repo.list_sessions(session, limit=args.limit)
                items = getattr(page, "items", page) if not hasattr(page, "items") else page.items
                _print({"sessions": [
                    {"id": s.id, "provider": s.provider,
                     "status": getattr(s, "state", None) or getattr(s, "status", None),
                     "created_at": getattr(s, "created_at", None)}
                    for s in items]})

    _asyncio.run(_run())


def cmd_health(_: argparse.Namespace) -> None:
    import httpx

    from omnistream.config.settings import get_settings

    s = get_settings()
    url = f"http://localhost:{s.port}/healthz"
    try:
        resp = httpx.get(url, timeout=5)
        _print({"url": url, "status": resp.status_code, "body": resp.json()})
    except Exception as exc:  # noqa: BLE001
        _print({"url": url, "status": "unreachable", "error": str(exc)})


def cmd_smoke(_: argparse.Namespace) -> None:
    """端到端冒烟：mock provider 一轮全双工。"""
    from omnistream.config.settings import get_settings
    from omnistream.providers import get_registry

    settings = get_settings()
    settings.providers.default = "mock"
    reg = get_registry(settings.providers)

    async def run() -> dict[str, Any]:
        from omnistream.schemas.session import SessionConfig

        provider = reg.create_or_mock("mock")
        config = SessionConfig(provider="mock", modalities=["text"])
        await provider.connect(config)
        await provider.send_text("你好")
        events = []
        async for ev in provider.events():
            events.append(ev.kind)
            if ev.kind in ("response_done", "closed", "error"):
                break
        await provider.close()
        return {"provider": provider.name, "events": events}

    result = asyncio.run(run())
    _print({"smoke": "ok", **result})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="oms", description="OmniStream CLI")
    p.add_argument("--config", default="config.yaml", help="配置文件路径")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("version").set_defaults(func=cmd_version)

    sp = sub.add_parser("providers")
    sp.add_argument("action", choices=["list"])
    sp.set_defaults(func=cmd_providers)

    ss = sub.add_parser("sessions")
    ss.add_argument("action", choices=["create", "list"])
    ss.add_argument("--provider", default=None)
    ss.add_argument("--limit", type=int, default=20)
    ss.set_defaults(func=cmd_sessions)

    sub.add_parser("health").set_defaults(func=cmd_health)
    sub.add_parser("smoke").set_defaults(func=cmd_smoke)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

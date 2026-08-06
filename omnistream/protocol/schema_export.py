"""导出协议 JSON Schema 到 ``docs/schemas/``，供 TS SDK 生成类型与文档站引用。

命令行用法::

    python -m omnistream.protocol.schema_export --out docs/schemas
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from omnistream.core.serialization import dumps
from omnistream.protocol.client_events import CLIENT_PAYLOAD_MODELS
from omnistream.protocol.enums import (
    AudioEncoding,
    ClientEventType,
    MemoryLayer,
    Modality,
    ServerEventType,
    SessionState,
    StopReason,
    ToolCallState,
    TurnRole,
    TurnState,
    VideoEncoding,
)
from omnistream.protocol.envelope import Envelope
from omnistream.protocol.server_events import SERVER_PAYLOAD_MODELS
from omnistream.protocol.version import PROTOCOL_VERSION

__all__ = ["build_schema_bundle", "export_schemas", "main"]

_ENUMS = {
    "Modality": Modality,
    "AudioEncoding": AudioEncoding,
    "VideoEncoding": VideoEncoding,
    "SessionState": SessionState,
    "TurnState": TurnState,
    "TurnRole": TurnRole,
    "StopReason": StopReason,
    "ToolCallState": ToolCallState,
    "MemoryLayer": MemoryLayer,
    "ClientEventType": ClientEventType,
    "ServerEventType": ServerEventType,
}


def build_schema_bundle() -> dict[str, Any]:
    """构造完整的协议 schema 包（单一 JSON 对象）。"""
    client: dict[str, Any] = {}
    for event_type, model in CLIENT_PAYLOAD_MODELS.items():
        client[str(event_type)] = model.model_json_schema(mode="validation")

    server: dict[str, Any] = {}
    for event_type, model in SERVER_PAYLOAD_MODELS.items():
        server[str(event_type)] = model.model_json_schema(mode="serialization")

    enums: dict[str, Any] = {
        name: {"type": "string", "enum": [str(member) for member in enum_cls]}
        for name, enum_cls in _ENUMS.items()
    }

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "omnistream realtime protocol",
        "protocol_version": PROTOCOL_VERSION,
        "envelope": Envelope.model_json_schema(mode="validation"),
        "enums": enums,
        "client_events": client,
        "server_events": server,
        "binary_frame": {
            "magic": "OMS1",
            "layout": [
                {"offset": 0, "size": 4, "field": "magic", "value": "OMS1"},
                {"offset": 4, "size": 1, "field": "kind", "enum": {"1": "audio", "2": "video"}},
                {"offset": 5, "size": 2, "field": "header_len", "type": "uint16le"},
                {"offset": 7, "size": "header_len", "field": "header", "type": "json-utf8"},
                {"offset": "7+header_len", "size": "rest", "field": "payload", "type": "bytes"},
            ],
            "audio_header": {"sid": "string", "pts": "int", "sr": "int", "ch": "int",
                             "enc": "string", "seq": "int?"},
            "video_header": {"sid": "string", "pts": "int", "w": "int", "h": "int",
                             "enc": "string", "ks": "float"},
            "limits": {"max_header_bytes": 4096, "max_payload_bytes": 4 * 1024 * 1024},
        },
    }


def export_schemas(out_dir: str | Path = "docs/schemas", *, split: bool = True) -> list[Path]:
    """把 schema 写入磁盘，返回生成的文件列表。

    Args:
        out_dir: 输出目录，不存在会自动创建。
        split: 为 True 时除了汇总包，还按事件类型拆分成单文件，便于 TS 侧按需引用。
    """
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    bundle = build_schema_bundle()

    written: list[Path] = []
    bundle_path = target / "protocol.schema.json"
    bundle_path.write_bytes(dumps(bundle, sort_keys=True, indent=True))
    written.append(bundle_path)

    if split:
        client_dir = target / "client"
        server_dir = target / "server"
        client_dir.mkdir(parents=True, exist_ok=True)
        server_dir.mkdir(parents=True, exist_ok=True)
        for name, schema in bundle["client_events"].items():
            path = client_dir / f"{name.replace('.', '_')}.json"
            path.write_bytes(dumps(schema, sort_keys=True, indent=True))
            written.append(path)
        for name, schema in bundle["server_events"].items():
            path = server_dir / f"{name.replace('.', '_')}.json"
            path.write_bytes(dumps(schema, sort_keys=True, indent=True))
            written.append(path)

    index_path = target / "index.json"
    index_path.write_bytes(dumps({
        "protocol_version": PROTOCOL_VERSION,
        "client_events": sorted(bundle["client_events"]),
        "server_events": sorted(bundle["server_events"]),
        "files": [str(p.relative_to(target)).replace("\\", "/") for p in written],
    }, sort_keys=True, indent=True))
    written.append(index_path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出 omnistream 协议 JSON Schema")
    parser.add_argument("--out", default="docs/schemas", help="输出目录（默认 docs/schemas）")
    parser.add_argument("--no-split", action="store_true", help="只生成汇总包，不拆分单文件")
    args = parser.parse_args(argv)

    files = export_schemas(args.out, split=not args.no_split)
    for path in files:
        print(f"wrote {path}")
    print(f"total {len(files)} file(s), protocol {PROTOCOL_VERSION}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())

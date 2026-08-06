#!/usr/bin/env python
"""Export the HTTP API as an OpenAPI 3.1 document.

Generating the spec offline (rather than scraping a running server) means CI can
publish SDKs and API docs without starting Postgres, Redis or any provider::

    python scripts/gen_openapi.py                       # -> docs/openapi.json
    python scripts/gen_openapi.py --format yaml         # -> docs/openapi.yaml
    python scripts/gen_openapi.py -o build/openapi.json
    python scripts/gen_openapi.py --check               # CI: fail if stale

The app is built with ``OMS_MOCK_MODE=true`` so router auto-discovery runs without
external dependencies; routers that fail to import are skipped with a warning by
``omnistream.api.registry`` rather than aborting the export.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Must be set before settings are first read.
os.environ.setdefault("OMS_MOCK_MODE", "true")
os.environ.setdefault("OMS_ENV", "dev")
os.environ.setdefault("OMS_OBSERVABILITY_LOG_LEVEL", "WARNING")

from omnistream.__version__ import __version__  # noqa: E402
from omnistream.api.app import create_app  # noqa: E402
from omnistream.api.registry import mount_routers  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "docs" / "openapi.json"


def build_spec() -> dict[str, Any]:
    """Build the OpenAPI document with all discoverable routers mounted.

    ``create_app`` defers router mounting to the lifespan handler so that startup
    ordering is observable; for a static export we mount them directly instead of
    spinning up an event loop.
    """
    app = create_app()
    mount_routers(app, prefix="/v1")
    spec: dict[str, Any] = app.openapi()

    # Stable, human-meaningful metadata for downstream SDK generators.
    spec.setdefault("info", {})
    spec["info"]["version"] = __version__
    spec["info"]["x-generated-by"] = "scripts/gen_openapi.py"
    spec["servers"] = [
        {"url": "http://localhost:8080", "description": "Local development"},
        {"url": "https://api.example.com", "description": "Self-hosted deployment"},
    ]
    components = spec.setdefault("components", {})
    components.setdefault("securitySchemes", {}).update({
        "bearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"},
        "apiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
    })
    spec.setdefault("security", [{"bearerAuth": []}, {"apiKeyAuth": []}])
    return spec


def _serialize(spec: dict[str, Any], fmt: str) -> str:
    """Render the spec as a deterministic JSON or YAML string."""
    if fmt == "json":
        return json.dumps(spec, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "yaml output requires PyYAML: pip install pyyaml"
        ) from exc
    return yaml.safe_dump(spec, sort_keys=True, allow_unicode=True, width=100)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gen_openapi",
        description="Export the OmniStream HTTP API as an OpenAPI 3.1 document.",
    )
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="output file (default: docs/openapi.<format>)")
    parser.add_argument("--format", choices=("json", "yaml"), default="json",
                        help="serialization format (default: json)")
    parser.add_argument("--check", action="store_true",
                        help="verify the committed spec matches the code; exit 1 if stale")
    args = parser.parse_args(argv)

    output: Path = args.output or DEFAULT_OUT.with_suffix(f".{args.format}")
    if not output.is_absolute():
        output = REPO_ROOT / output

    spec = build_spec()
    rendered = _serialize(spec, args.format)

    if args.check:
        if not output.exists():
            print(f"missing: {output}", file=sys.stderr)
            print("run `python scripts/gen_openapi.py` and commit the result", file=sys.stderr)
            return 1
        if output.read_text(encoding="utf-8") != rendered:
            print(f"stale: {output}", file=sys.stderr)
            print("run `python scripts/gen_openapi.py` and commit the result", file=sys.stderr)
            return 1
        print(f"openapi spec is up to date ({len(spec.get('paths', {}))} paths)")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")

    try:
        shown = output.relative_to(REPO_ROOT)
    except ValueError:
        shown = output
    print(f"wrote {shown}")
    print(f"  version : {spec['info']['version']}")
    print(f"  paths   : {len(spec.get('paths', {}))}")
    print(f"  schemas : {len(spec.get('components', {}).get('schemas', {}))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

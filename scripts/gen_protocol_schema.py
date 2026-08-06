#!/usr/bin/env python
"""Export the ``oms.v1`` protocol as JSON Schema.

The generated bundle is the single source of truth for SDK codegen (TypeScript, Go,
Swift) and for contract tests. Run it whenever protocol models change::

    python scripts/gen_protocol_schema.py                     # -> docs/schemas/
    python scripts/gen_protocol_schema.py -o build/schema     # custom directory
    python scripts/gen_protocol_schema.py --bundle-only       # one combined file
    python scripts/gen_protocol_schema.py --check             # CI: fail if stale

``--check`` regenerates into a temporary directory and diffs against the committed
output, so a pull request that changes a protocol model without refreshing the schema
fails CI instead of silently drifting from the SDKs.
"""
from __future__ import annotations

import argparse
import filecmp
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omnistream.protocol.schema_export import build_schema_bundle, export_schemas  # noqa: E402
from omnistream.protocol.version import PROTOCOL_VERSION  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "schemas"


def _write_bundle(out_dir: Path) -> Path:
    """Write the single combined bundle file and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, Any] = build_schema_bundle()
    target = out_dir / "oms-protocol.schema.json"
    target.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def _generate(out_dir: Path, *, bundle_only: bool) -> list[Path]:
    """Generate schema files into ``out_dir`` and return everything written."""
    written: list[Path] = [_write_bundle(out_dir)]
    if not bundle_only:
        written.extend(export_schemas(out_dir, split=True))
    return sorted(set(written))


def _check(out_dir: Path, *, bundle_only: bool) -> int:
    """Regenerate into a temp dir and compare with ``out_dir``. Returns an exit code."""
    with tempfile.TemporaryDirectory(prefix="oms-schema-") as tmp:
        tmp_dir = Path(tmp)
        fresh = _generate(tmp_dir, bundle_only=bundle_only)
        stale: list[str] = []
        missing: list[str] = []
        for path in fresh:
            committed = out_dir / path.relative_to(tmp_dir)
            if not committed.exists():
                missing.append(str(committed.relative_to(REPO_ROOT)))
            elif not filecmp.cmp(path, committed, shallow=False):
                stale.append(str(committed.relative_to(REPO_ROOT)))

    if not stale and not missing:
        print(f"schema is up to date ({len(fresh)} files, protocol {PROTOCOL_VERSION})")
        return 0

    for name in missing:
        print(f"missing: {name}", file=sys.stderr)
    for name in stale:
        print(f"stale:   {name}", file=sys.stderr)
    print("\nrun `python scripts/gen_protocol_schema.py` and commit the result",
          file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gen_protocol_schema",
        description="Export the oms.v1 protocol as JSON Schema.",
    )
    parser.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"output directory (default: {DEFAULT_OUT_DIR.relative_to(REPO_ROOT)})")
    parser.add_argument("--bundle-only", action="store_true",
                        help="only write the combined bundle, skip per-model files")
    parser.add_argument("--check", action="store_true",
                        help="verify committed schema matches the code; exit 1 if stale")
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir

    if args.check:
        return _check(out_dir, bundle_only=args.bundle_only)

    written = _generate(out_dir, bundle_only=args.bundle_only)
    for path in written:
        try:
            shown = path.relative_to(REPO_ROOT)
        except ValueError:
            shown = path
        print(f"wrote {shown}")
    print(f"\n{len(written)} file(s), protocol version {PROTOCOL_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

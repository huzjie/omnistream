"""docker-compose 健康检查脚本。"""
from __future__ import annotations

import sys
import time

import httpx


def main() -> int:
    url = "http://localhost:8000/healthz"
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=3)
            if resp.status_code == 200:
                print(f"✅ API 就绪: {resp.json()}")
                return 0
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    print("❌ API 未在 60s 内就绪", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

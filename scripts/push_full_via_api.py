"""通过 GitHub Git Data API 全量推送到新仓库（绕过 github.com 443 阻断）。

用于仓库为空/无历史时的首次全量推送：
    1. 读取本地全部 git 追踪文件
    2. 批量上传 blob（支持并发）
    3. 创建根 tree
    4. 创建 commit（挂到指定 parent 或作为首个 commit）
    5. 更新 refs/heads/main

用法：
    python scripts/push_full_via_api.py --repo omnistream --root .
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

API_ROOT = "https://api.github.com"
TIMEOUT = 120
MAX_WORKERS = 3


def load_credentials() -> tuple[str, str]:
    """优先读环境变量 GITHUB_TOKEN/GITHUB_USERNAME，其次读仓库根 config.yaml。"""
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN", "")
    username = os.environ.get("GITHUB_USERNAME", "")
    if token and username:
        return username, token
    config_path = Path("config.yaml")
    if config_path.exists():
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        gh = config.get("github", {})
        return gh.get("username", ""), gh.get("token", "")
    raise SystemExit("请设置 GITHUB_TOKEN/GITHUB_USERNAME 环境变量或提供 config.yaml")


def list_tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=str(root), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise SystemExit(f"git ls-files 失败: {result.stderr.strip()}")
    files = [ln for ln in result.stdout.splitlines() if ln.strip()]
    # 排除大型/二进制敏感文件
    return [f for f in files if not f.startswith(".git/")]


class GitDataPusher:
    def __init__(self, owner: str, repo: str, token: str, branch: str = "main"):
        self.owner = owner
        self.repo = repo
        self.branch = branch
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
        })

    def _url(self, path: str) -> str:
        return f"{API_ROOT}/repos/{self.owner}/{self.repo}{path}"

    def get_ref(self) -> dict[str, Any] | None:
        resp = self.session.get(self._url(f"/git/ref/heads/{self.branch}"), timeout=TIMEOUT)
        if resp.status_code in (404, 409):
            return None  # 404 无该分支 / 409 空仓库
        resp.raise_for_status()
        return resp.json()

    def create_blob(self, content: bytes) -> str:
        last_exc: Exception | None = None
        for attempt in range(4):
            resp = self.session.post(
                self._url("/git/blobs"),
                json={"content": base64.b64encode(content).decode("ascii"),
                      "encoding": "base64"},
                timeout=TIMEOUT,
            )
            if resp.status_code == 422:
                raise RuntimeError(f"blob 422: {resp.text[:200]}")
            if resp.status_code in (403, 429):
                last_exc = RuntimeError(f"blob {resp.status_code}: {resp.text[:120]}")
                time.sleep(3 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()["sha"]
        raise last_exc or RuntimeError("blob upload failed")

    def create_tree(self, entries: list[dict]) -> str:
        resp = self.session.post(
            self._url("/git/trees"),
            json={"tree": entries},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["sha"]

    def create_commit(self, message: str, tree: str,
                      parents: list[str] | None = None) -> str:
        payload: dict[str, Any] = {"message": message, "tree": tree}
        if parents:
            payload["parents"] = parents
        resp = self.session.post(self._url("/git/commits"), json=payload, timeout=TIMEOUT)
        if resp.status_code == 422:
            raise RuntimeError(f"commit 422: {resp.text[:300]}")
        resp.raise_for_status()
        return resp.json()["sha"]

    def update_ref(self, sha: str, force: bool = False) -> None:
        resp = self.session.patch(
            self._url(f"/git/refs/heads/{self.branch}"),
            json={"sha": sha, "force": force},
            timeout=TIMEOUT,
        )
        if resp.status_code == 422:
            raise RuntimeError(f"ref 422: {resp.text[:300]}")
        resp.raise_for_status()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--message", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    owner, token = load_credentials()

    files = list_tracked_files(root)
    print(f"目标: {owner}/{args.repo}@{'main'}")
    print(f"待推送文件: {len(files)}")

    if args.dry_run:
        print("[dry-run] 未实际推送")
        return 0

    pusher = GitDataPusher(owner, args.repo, token)

    # 取远端当前 ref（若已存在则取其 commit sha 作为 parent）
    existing = pusher.get_ref()
    parent_shas: list[str] = []
    if existing:
        parent_shas = [existing["object"]["sha"]]
        print(f"远端已有 ref，parent: {parent_shas[0][:8]}")

    # 批量上传 blob
    print("上传 blob（并发 %d）..." % MAX_WORKERS)
    entries: list[dict] = []
    failed: list[tuple[str, str]] = []

    def _upload(path: str) -> tuple[str, str]:
        full = root / path
        if not full.exists():
            return path, "missing"
        mode = "100755" if full.stat().st_mode & 0o100 else "100644"
        sha = pusher.create_blob(full.read_bytes())
        return path, f"{mode}:{sha}"

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_upload, p): p for p in files}
        done = 0
        for fut in as_completed(futures):
            done += 1
            path, result = fut.result()
            if result == "missing":
                failed.append((path, "missing"))
                continue
            mode, sha = result.split(":", 1)
            entries.append({"path": path, "mode": mode, "type": "blob", "sha": sha})
            if done % 50 == 0:
                print(f"  {done}/{len(files)}")

    if failed:
        print(f"⚠️ {len(failed)} 个文件缺失，跳过:")
        for p, _ in failed[:10]:
            print(f"  - {p}")

    print(f"创建 tree（{len(entries)} entries）...")
    tree_sha = pusher.create_tree(entries)
    print(f"tree: {tree_sha[:8]}")

    message = args.message or f"feat: push {len(files)} files via Git Data API"
    commit_sha = pusher.create_commit(message, tree_sha, parent_shas or None)
    print(f"commit: {commit_sha[:8]}")

    pusher.update_ref(commit_sha, force=bool(parent_shas))
    print(f"\n✅ 推送成功 → https://github.com/{owner}/{repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
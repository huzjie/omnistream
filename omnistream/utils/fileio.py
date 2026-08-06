"""文件与路径工具：原子写、安全拼接、JSON/JSONL 读写、目录清理。

录音落盘、审计日志归档、导出 OpenAPI/JSON Schema 都走这里。核心诉求只有两条：

* **原子性** —— 进程被 kill 时不能留下半截文件（先写 ``.tmp`` 再 ``os.replace``）；
* **路径安全** —— 任何来自用户的相对路径都必须经过 :func:`safe_join`，
  否则 ``../../etc/passwd`` 就穿透了。
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from omnistream.core.error_codes import ErrorCode
from omnistream.core.errors import OmniError, ValidationError

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "atomic_write_bytes",
    "atomic_write_text",
    "dir_size",
    "ensure_dir",
    "human_size",
    "iter_chunks",
    "prune_old_files",
    "read_json",
    "read_jsonl",
    "safe_filename",
    "safe_join",
    "write_json",
    "write_jsonl",
]

#: 流式读文件的默认块大小（1 MiB），兼顾内存占用与系统调用次数。
DEFAULT_CHUNK_SIZE: int = 1 << 20
_UNSAFE_CHARS = set('<>:"/\\|?*\0')
_SIZE_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """确保目录存在（含父目录），返回 :class:`~pathlib.Path`。"""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_join(base: str | os.PathLike[str], *parts: str) -> Path:
    """在 ``base`` 之下安全拼接路径，越界立即抛 :class:`ValidationError`。"""
    root = Path(base).resolve()
    candidate = root.joinpath(*parts).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValidationError(
            "path traversal detected",
            code=ErrorCode.INVALID_REQUEST,
            detail={"base": str(root), "resolved": str(candidate)},
        )
    return candidate


def safe_filename(name: str, *, fallback: str = "file", max_length: int = 180) -> str:
    """把任意字符串净化成跨平台安全的文件名。"""
    cleaned = "".join("_" if ch in _UNSAFE_CHARS or ord(ch) < 32 else ch for ch in name)
    cleaned = cleaned.strip(" .")
    if not cleaned:
        cleaned = fallback
    if len(cleaned) > max_length:
        stem, dot, suffix = cleaned.rpartition(".")
        if dot and len(suffix) <= 8:
            keep = max_length - len(suffix) - 1
            cleaned = f"{stem[:keep]}.{suffix}"
        else:
            cleaned = cleaned[:max_length]
    return cleaned


# --------------------------------------------------------------------------- 原子写
def atomic_write_bytes(path: str | os.PathLike[str], data: bytes, *,
                       mode: int = 0o644) -> Path:
    """原子写入二进制内容：同目录临时文件 + ``fsync`` + ``os.replace``。"""
    target = Path(path)
    ensure_dir(target.parent)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.",
                                    suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, target)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise OmniError(f"atomic write failed: {target}",
                        code=ErrorCode.INTERNAL, cause=exc) from exc
    return target


def atomic_write_text(path: str | os.PathLike[str], text: str, *,
                      encoding: str = "utf-8", mode: int = 0o644) -> Path:
    """原子写入文本内容。"""
    return atomic_write_bytes(path, text.encode(encoding), mode=mode)


# --------------------------------------------------------------------------- JSON
def read_json(path: str | os.PathLike[str], *, default: Any = None) -> Any:
    """读取 JSON 文件。文件不存在时返回 ``default``；内容损坏则抛错。"""
    target = Path(path)
    if not target.exists():
        return default
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OmniError(f"malformed json file: {target}",
                        code=ErrorCode.CONFIG_INVALID,
                        detail={"line": exc.lineno, "column": exc.colno},
                        cause=exc) from exc


def write_json(path: str | os.PathLike[str], payload: Any, *, indent: int = 2,
               sort_keys: bool = False) -> Path:
    """原子写入 JSON（UTF-8、不转义非 ASCII、末尾带换行）。"""
    text = json.dumps(payload, ensure_ascii=False, indent=indent,
                      sort_keys=sort_keys, default=str)
    return atomic_write_text(path, text + "\n")


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[Any]:
    """惰性逐行读取 JSONL，自动跳过空行。适合 GB 级审计日志。"""
    target = Path(path)
    if not target.exists():
        return
    with target.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise OmniError(f"malformed jsonl at {target}:{lineno}",
                                code=ErrorCode.CONFIG_INVALID, cause=exc) from exc


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Any], *,
                append: bool = False) -> int:
    """写入 JSONL，返回写入行数。``append=True`` 时直接追加（非原子，但有 flush）。"""
    target = Path(path)
    ensure_dir(target.parent)
    count = 0
    if append:
        with target.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                count += 1
            handle.flush()
        return count

    buffer: list[str] = []
    for row in rows:
        buffer.append(json.dumps(row, ensure_ascii=False, default=str))
        count += 1
    atomic_write_text(target, "\n".join(buffer) + ("\n" if buffer else ""))
    return count


# --------------------------------------------------------------------------- 流式与维护
def iter_chunks(path: str | os.PathLike[str],
                chunk_size: int = DEFAULT_CHUNK_SIZE) -> Iterator[bytes]:
    """按块读取文件，用于上传对象存储或计算大文件哈希。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                return
            yield block


def dir_size(path: str | os.PathLike[str]) -> int:
    """递归统计目录占用字节数（忽略无权限的条目）。"""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def human_size(n_bytes: float) -> str:
    """字节数 → 人类可读字符串（``1.5 MB``）。"""
    value = float(n_bytes)
    for unit in _SIZE_UNITS:
        if abs(value) < 1024.0 or unit == _SIZE_UNITS[-1]:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


def prune_old_files(path: str | os.PathLike[str], *, max_age_s: float,
                    pattern: str = "*", now: float | None = None) -> int:
    """删除目录下超过 ``max_age_s`` 的文件，返回删除数量。

    录音与临时切片的保留策略（``storage.retention_days``）由定时任务调用它执行。
    """
    directory = Path(path)
    if not directory.exists():
        return 0
    import time

    cutoff = (now if now is not None else time.time()) - max_age_s
    removed = 0
    for item in directory.glob(pattern):
        if not item.is_file():
            continue
        try:
            if item.stat().st_mtime < cutoff:
                item.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def _copy_tree(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> Path:
    """递归复制目录（内部用，导出静态资源时调用）。"""
    destination = Path(dst)
    ensure_dir(destination.parent)
    shutil.copytree(src, destination, dirs_exist_ok=True)
    return destination

"""Celery worker 启动模块。

用法：
    python -m omnistream.workers.celery_app worker --loglevel=info
    python -m omnistream.workers.celery_app beat
"""
from __future__ import annotations

import sys
from typing import Any

from omnistream.config.settings import get_settings
from omnistream.workers.tasks import init_celery


def build_app() -> Any:
    """根据配置构建 celery app（无 Redis 时返回 None 并禁用）。"""
    settings = get_settings()
    broker = None
    if settings.storage.use_redis and settings.storage.redis_url:
        broker = settings.storage.redis_url
    return init_celery(broker)


def main() -> None:
    app = build_app()
    if app is None:
        print("Celery 未启用：未配置 Redis（OMS_STORAGE__USE_REDIS=false）。")
        print("任务将以内联模式在 API 进程内执行，无需 worker。")
        sys.exit(0)
    # 交给 celery CLI
    from celery.bin.celery import celery as celery_cli

    celery_cli.main(argv=sys.argv[1:])


if __name__ == "__main__":
    main()

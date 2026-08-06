# =========================================================================
# OmniStream — 企业级实时全双工音视频多模态交互平台
# Apache-2.0 License
# =========================================================================

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 系统依赖（可选媒体库最小集）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopus0 libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# --- 依赖层 ---
FROM base AS deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- 运行层 ---
FROM base AS runtime
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

COPY . .

EXPOSE 8000 8080 9090 9464

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/healthz || exit 1

ENTRYPOINT ["python", "-m", "omnistream.api.app"]

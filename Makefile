# OmniStream Makefile
.PHONY: help install dev run mock test lint format migrate upgrade build up down

help:
	@echo "OmniStream 常用命令:"
	@echo "  make install   安装依赖"
	@echo "  make run       启动 API 服务"
	@echo "  make mock      以 mock 模式启动（零 Key 零 GPU）"
	@echo "  make dev       开发模式（热重载）"
	@echo "  make migrate   执行数据库迁移"
	@echo "  make build     构建 Docker 镜像"
	@echo "  make up        启动完整编排（API+Redis+Worker）"
	@echo "  make down      停止编排"

install:
	pip install -r requirements.txt -r requirements-dev.txt

run:
	python -m omnistream.api.app

mock:
	OMS_PROVIDERS__DEFAULT=mock python -m omnistream.api.app

dev:
	OMS_DEBUG=true python -m omnistream.api.app

lint:
	ruff check omnistream sdk-python/src scripts

migrate:
	alembic upgrade head

build:
	docker compose build

up:
	docker compose --profile full up -d

down:
	docker compose down

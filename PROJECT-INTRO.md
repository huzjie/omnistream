# OmniStream 项目介绍

> 企业级实时全双工音视频多模态交互平台（Realtime Full-Duplex Omni-Modal Interaction Platform）
>
> 一句话：把「边看、边听、边说」的实时全双工多模态 AI 交互能力，从各家私有 App 里解耦出来，做成任何企业都能自托管、多 Provider 可插拔、可观测可治理的开放中台。

---

## 1. 为什么需要 OmniStream

2026 年 8 月，实时多模态交互迎来爆发：

- **字节跳动 SeedRealtime**：原生音视频全双工大模型，统一架构原生融合音频/视频/文本，在连续多模态信息流上实时交互，已在豆包 App 全量上线；
- **京东 JoyAI-Video-Edit**：实时流式视频编辑，边看边改人物与场景；
- **MiniMax H3**：新一代通用多模态生成模型，视频编辑评测全球第一；
- **DeepSeek V4 / Qwen3.8-Max（2.4T MoE、1M 上下文）**：实时多模态推理成本大幅下降；
- **Meta 主动记忆 Agent**：独立于行动 Agent 的记忆管理模块，选择性注入"一条定向提醒"。

这些能力目前散落在各家私有 App / 单一厂商 API 中。**企业要把实时语音/视频 AI 接入自己的产品，面临三大痛点**：

1. **厂商锁定**：绑定单一云厂商，无法切换、无法降级、无法成本优化；
2. **无法自托管**：语音/视频数据必须经过第三方，数据合规难以满足；
3. **缺乏治理**：没有会话录制、质检、审计、限流、计费、可观测这些企业级能力。

**OmniStream 把"实时全双工多模态交互"做成一个开放、可自托管、可插拔的中间件平台**，让任何企业都能用一套 API 接入任意实时多模态大模型，并开箱即用地获得治理与可观测能力。

---

## 2. 核心能力

### 2.1 实时媒体接入层
- WebRTC（aiortc）+ WebSocket 双通道传输；
- 音频编解码与重采样（Opus / PCM16 / 采样率转换）；
- 视频帧抽取与自适应采样（关键帧选择、降采样、JPEG/WebP 编码）；
- 抖动缓冲（jitter buffer）+ 网络自适应码率。

### 2.2 全双工对话内核
- 双引擎 VAD（webrtcvad + Silero）+ 能量兜底；
- 端点检测、**barge-in 打断**、轮次状态机（listening / thinking / speaking / interrupted）；
- 静音超时、说话人切换、**主动发起 proactive turn**。

### 2.3 流式感知管线
- 流式 ASR（faster-whisper 本地 / 云端流式 / mock）；
- 流式 TTS（Piper 本地 / 云端 / mock）；
- 视觉理解（关键帧采样 + VLM 调用 + 视频记忆图谱）。

### 2.4 Provider 适配层（8 个真实适配器）
| Provider | 类型 | 特点 |
|---|---|---|
| `mock` | 原生全双工 | **零 API Key、零 GPU 即可端到端跑通**（演示/开发默认） |
| `doubao_seedrealtime` | 原生全双工 | 字节 SeedRealtime，音视频+文本原生融合 |
| `openai_realtime` | 原生全双工 | OpenAI Realtime API |
| `gemini_live` | 原生全双工 | Google Gemini Live，支持实时视觉 |
| `qwen_omni` | 原生全双工 | 阿里 Qwen-Omni / Qwen3.8-Max（2.4T / 1M 上下文） |
| `deepseek_v4` | 组合式 | DeepSeek V4，低成本流式 Agent |
| `kimi_k3` | 组合式 | Moonshot K3（2.8T 开源 MoE） |
| `local_pipeline` | 组合式 | faster-whisper + Ollama + Piper 全本地离线 |

### 2.5 主动记忆治理层（对标 Meta Proactive Memory Agent）
- **记忆四层**：session（会话）/ private_state（私有状态）/ stable_knowledge（稳定知识）/ procedural（程序性记忆）；
- **三闸门选择性注入**：relevance gate → confidence gate → timing gate，全过才注入"一条定向提醒"，否则**保持沉默**；
- 向量检索（ChromaDB 可选，numpy 零依赖回退）、置信度校准、遗忘与压缩。

### 2.6 实时工具调用编排
- 工具注册、并行执行（Semaphore 限流）、流式中途调用、超时与降级；
- 内置工具：时间、回显、安全计算、天气（mock）、HTTP 抓取。

### 2.7 平台能力
- 多租户、API Key / JWT 双通道鉴权、RBAC 角色权限矩阵；
- 令牌桶限流、配额、计费计量；
- 会话录制、回放、转写归档、质检评分、PII 脱敏、敏感词过滤、合规审计日志；
- Celery 离线任务（Redis 未配置自动内联降级）。

### 2.8 可观测性
- Prometheus 指标：**首字延迟 TTFT / 端到端 RTT / 打断响应时延 / VAD 时延**；
- OpenTelemetry 追踪、结构化日志（structlog）；
- Grafana 实时延迟看板（随仓库提供 dashboard JSON）。

### 2.9 接入形态
- FastAPI REST + WebSocket 实时通道；
- React 控制台（Vite + TS + Tailwind）：总览 / Providers / 会话 / 记忆 / 设置；
- TypeScript SDK、Python SDK、`oms` CLI。

---

## 3. 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                        客户端接入形态                          │
│   React 控制台 │ Python SDK │ TS SDK │ oms CLI │ WebRTC/WS  │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP / WebSocket
┌────────────────────────────▼────────────────────────────────┐
│                      OmniStream 网关 (FastAPI)               │
│   REST 路由 │ WebSocket 实时通道 │ 鉴权/租户/限流/配额/计费     │
└───────┬────────────────┬────────────────┬───────────────────┘
        │                │                │
┌───────▼───────┐ ┌──────▼───────┐ ┌──────▼──────────┐
│ 实时媒体接入层  │ │ 全双工对话内核 │ │ 流式感知管线      │
│ WebRTC/WS     │ │ VAD/barge-in │ │ ASR/TTS/VLM     │
│ 编解码/抖动缓冲 │ │ 轮次状态机    │ │ 视频记忆图谱      │
└───────┬───────┘ └──────┬───────┘ └──────┬──────────┘
        │                │                │
┌───────▼────────────────▼────────────────▼──────────┐
│              Provider 适配层 (8 个适配器)             │
│  mock │ Doubao │ OpenAI │ Gemini │ Qwen │ DeepSeek  │
│  Kimi │ Local Pipeline                                │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│ 平台治理与底座                                          │
│  主动记忆治理(三闸门) │ 工具编排 │ 录制/质检/审计/脱敏      │
│  SQLAlchemy+SQLite/PostgreSQL │ Redis │ Celery       │
│  Prometheus │ OpenTelemetry │ Grafana                 │
└──────────────────────────────────────────────────────┘
```

---

## 4. 快速开始（零 API Key 跑通）

### 方式一：Docker（约 60 秒）

```bash
git clone https://github.com/huzjie/omnistream.git
cd omnistream
cp .env.example .env        # 默认 mock 模式，无需任何修改
docker compose up --build   # 启动 API + (可选 Redis/Worker)
```

验证：

```bash
curl http://localhost:8000/healthz
# {"status":"ok","version":"1.0.0","env":"dev"}

curl http://localhost:8000/version
# {"version":"1.0.0","protocol_version":"oms.v1","env":"dev",
#  "mock_mode":true,"default_provider":"mock"}
```

### 方式二：本地 Python

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m omnistream.api.app                          # 启动 API
```

### 端到端冒烟（mock 模式全链路验证）

```bash
python scripts/smoke_mock.py
# 事件序列: connected → text_delta → audio_delta → ... → response_done
# 结果: ✅ 通过
```

### oms CLI

```bash
python -m omnistream.cli.main version        # OmniStream v1.0.0
python -m omnistream.cli.main providers list # 查看已注册 Provider
python -m omnistream.cli.main sessions create --provider mock
```

### 接入真实 Provider（以 OpenAI Realtime 为例）

```bash
# .env
OMS_PROVIDERS__OPENAI__API_KEY=sk-xxxx
OMS_PROVIDERS__DEFAULT=openai_realtime
```

---

## 5. 使用 SDK

### Python SDK

```python
import asyncio
from omnistream_sdk import RealtimeClient

async def main():
    client = RealtimeClient(provider="mock")
    await client.connect()
    await client.send_text("你好")
    async for ev in client.events():
        if ev.kind == "text_delta":
            print(ev.text, end="")
        if ev.kind == "response_done":
            break
    await client.close()

asyncio.run(main())
```

### TypeScript SDK

```typescript
import { RealtimeClient, EventKind } from '@omnistream/sdk';

const client = new RealtimeClient({ provider: 'mock' });
await client.connect();
client.on(EventKind.TEXT_DELTA, (ev) => process.stdout.write(ev.text ?? ''));
client.on(EventKind.RESPONSE_DONE, () => client.close());
client.sendText('你好');
```

---

## 6. 目录结构

```
omnistream/          后端核心（265 个 Python 文件）
├── api/             FastAPI 应用骨架与路由
├── core/            内核原语（错误/ID/背压/熔断/事件总线）
├── protocol/        实时事件协议（枚举/信封/二进制帧）
├── schemas/         领域模型契约
├── config/          pydantic-settings 配置体系
├── db/              数据持久层 + Alembic 迁移
├── platform/        多租户/鉴权/RBAC/配额/计费
├── governance/      录制/质检/审计/PII 脱敏/敏感词
├── media/           WebRTC/WS 传输、编解码、抖动缓冲
├── duplex/          全双工内核（VAD/barge-in/状态机）
├── perception/      流式 ASR/TTS/视觉
├── providers/       8 个 Provider 适配器
├── tools/           实时工具调用编排
├── memory/          主动记忆治理（三闸门）
├── observability/   Prometheus/OTel/延迟追踪
├── workers/         Celery 离线任务
└── cli/             oms 命令行
frontend/            React 控制台（Vite + TS + Tailwind）
sdk-ts/              TypeScript SDK
sdk-python/          Python SDK
deploy/              K8s/Helm/Prometheus/Grafana
docs/                架构文档（含完整 ARCHITECTURE.md）
.github/             CI/Docker/Dependabot
```

---

## 7. 部署

### Docker Compose（单机快速部署）

```bash
docker compose --profile full up -d   # API + Redis + Worker
docker compose --profile observability up -d  # + Prometheus + Grafana
```

### Kubernetes（Helm）

```bash
helm install omnistream deploy/helm/omnistream \
  --set global.image.tag=main \
  --namespace omnistream --create-namespace
```

### 可观测性接入

- Prometheus 抓取端点：`:9464/metrics`
- Grafana 看板：`deploy/grafana/provisioning/dashboards/omnistream-realtime.json`（TTFT / RTT / Barge-in / 会话数 / 事件吞吐）

---

## 8. 设计哲学与亮点

1. **Mock 优先，零依赖可跑**：默认 mock provider 让整条全双工链路在无 API Key、无 GPU、无 Redis 的情况下端到端跑通 —— 开发、CI、演示开箱即用；
2. **可选依赖降级**：aiortc / faster-whisper / piper / chroma 全部可选，缺失时自动降级到等价路径，绝不因缺依赖而崩溃；
3. **契约先行，并行开发**：核心契约文件（协议/模型/基类/背压）独立成文，6 个工程包可并行施工、零文件冲突；
4. **对标前沿**：主动记忆三闸门（Meta）、原生全双工（字节 SeedRealtime）、超长上下文（Qwen3.8 / Kimi K3）等 2026 年 8 月最新技术形态全部落地；
5. **企业级治理**：录制、质检、审计、脱敏、限流、计费、可观测 —— 拿来即用，不是玩具。

---

## 9. License

Apache-2.0。可自由使用、修改、商用，保留版权声明即可。

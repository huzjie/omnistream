# Configuration

All configuration is environment-driven (12-factor) and validated by
`pydantic-settings` at startup. An invalid value fails fast with `OMS-1100
CONFIG_INVALID` — the process never starts in a half-configured state.

Precedence, highest first:

1. explicit constructor arguments (tests only)
2. environment variables
3. `.env` file in the working directory
4. field defaults

Nested subsystems use their own prefix, so the variable name always tells you which
`BaseSettings` class owns it:

| Prefix | Class | File |
| --- | --- | --- |
| `OMS_` | `Settings` | `config/settings.py` |
| `OMS_APP_` | `AppSettings` | `config/app.py` |
| `OMS_MEDIA_` | `MediaSettings` | `config/media.py` |
| `OMS_DUPLEX_` | `DuplexSettings` | `config/duplex.py` |
| `OMS_PERCEPTION_` | `PerceptionSettings` | `config/perception.py` |
| `OMS_PROVIDERS_` | `ProvidersSettings` | `config/providers.py` |
| `OMS_MEMORY_` | `MemorySettings` | `config/memory.py` |
| `OMS_STORAGE_` | `StorageSettings` | `config/storage.py` |
| `OMS_SECURITY_` | `SecuritySettings` | `config/security.py` |
| `OMS_OBSERVABILITY_` | `ObservabilitySettings` | `config/observability.py` |

---

## Top level — `OMS_`

| Variable | Default | Notes |
| --- | --- | --- |
| `OMS_ENV` | `dev` | `dev｜staging｜prod`. `prod` disables the dev principal and refuses placeholder secrets |
| `OMS_DEBUG` | `false` | Enables uvicorn reload and verbose tracebacks |
| `OMS_APP_NAME` | `omnistream` | Used in logs, metrics and OTLP resource attributes |
| `OMS_HOST` | `0.0.0.0` | Bind address |
| `OMS_PORT` | `8080` | Bind port |
| `OMS_ROOT_PATH` | *(empty)* | Set when served behind a path-stripping proxy |
| `OMS_CORS_ORIGINS` | `["http://localhost:5173"]` | JSON array or comma-separated |
| **`OMS_MOCK_MODE`** | `false` | **The one switch that matters for evaluation** — see below |

### `OMS_MOCK_MODE=true`

Forces every external dependency to a deterministic in-process implementation:

```
providers.default   → mock
perception.asr      → mock
perception.tts      → mock
perception.vision   → mock
duplex.vad_backend  → energy      (pure numpy, no ONNX)
memory.store        → inmemory
memory.embedder     → hash
```

No API keys, no GPU, no model download, no Postgres, no Redis. The full duplex loop
still runs: real VAD on real audio energy, real endpointing, real barge-in timing,
real turn metrics — only the *content* is synthesised.

---

## Media — `OMS_MEDIA_`

| Variable | Default | Notes |
| --- | --- | --- |
| `OMS_MEDIA_INPUT_SAMPLE_RATE` | `16000` | Uplink rate; must be in `{8000,16000,22050,24000,44100,48000}` |
| `OMS_MEDIA_OUTPUT_SAMPLE_RATE` | `24000` | Downlink rate, matches mainstream TTS |
| `OMS_MEDIA_FRAME_MS` | `20` | Atomic frame length; also the barge-in truncation granularity |
| `OMS_MEDIA_CHANNELS` | `1` | Stereo input is down-mixed |
| `OMS_MEDIA_JITTER_TARGET_MS` | `60` | Playout buffer target |
| `OMS_MEDIA_MAX_UPLOAD_MB` | `4` | Hard cap per binary frame payload |
| `OMS_MEDIA_VIDEO_FPS` | `1.0` | Sampled frames per second sent to vision |
| `OMS_MEDIA_VIDEO_WIDTH` / `_HEIGHT` | `768` / `432` | Downscale target before vision |
| `OMS_MEDIA_JPEG_QUALITY` | `80` | Vision frame encoding quality |

## Duplex — `OMS_DUPLEX_`

| Variable | Default | Notes |
| --- | --- | --- |
| `OMS_DUPLEX_VAD_BACKEND` | `silero` | `energy｜webrtc｜silero｜ensemble` |
| `OMS_DUPLEX_VAD_THRESHOLD` | `0.5` | Speech probability threshold |
| `OMS_DUPLEX_MIN_SPEECH_MS` | `120` | Debounce against clicks and coughs |
| `OMS_DUPLEX_MIN_SILENCE_MS` | `480` | End-of-utterance silence |
| `OMS_DUPLEX_PREFIX_PADDING_MS` | `300` | Audio retained *before* speech onset |
| `OMS_DUPLEX_BARGE_IN_ENABLED` | `true` | Master switch for interruption |
| `OMS_DUPLEX_BARGE_IN_BUDGET_MS` | `200` | SLO; exceeding it emits an SLO violation |
| `OMS_DUPLEX_CANCEL_BUDGET_MS` | `100` | Provider must stop emitting within this |
| `OMS_DUPLEX_MAX_TURN_MS` | `120000` | Runaway response guard |
| `OMS_DUPLEX_IDLE_TIMEOUT_MS` | `300000` | Auto-close silent sessions |
| `OMS_DUPLEX_SEMANTIC_ENDPOINTING` | `true` | Use sentence completeness, not just silence |

## Perception — `OMS_PERCEPTION_`

| Variable | Default | Notes |
| --- | --- | --- |
| `OMS_PERCEPTION_ASR_ENGINE` | `funasr` | `mock｜funasr｜whisper｜provider` |
| `OMS_PERCEPTION_TTS_ENGINE` | `edge` | `mock｜edge｜cosyvoice｜provider` |
| `OMS_PERCEPTION_VISION_ENGINE` | `qwen_vl` | `mock｜qwen_vl｜provider` |
| `OMS_PERCEPTION_ASR_LANGUAGE` | `zh-CN` | BCP-47 tag |
| `OMS_PERCEPTION_MODEL_DIR` | `./models` | Local weights root |
| `OMS_PERCEPTION_DEVICE` | `cpu` | `cpu｜cuda｜mps` |

## Providers — `OMS_PROVIDERS_`

| Variable | Default | Notes |
| --- | --- | --- |
| `OMS_PROVIDERS_DEFAULT` | `mock` | Provider used when a session doesn't specify one |
| `OMS_PROVIDERS_FALLBACK_CHAIN` | `["mock"]` | Ordered fallback on circuit-open / timeout |
| `OMS_PROVIDERS_DOUBAO_API_KEY` | `null` | Per-provider entries follow `_<NAME>_<FIELD>` |
| `OMS_PROVIDERS_OPENAI_API_KEY` | `null` | |
| `OMS_PROVIDERS_<NAME>_BASE_URL` | vendor default | Override for proxies / private endpoints |
| `OMS_PROVIDERS_<NAME>_ENABLED` | `false` unless key present | |
| `OMS_PROVIDERS_TIMEOUT_S` | `60` | Per-request ceiling |
| `OMS_PROVIDERS_CIRCUIT_FAILURE_THRESHOLD` | `5` | Consecutive failures before opening |

Credentials are **never** logged and are encrypted at rest with
`OMS_SECURITY_CREDENTIAL_ENCRYPTION_KEY` (AES-256-GCM).

## Memory — `OMS_MEMORY_`

| Variable | Default | Notes |
| --- | --- | --- |
| `OMS_MEMORY_ENABLED` | `true` | |
| `OMS_MEMORY_STORE` | `pgvector` | `inmemory｜sqlite｜pgvector` |
| `OMS_MEMORY_EMBEDDER` | `bge_small` | `hash｜bge_small｜provider` |
| `OMS_MEMORY_EMBEDDING_DIM` | `384` | Must match the store's column |
| `OMS_MEMORY_RELEVANCE_THRESHOLD` | `0.62` | Below this, never inject |
| `OMS_MEMORY_NOVELTY_THRESHOLD` | `0.35` | Suppress restating known facts |
| `OMS_MEMORY_MAX_INJECTIONS_PER_TURN` | `1` | Deliberately conservative |
| `OMS_MEMORY_TTL_DAYS_PRIVATE_STATE` | `7` | Per-layer retention |

## Storage — `OMS_STORAGE_`

| Variable | Default | Notes |
| --- | --- | --- |
| `OMS_STORAGE_DATABASE_URL` | `sqlite+aiosqlite:///./data/omnistream.db` | Postgres recommended for production |
| `OMS_STORAGE_DB_POOL_SIZE` | `10` | |
| `OMS_STORAGE_AUTO_MIGRATE` | `true` | Runs `alembic upgrade head` at startup |
| `OMS_STORAGE_REDIS_URL` | `redis://localhost:6379/0` | |
| `OMS_STORAGE_EVENT_BUS` | `inproc` | `inproc｜redis` — use `redis` for multi-replica |
| `OMS_STORAGE_OBJECT_STORE` | `local` | `local｜s3｜minio｜oss｜cos` |
| `OMS_STORAGE_RETENTION_DAYS` | `30` | Recording / audit retention |

## Security — `OMS_SECURITY_`

| Variable | Default | Notes |
| --- | --- | --- |
| `OMS_SECURITY_JWT_SECRET` | dev placeholder | **Must** be overridden in production |
| `OMS_SECURITY_JWT_ALGORITHM` | `HS256` | `HS256｜HS512｜RS256` |
| `OMS_SECURITY_ACCESS_TOKEN_TTL_S` | `3600` | |
| `OMS_SECURITY_API_KEY_SALT` | dev placeholder | **Must** be overridden |
| `OMS_SECURITY_CREDENTIAL_ENCRYPTION_KEY` | `null` | Required to store provider keys |
| `OMS_SECURITY_REDACT_PII` | `true` | Applies to logs, recordings and audit |
| `OMS_SECURITY_LOG_TRANSCRIPTS` | `false` | Off by default — transcripts are user content |
| `OMS_SECURITY_REQUIRE_HTTPS` | `false` | Enable behind a TLS terminator |

`Settings.security.using_dev_secrets` is checked at startup; in `OMS_ENV=prod` it
refuses to boot with placeholder values.

## Observability — `OMS_OBSERVABILITY_`

| Variable | Default | Notes |
| --- | --- | --- |
| `OMS_OBSERVABILITY_LOG_LEVEL` | `INFO` | |
| `OMS_OBSERVABILITY_LOG_FORMAT` | `console` | Use `json` in production |
| `OMS_OBSERVABILITY_ACCESS_LOG` | `true` | |
| `OMS_OBSERVABILITY_SLOW_REQUEST_MS` | `1000` | Logged at WARNING above this |
| `OMS_OBSERVABILITY_METRICS_ENABLED` | `true` | Prometheus at `/metrics` |
| `OMS_OBSERVABILITY_TRACING_ENABLED` | `false` | OTLP exporter |
| `OMS_OBSERVABILITY_OTLP_ENDPOINT` | `http://localhost:4317` | |
| `OMS_OBSERVABILITY_SLO_TTFA_MS` | `800` | Time-to-first-audio objective |
| `OMS_OBSERVABILITY_SLO_BARGE_IN_MS` | `200` | |

---

## Profiles

### Local evaluation (zero dependencies)

```bash
OMS_ENV=dev
OMS_MOCK_MODE=true
OMS_OBSERVABILITY_LOG_FORMAT=console
```

### Single-node production

```bash
OMS_ENV=prod
OMS_MOCK_MODE=false
OMS_STORAGE_DATABASE_URL=postgresql+asyncpg://oms:***@db:5432/omnistream
OMS_STORAGE_REDIS_URL=redis://redis:6379/0
OMS_SECURITY_JWT_SECRET=<64 random bytes, base64>
OMS_SECURITY_API_KEY_SALT=<32 random bytes, base64>
OMS_SECURITY_CREDENTIAL_ENCRYPTION_KEY=<32 random bytes, base64>
OMS_OBSERVABILITY_LOG_FORMAT=json
OMS_PROVIDERS_DEFAULT=doubao
OMS_PROVIDERS_DOUBAO_API_KEY=<key>
```

### Multi-replica

Everything above, plus:

```bash
OMS_STORAGE_EVENT_BUS=redis
OMS_STORAGE_AUTO_MIGRATE=false     # migrate once from a job, not from every replica
OMS_OBSERVABILITY_TRACING_ENABLED=true
```

---

## Inspecting effective configuration

```bash
omnistream doctor          # validates config, checks connectivity, prints a report
curl localhost:8080/version
```

`doctor` never prints secret values — fields declared with `repr=False` are masked.

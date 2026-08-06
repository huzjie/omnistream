# Error codes

Every error crossing a process boundary — HTTP response, WebSocket `error` event, CLI
exit — carries a stable machine-readable code. Log messages change; codes do not.

## Format

```
OMS-<domain><serial>
     ^^      ^^
     |       └── two digits, sequential within the domain
     └────────── two digits, the subsystem
```

## Uniform error body

```json
{
  "code": "OMS-6003",
  "message": "provider timed out after 60s",
  "detail": {"provider": "doubao", "phase": "create_response"},
  "request_id": "req_01J8Z9Q1M4K3T7X2A6B5C4D3E2",
  "retryable": true
}
```

| Field | Always present | Meaning |
| --- | --- | --- |
| `code` | yes | The `OMS-XXNN` identifier |
| `message` | yes | Human-readable, English, never contains user content |
| `detail` | yes (may be `{}`) | Structured context — safe to log, never contains secrets |
| `request_id` | yes when available | Correlates with `X-Request-Id` and every log line |
| `retryable` | yes | Whether an identical retry can succeed |

---

## 10 — General

| Code | Name | HTTP | Retryable | When |
| --- | --- | --- | --- | --- |
| `OMS-1000` | `INTERNAL` | 500 | no | Unhandled exception; a bug |
| `OMS-1001` | `INVALID_REQUEST` | 400 | no | Schema or semantic validation failed |
| `OMS-1002` | `NOT_FOUND` | 404 | no | Resource does not exist or is not visible to this tenant |
| `OMS-1003` | `CONFLICT` | 409 | no | Version conflict, duplicate unique key |
| `OMS-1004` | `TIMEOUT` | 504 | **yes** | Internal stage exceeded its budget |
| `OMS-1005` | `UNAVAILABLE` | 503 | **yes** | Dependency down, shutting down, or not yet ready |
| `OMS-1006` | `NOT_IMPLEMENTED` | 501 | no | Endpoint or capability intentionally absent |

## 11 — Configuration

| Code | Name | HTTP | Retryable | When |
| --- | --- | --- | --- | --- |
| `OMS-1100` | `CONFIG_INVALID` | 400 | no | Value present but nonsensical; also decryption failure |
| `OMS-1101` | `CONFIG_MISSING` | 400 | no | Required setting absent (e.g. encryption key) |

## 20 — Authentication & authorization

| Code | Name | HTTP | Retryable | When |
| --- | --- | --- | --- | --- |
| `OMS-2000` | `UNAUTHENTICATED` | 401 | no | No credential supplied |
| `OMS-2001` | `INVALID_API_KEY` | 401 | no | Key unknown, disabled, or digest mismatch |
| `OMS-2002` | `TOKEN_EXPIRED` | 401 | no | JWT past `exp`; refresh and retry |
| `OMS-2003` | `FORBIDDEN` | 403 | no | Authenticated but lacks the permission |
| `OMS-2004` | `TENANT_SUSPENDED` | 403 | no | Billing or compliance hold |

## 21 — Quota & rate limiting

| Code | Name | HTTP | Retryable | When |
| --- | --- | --- | --- | --- |
| `OMS-2100` | `QUOTA_EXCEEDED` | 402 | no | Plan allowance consumed; upgrade or wait for reset |
| `OMS-2101` | `RATE_LIMITED` | 429 | **yes** | Too many requests; honour `Retry-After` |
| `OMS-2102` | `CONCURRENCY_LIMIT` | 429 | no | Too many simultaneous sessions |

## 30 — Protocol

| Code | Name | HTTP | Retryable | When |
| --- | --- | --- | --- | --- |
| `OMS-3000` | `PROTOCOL_VERSION_MISMATCH` | 400 | no | Client major ≠ server major |
| `OMS-3001` | `PROTOCOL_MALFORMED` | 400 | no | Bad envelope, bad magic, oversized frame |
| `OMS-3002` | `PROTOCOL_UNKNOWN_EVENT` | 400 | no | Unrecognised `type` from client |
| `OMS-3003` | `PROTOCOL_SEQ_GAP` | 500 | no | Detected sequence discontinuity |

## 40 — Media

| Code | Name | HTTP | Retryable | When |
| --- | --- | --- | --- | --- |
| `OMS-4000` | `TRANSPORT_CLOSED` | 500 | no | Peer disappeared mid-stream |
| `OMS-4001` | `CODEC_ERROR` | 500 | no | Opus/JPEG encode or decode failure |
| `OMS-4002` | `RESAMPLE_ERROR` | 500 | no | Unsupported rate conversion |
| `OMS-4003` | `CLOCK_DRIFT` | 500 | no | PTS drift beyond correction window |
| `OMS-4004` | `ICE_FAILED` | 500 | no | WebRTC connectivity establishment failed |
| `OMS-4005` | `MEDIA_STALLED` | 500 | no | No frames for longer than the stall threshold |

## 50 — Duplex & perception

| Code | Name | HTTP | Retryable | When |
| --- | --- | --- | --- | --- |
| `OMS-5000` | `SESSION_NOT_ACTIVE` | 500 | no | Operation on a closed or failed session |
| `OMS-5001` | `ILLEGAL_TURN_TRANSITION` | 409 | no | State machine invariant violated |
| `OMS-5002` | `ASR_FAILED` | 500 | no | Recogniser crashed or returned garbage |
| `OMS-5003` | `TTS_FAILED` | 500 | no | Synthesiser failed |
| `OMS-5004` | `VISION_FAILED` | 500 | no | Vision model failed |
| `OMS-5005` | `VAD_BACKEND_MISSING` | 500 | no | Selected VAD backend not installed — fall back to `energy` |

## 60 — Providers

| Code | Name | HTTP | Retryable | When |
| --- | --- | --- | --- | --- |
| `OMS-6000` | `PROVIDER_NOT_FOUND` | 404 | no | Name not in the registry |
| `OMS-6001` | `PROVIDER_UNAUTHORIZED` | 401 | no | Upstream rejected the credential |
| `OMS-6002` | `PROVIDER_RATE_LIMITED` | 429 | **yes** | Upstream 429; fallback chain engages |
| `OMS-6003` | `PROVIDER_TIMEOUT` | 504 | **yes** | Upstream exceeded `providers.timeout_s` |
| `OMS-6004` | `PROVIDER_BAD_RESPONSE` | 500 | **yes** | Malformed or truncated upstream stream |
| `OMS-6005` | `PROVIDER_CAPABILITY_UNSUPPORTED` | 501 | no | e.g. video requested from an audio-only provider |
| `OMS-6006` | `PROVIDER_CIRCUIT_OPEN` | 503 | no | Circuit breaker tripped; retry after cooldown |

## 70 — Tools

| Code | Name | HTTP | Retryable | When |
| --- | --- | --- | --- | --- |
| `OMS-7000` | `TOOL_NOT_FOUND` | 404 | no | Model called an unregistered tool |
| `OMS-7001` | `TOOL_INVALID_ARGS` | 400 | no | Arguments fail the tool's JSON Schema |
| `OMS-7002` | `TOOL_TIMEOUT` | 504 | no | Exceeded `DEFAULT_TOOL_TIMEOUT_MS` |
| `OMS-7003` | `TOOL_EXECUTION_FAILED` | 500 | no | Tool raised |

## 80 — Memory

| Code | Name | HTTP | Retryable | When |
| --- | --- | --- | --- | --- |
| `OMS-8000` | `MEMORY_STORE_ERROR` | 500 | **yes** | Vector store unreachable |
| `OMS-8001` | `EMBEDDING_FAILED` | 500 | no | Embedder crashed or dimension mismatch |
| `OMS-8002` | `MEMORY_ITEM_NOT_FOUND` | 404 | no | Deleting or updating a missing item |

## 90 — Governance

| Code | Name | HTTP | Retryable | When |
| --- | --- | --- | --- | --- |
| `OMS-9000` | `RECORDING_FAILED` | 500 | no | Object store write failed |
| `OMS-9001` | `PII_POLICY_VIOLATION` | 403 | no | Content violates the tenant's PII policy |
| `OMS-9002` | `CONTENT_BLOCKED` | 403 | no | Safety filter blocked the content |

---

## Client handling guide

```python
RETRY_WITH_BACKOFF = {"OMS-1004", "OMS-1005", "OMS-2101",
                      "OMS-6002", "OMS-6003", "OMS-6004", "OMS-8000"}

if body["retryable"]:
    sleep(backoff_with_jitter(attempt))
    retry()
elif body["code"].startswith("OMS-20"):
    reauthenticate()
elif body["code"] == "OMS-2100":
    surface_upgrade_prompt()
else:
    report(body["request_id"])
```

Prefer the `retryable` boolean over hard-coding the set — it is derived server-side from
`core.error_codes.RETRYABLE` and stays correct as codes are added.

## Adding a new code

1. Add the member to `ErrorCode` in `omnistream/core/error_codes.py` (never reuse a number).
2. Map it in `HTTP_STATUS_MAP` (omitting it means 500).
3. If a retry can plausibly succeed, add it to `RETRYABLE`.
4. Add a row to this document — CI checks that every enum member appears here.

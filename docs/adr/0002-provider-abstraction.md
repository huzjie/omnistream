# ADR 0002 — Provider abstraction as a single-direction event stream

* **Status:** Accepted
* **Date:** 2026-07-13
* **Deciders:** Core team
* **Related:** ADR 0001

## Context

OmniStream has to sit in front of two structurally different kinds of backend and make
them indistinguishable to the duplex engine:

1. **Native realtime APIs** — Doubao / SeedRealtime, OpenAI Realtime, Gemini Live,
   Qwen-Omni, StepFun, MiniMax. Bidirectional WebSocket, server-driven event streams,
   built-in VAD and interruption, audio in / audio out.
2. **Composed pipelines** — a streaming ASR + a text LLM + a streaming TTS, wired
   together locally. Three independent streams that we must fuse into one coherent
   turn, including making interruption work across all three.

Each vendor also disagrees about nearly everything: event naming, whether the server
does VAD, whether cancellation is acknowledged, whether audio arrives as base64 JSON or
binary frames, sample rates, and how usage is reported.

The abstraction has to be narrow enough that a new adapter is a day of work, and strict
enough that the duplex engine never grows a `if provider == "openai"` branch.

## Decision

**One abstract port, `BaseRealtimeProvider`, whose output is a single
`AsyncIterator[ProviderEvent]`.**

```python
class BaseRealtimeProvider(ABC):
    # inbound (engine → provider): plain coroutines
    async def connect(self, session: SessionConfig) -> None: ...
    async def append_audio(self, chunk: AudioChunk) -> None: ...
    async def commit_audio(self) -> None: ...
    async def send_text(self, text: str) -> None: ...
    async def send_video_frame(self, frame: VideoFrame) -> None: ...
    async def create_response(self, params: GenerationParams) -> None: ...
    async def cancel_response(self) -> None: ...
    async def close(self) -> None: ...

    # outbound (provider → engine): exactly one stream
    def events(self) -> AsyncIterator[ProviderEvent]: ...

    # introspection
    @property
    def capabilities(self) -> ProviderCapabilities: ...
    async def health(self) -> ProviderHealth: ...
```

Supporting rules:

1. **`ProviderEvent` is a `@dataclass(slots=True)`, not a Pydantic model.** It is
   allocated 50+ times per second per session; validation overhead and dict allocation
   are not acceptable on that path. Pydantic is used at the *protocol* boundary, where
   messages cross a trust boundary; providers are internal.
2. **`cancel_response()` must take effect within `CANCEL_BUDGET_MS = 100`.** Adapters
   that cannot get an upstream acknowledgement in time must locally stop forwarding
   audio events and mark the response cancelled. The engine measures this and records
   an SLO violation.
3. **Capabilities are declarative.** `ProviderCapabilities.missing_for(config)` tells
   the router *before* connecting whether a session's requirements can be met, which is
   what makes the fallback chain deterministic instead of trial-and-error.
4. **Registration is decentralised.** `@register_provider` on the class plus a module in
   `omnistream/providers/<name>/` is the whole integration; no central list to edit,
   no merge conflicts between parallel workstreams.
5. **A reference implementation ships in-tree.** `MockRealtimeProvider` implements
   every method with deterministic synthesised audio and a log-normal latency model —
   it is both the zero-dependency demo path and the conformance yardstick for new
   adapters.

## Consequences

### Positive

* **The duplex engine has one consumer loop.** `async for event in provider.events()`
  — no callback ordering hazards, no partially-observed state, natural backpressure,
  and cancellation propagates through the normal `asyncio` mechanism.
* **Composed providers are ordinary providers.** The engine cannot distinguish a native
  realtime API from ASR+LLM+TTS glue, so features like barge-in are implemented once.
* **Testing is trivial.** Any test can hand the engine a provider that yields a scripted
  event list.
* **Fallback is mechanical.** On `PROVIDER_TIMEOUT` / `CIRCUIT_OPEN`, the router walks
  `providers.fallback_chain` and re-establishes with the same `SessionConfig`.

### Negative

* **Vendor-specific superpowers get flattened.** If one vendor exposes, say, native
  emotion tags, it either becomes a generic `ProviderEvent` kind or lives in
  `event.extra`. *Mitigation:* `ProviderEvent.extra: dict` is an explicit escape hatch,
  and genuinely useful capabilities get promoted to first-class after they appear in
  two vendors.
* **Composed adapters carry real complexity.** Aligning three streams, and cancelling
  all three within 100 ms, is the hardest code in the repo. *Mitigation:* it is written
  **once**, in `providers/composed/`, rather than once per vendor.
* **A misbehaving adapter can stall the engine.** *Mitigation:*
  `utils.asyncio_utils.iter_with_timeout` wraps `events()` with an idle timeout, and the
  circuit breaker opens after repeated stalls.

## Alternatives considered

| Option | Why not |
| --- | --- |
| **Callback registration (`on_audio`, `on_text`, …)** | Ordering between callbacks is unspecified, error propagation is awkward, and cancellation semantics are ad hoc. |
| **Two streams (audio stream + control stream)** | Interleaving matters: "text delta then audio delta then done" must stay ordered. Two streams reintroduce the ordering problem the engine was trying to avoid. |
| **A queue the provider writes into** | Same as one stream, but the lifecycle and completion signalling become the caller's problem, and backpressure gets hand-rolled. |
| **LangChain/LiteLLM-style universal client** | Built around request/response text completion; realtime bidirectional audio with interruption is not the shape they model. |

## Conformance checklist for a new adapter

- [ ] `capabilities` accurately declares audio-in/out, video-in, tools, and native VAD.
- [ ] `events()` yields a terminal event exactly once per response.
- [ ] `cancel_response()` stops audio emission within 100 ms, measured.
- [ ] Upstream errors are translated to `OMS-60xx` codes, never leaked raw.
- [ ] Credentials are read from settings, never logged, never included in `extra`.
- [ ] `close()` is idempotent and releases the socket even if a response is in flight.

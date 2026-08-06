# ADR 0001 — Python-first modular monolith

* **Status:** Accepted
* **Date:** 2026-07-13
* **Deciders:** Core team
* **Supersedes:** —

## Context

OmniStream must deliver sub-200 ms barge-in on realtime audio while remaining
approachable to the people who actually build voice agents. Those two goals pull in
opposite directions:

* The realtime path is latency-sensitive and concurrency-heavy — the classic argument
  for Go or Rust.
* The AI ecosystem the platform must integrate with — FunASR, Whisper, Silero VAD,
  CosyVoice, sentence-transformers, every vendor SDK — is overwhelmingly Python.
  A non-Python core means writing and maintaining an FFI shim per model.

We also have to be honest about the scale we are designing for. A self-hosted
deployment typically serves tens to low hundreds of concurrent sessions, not millions.
Premature microservice decomposition would multiply the operational surface (service
discovery, distributed tracing across ten hops, N deployment pipelines) for an audience
that mostly wants `docker compose up`.

## Decision

**Build a Python 3.11+ modular monolith with hexagonal boundaries.**

Concretely:

1. One deployable process (plus an optional Celery worker for non-realtime jobs).
2. `asyncio` throughout the realtime path; no threads except a bounded pool for
   numpy/model inference via `asyncio.to_thread`.
3. Hard internal seams: the layering rules in `docs/architecture.md` are enforced in CI,
   so a package's dependencies are always a strict subset of the layers below it.
4. Ports & adapters for everything external: providers, VAD backends, ASR/TTS engines,
   memory stores, object stores. Each port is an ABC in the layer that *uses* it, and
   adapters live in their own subpackage.
5. Latency discipline as an explicit budget, not a hope: every stage has a documented
   millisecond allowance, measured per turn and exported as metrics.

## Consequences

### Positive

* **Model integration is free.** Adding FunASR or Silero is `pip install` plus one
  adapter class, not a gRPC service with its own Dockerfile.
* **One process to reason about.** A single log stream, a single `request_id`, a single
  place where a turn's latency budget is spent. Debugging realtime bugs across five
  services is materially harder.
* **Extraction stays possible.** Because the seams are real, `perception` or `memory`
  can become a service later. The port interface becomes the RPC contract; callers
  don't change.
* **Contributor onboarding is one clone and one `pip install`.**

### Negative

* **GIL exposure.** CPU-bound work (resampling, VAD inference, JPEG encode) must be
  pushed off the loop or it will add jitter to every other session in the process.
  *Mitigation:* `run_in_thread` for anything above ~2 ms; numpy releases the GIL for
  the heavy parts; horizontal scaling is per-process, not per-thread.
* **Vertical scaling ceiling per process.** A single worker saturates well before a Go
  equivalent would. *Mitigation:* the process is stateless with respect to sessions
  once `event_bus=redis`, so scaling is "run more replicas", which is the answer we'd
  give anyway.
* **Blast radius.** A crash takes down every session on that replica.
  *Mitigation:* strict bounded queues, circuit breakers per provider, `critical=False`
  lifecycle hooks for optional subsystems, and an explicit rule that no background task
  may swallow exceptions silently.
* **Dependency weight.** A full install pulls torch-adjacent packages.
  *Mitigation:* three requirement tiers — `requirements.txt` (core, no models),
  `requirements-local-models.txt` (opt-in heavy), `requirements-dev.txt`.

## Alternatives considered

| Option | Why not |
| --- | --- |
| **Go core + Python sidecar for models** | Two languages, two toolchains, an IPC hop on the hot path, and every new model integration needs work in both. The latency win is real but smaller than the media path's inherent 20 ms frame granularity. |
| **Rust core** | Best latency profile, worst ecosystem fit and contributor pool for an AI platform. |
| **Python microservices from day one** | Same GIL constraints, plus network hops, plus ten deployment units. Strictly worse for the target scale. |
| **Node/TypeScript core** | Good WebSocket story, poor numeric/ML story, would still need a Python sidecar. |

## Revisit when

* A single tenant regularly needs >500 concurrent sessions per replica, **or**
* profiling shows GIL contention (not I/O wait) is the dominant p99 contributor, **or**
* the perception layer stabilises enough that extracting it stops being churn.

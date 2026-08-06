# ADR 0003 — Custom binary WebSocket frame for media

* **Status:** Accepted
* **Date:** 2026-07-13
* **Deciders:** Core team
* **Related:** ADR 0002, `docs/protocol-spec.md` §5

## Context

At the platform's atomic frame size — 20 ms of 16 kHz mono pcm16 = **640 bytes** —
audio flows at **50 frames per second per direction per session**. Downlink at 24 kHz
is 960 bytes per frame.

The obvious approach, base64 inside the JSON envelope, has measurable costs:

| Concern | Base64-in-JSON | Raw binary |
| --- | --- | --- |
| Bytes on the wire (640 B payload) | ~853 B payload + ~180 B envelope ≈ **1033 B** | 640 B + 7 B + ~90 B header ≈ **737 B** |
| Bandwidth overhead | **+61 %** | +15 % |
| Per-frame allocations | str → bytes → decode → bytes (4+) | 1 slice, zero-copy into numpy |
| CPU at 100 sessions | 5 000 base64 decodes/s | memoryview slicing |

At 100 concurrent duplex sessions that is roughly 30 MB/s of avoidable traffic and a
continuous stream of short-lived objects for the GC to chase — exactly the kind of
background jitter that shows up as p99 latency in a realtime system.

We also need something a browser can produce and consume trivially. WebSocket already
distinguishes text and binary frames, and browsers hand JS an `ArrayBuffer` for the
latter, so the transport-level split is free.

## Decision

**Media travels as WebSocket binary frames with a 7-byte fixed prefix plus a compact
JSON header. Control travels as WebSocket text frames with the JSON envelope.**

```
offset  size  field
0       4     magic       = "OMS1"
4       1     kind        1 = audio, 2 = video
5       2     header_len  uint16, little-endian
7       H     header      UTF-8 JSON, ≤ 4096 bytes
7+H     N     payload     raw media bytes, ≤ 4 MiB
```

Design points, each deliberate:

* **Magic bytes.** Cheap, unambiguous rejection of anything that isn't ours — including
  a client that mistakenly sends a text frame as binary. Costs 4 bytes.
* **`kind` as a byte, not a string.** Discriminating audio vs video must not require
  parsing JSON first.
* **JSON header rather than a fixed binary struct.** The header is ~90 bytes and parsed
  by `orjson` in well under a microsecond, but it stays *evolvable*: adding
  `"lang":"zh"` or `"vad":0.91` needs no version bump and no lockstep client update.
  A fixed struct would have bought ~80 bytes per frame and cost us every future field.
* **Explicit size caps.** `MAX_HEADER = 4 KiB`, `MAX_PAYLOAD = 4 MiB`, both checked
  *before* allocation. A malicious `header_len` cannot cause a large allocation.
* **`pts` in the header, not implied by arrival order.** Networks reorder; the jitter
  buffer needs an authoritative timestamp. `seq` is carried separately for loss
  detection.

## Consequences

### Positive

* ~30 % less bandwidth than base64 JSON, and near-zero per-frame allocation:
  `numpy.frombuffer(memoryview(raw)[7+H:], dtype="<i2")` is a view, not a copy.
* Trivially debuggable: `hexdump | head -c 7` tells you kind and header length, and the
  header itself is human-readable JSON.
* Language-agnostic — the Python, TypeScript and Go SDKs each implement the codec in
  under 60 lines.
* The text/binary split means a client can handle control messages and media on
  separate code paths without sniffing content.

### Negative

* **Not a standard.** Anyone integrating without our SDK must implement the codec.
  *Mitigation:* it is 20 lines, fully specified in `docs/protocol-spec.md`, and the
  JSON fallback (`input_audio.append` with base64) remains supported for
  low-throughput or exploratory clients.
* **Two parsing paths to maintain and fuzz.** *Mitigation:* both live in
  `omnistream/protocol/`, and `decode_frame` is fuzz-tested against truncated,
  oversized and lying-length inputs.
* **Header duplication.** `sid` repeats on every frame (~20 bytes).
  *Mitigation:* accepted deliberately — it makes frames self-describing, which matters
  for recording, replay and multiplexed transports where per-connection state is not
  available.

## Alternatives considered

| Option | Why not |
| --- | --- |
| **Base64 in the JSON envelope** | +61 % bandwidth, 4+ allocations per frame, GC pressure at 50 fps × N sessions. Kept only as an optional compatibility path. |
| **Protobuf / FlatBuffers for the whole protocol** | Adds a compiler to every SDK's build and makes the control plane far less debuggable. The bandwidth win over "binary payload + small JSON header" is a few dozen bytes per frame. |
| **MessagePack envelope with binary payload** | Nearly as compact, but loses `curl`/devtools readability for the control plane and adds a dependency to every SDK. |
| **Separate WebSocket connection per media stream** | Multiplies connection count, breaks ordering guarantees between control and media, and complicates auth and reconnect. |
| **WebRTC only** | Correct for browsers on lossy networks and fully supported, but too heavy a dependency to *require* for server-to-server and IoT clients. Both transports coexist; this ADR is about the WebSocket path. |

## Revisit when

* Median session bitrate rises far enough that the ~90-byte JSON header becomes a
  meaningful fraction (e.g. sub-10 ms frames), **or**
* a second major transport adopts a natural framing that makes ours redundant.

# Relay body-queue guardrails

The server bounds the memory one tunneled response can pin in its body queue.
A body queue holds `ResponseBodyFrame` chunks between the runner tunnel and the
session consumer, so a slow consumer can otherwise grow it without limit.

## Behavior

The queue stays unbounded by frame count, and `put` never blocks. Nothing is
dropped below the fuse.

- **8 MiB warning:** the first frame that pushes a request's queued body bytes
  past `8 * 1024 * 1024` logs one `WARNING` for that request. The request keeps
  streaming and later frames do not warn again.
- **256 MiB fuse:** the frame that pushes a request's queued body bytes past
  the fuse aborts only that request. The abort logs an `ERROR`, raises
  `BodyQueueFuseExceeded` (an `httpx.RemoteProtocolError`) through that
  request's body stream, and moves the request onto the transport-failure path,
  so only the offending relay retries. Other requests on the same runner socket
  keep streaming and the socket stays registered.

Bytes are counted as UTF-8 body bytes, matching what the queue holds.

## Configuration

`OMNIGENT_RELAY_BODY_QUEUE_FUSE_BYTES` sets the per-request fuse in bytes. It
defaults to 268435456 (256 MiB). An unset, non-numeric, or non-positive value
falls back to the default, so a typo cannot disable the fuse. The variable is
read when the counter module builds its recorder, which happens at server
import, so set it in the server process environment.

The warning threshold is not configurable; tests override both thresholds
through `RelayStats(warn_threshold_bytes=..., fuse_threshold_bytes=...)`.

## Observing guardrails

Guardrail state rides the same `relay_stats` record as the Phase 0 counters.
The server logs `relay_stats {json}` on the `omnigent.server.relay_stats`
logger every `OMNIGENT_RELAY_STATS_INTERVAL_S` seconds (default 10). The
guardrail fields:

| Field | Meaning |
| --- | --- |
| `body_queue.depth`, `body_queue.bytes` | Frames and UTF-8 bytes queued across all requests now. |
| `body_queue.max_depth`, `body_queue.max_bytes` | Peaks since the server started. |
| `body_queue.warned_requests` | Requests that logged their one pressure warning. |
| `body_queue.warn_bytes` | Active warn threshold in bytes. |
| `body_queue.fuse.trips` | Requests aborted by the fuse. |
| `body_queue.fuse.limit_bytes` | Active fuse threshold in bytes. |
| `body_queue.requests[id]` | Per-request `depth`, `bytes`, and `warned` for the busiest ten requests. |
| `outbound_queue.depth`, `outbound_queue.max_depth` | Frames waiting behind the server's outbound sender, summed per runner, plus its peak. |
| `outbound_queue.runners`, `outbound_queue.runners_tracked` | Per-runner outbound depth and how many runner queues the gauge tracks. |
| `outbound_queue.dropped_samples` | Samples skipped because the runner map was at its 256-runner bound. |
| `session_stream.subscriber_overflows` | Session outcome streams closed because their subscriber queue passed 1024 pending events. |

The per-request map is bounded at 512 tracked requests; `body_queue.dropped_requests`
counts frames from requests past that bound. A request rejected at the bound is
neither counted nor guarded, so its memory is untracked rather than misreported.

`OMNIGENT_RELAY_STATS_INTERVAL_S=0` disables the record. Read the counters
in-process with `omnigent.server.relay_stats.snapshot()`.

The `dev/loadtest/` T6 harness reads the fuse and overflow counts from the
`relay_stats` payload and reports them with `source = "counter"`. It falls back
to matching server log lines only for older builds that lack the fields.

## Verification

```sh
uv run --no-sync pytest -q tests/server/routes/test_relay_stats.py \
  tests/dev/test_t6_harness.py --reruns 0
```

For a live check, start a disposable local server with
`OMNIGENT_RELAY_BODY_QUEUE_FUSE_BYTES=1048576`, run one turn whose streamed
reply exceeds 1 MiB while a second turn streams normally, and confirm the first
turn fails with the fuse `ERROR` row while the second completes.

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

Both guardrails apply to every request. Their byte totals live on the request's
own state, which dies with the request, so the 512-request bound on the
telemetry map never limits them: the 513th concurrent request still warns and
still fuses. A single frame that crosses both thresholds logs its `WARNING`
before the fuse `ERROR`, so `body_queue.warned_requests` and
`body_queue.fuse.trips` agree with the log. A fused frame is never queued, so it
moves no body-queue gauge.

Bytes are counted as UTF-8 body bytes, matching what the queue holds. Consuming
a frame lowers the request's queued total, so a stream that keeps draining stays
below the thresholds.

## Configuration

`OMNIGENT_RELAY_BODY_QUEUE_WARN_BYTES` sets the per-request warning threshold in
bytes. It defaults to 8388608 (8 MiB). `OMNIGENT_RELAY_BODY_QUEUE_FUSE_BYTES`
sets the per-request fuse in bytes. It defaults to 268435456 (256 MiB). An
unset, non-numeric, or non-positive value falls back to the default, so a typo
cannot disable a guardrail. The server reads both when it creates a request's
guard state, so set them in the server process environment.

## Observing guardrails

Guardrail state rides the same `relay_stats` record as the Phase 0 counters.
The server logs `relay_stats {json}` on the `omnigent.server.relay_stats`
logger every `OMNIGENT_RELAY_STATS_INTERVAL_S` seconds (default 10). The
guardrail fields:

| Field | Meaning |
| --- | --- |
| `body_queue.depth`, `body_queue.bytes` | Frames and UTF-8 bytes queued across all requests now. |
| `body_queue.max_depth`, `body_queue.max_bytes` | Peaks since the server started. |
| `body_queue.warned_requests` | Requests that logged their one pressure warning, including requests the map does not track. |
| `body_queue.warn_bytes` | Active warn threshold in bytes. |
| `body_queue.fuse.trips` | Requests aborted by the fuse, including requests the map does not track. |
| `body_queue.fuse.limit_bytes` | Active fuse threshold in bytes. |
| `body_queue.requests[id]` | Per-request `depth`, `bytes`, and `warned` for the busiest ten requests. |
| `outbound_queue.depth`, `outbound_queue.max_depth` | Frames waiting behind the server's outbound sender, summed per runner, plus its peak. |
| `outbound_queue.runners`, `outbound_queue.runners_tracked` | Per-runner outbound depth and how many runner queues the gauge tracks. |
| `outbound_queue.dropped_samples` | Samples skipped because the runner map was at its 256-runner bound. |
| `session_stream.subscriber_overflows` | Session outcome streams closed because their subscriber queue passed 1024 pending events. |

The per-request map is bounded at 512 tracked requests; `body_queue.dropped_requests`
counts frames from requests past that bound. The bound is telemetry only: a
request rejected at the bound is still warned and still fused, but its frames
move no gauge, so the aggregate depth and bytes stay the sum over tracked
requests.

`OMNIGENT_RELAY_STATS_INTERVAL_S=0` disables the record. Read the counters
in-process with `omnigent.server.relay_stats.snapshot()`.

The `dev/loadtest/` T6 harness reads `body_queue.fuse.trips` and
`session_stream.subscriber_overflows` from the `relay_stats` payload and
reports them with `source = "counter"`. It reads only those paths, so a request
id that contains `fuse` or `overflow` cannot inflate the counts. It falls back
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

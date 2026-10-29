"""Unit tests for the relay counters and queue guardrails.

Each test pins one observable counter, its bound, a guardrail threshold, or
the structured ``relay_stats`` log record that the load harness reads.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from omnigent.runner.transports.ws_tunnel.frames import (
    HelloFrame,
    ResponseBodyFrame,
    ResponseEndFrame,
    ResponseHeadFrame,
    decode_body,
)
from omnigent.runner.transports.ws_tunnel.registry import (
    BodyQueueFuseExceeded,
    TunnelRegistry,
)
from omnigent.runner.transports.ws_tunnel.transport import _TunneledByteStream
from omnigent.runtime import session_stream
from omnigent.server import presence, relay_stats, shutdown_state
from omnigent.server.relay_stats import (
    DEFAULT_RELAY_STATS_INTERVAL_S,
    RELAY_STATS_INTERVAL_ENV_VAR,
    RELAY_STATS_LOG_EVENT,
    SOURCE_EVENT_BATCH,
    SOURCE_HTTP_POST,
    SOURCE_RELAY_SSE,
    BodyQueueSnapshot,
    RelayStats,
)
from omnigent.server.routes._sessions.helpers import _relay_persist
from omnigent.server.routes.runner_tunnel import _sender_loop
from omnigent.server.routes.sessions import _stream_live_events

SESSION_ID = "conv_relay_stats"


class _FakeClock:
    """Deterministic monotonic clock for rate-window assertions."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _NoopWS:
    """Minimal WebSocket fake for registry tests that never send."""

    async def send_text(self, data: str) -> None:
        pass

    async def receive_text(self) -> str:
        return await asyncio.Future()


class _ConnectedRequest:
    """Minimal request stand-in: the client stays connected."""

    async def is_disconnected(self) -> bool:
        return False


def _hello() -> HelloFrame:
    return HelloFrame(runner_version="0.1.0", frame_protocol_version=1, harnesses=[], envs=[])


@pytest.fixture(autouse=True)
def _reset_relay_stats_state() -> Any:
    """Isolate module-global counters, subscribers, presence, and shutdown state."""
    relay_stats.reset_for_tests()
    presence.reset_for_tests()
    shutdown_state.reset_for_tests()
    session_stream._subscribers.clear()
    yield
    relay_stats.reset_for_tests()
    presence.reset_for_tests()
    shutdown_state.reset_for_tests()
    session_stream._subscribers.clear()


def test_body_queue_tracks_depth_and_bytes_per_request() -> None:
    """Enqueue/dequeue update per-request gauges and all-time maxima."""
    stats = RelayStats()
    stats.record_body_queue_enqueue("req-1", byte_count=100)
    stats.record_body_queue_enqueue("req-1", byte_count=200)
    stats.record_body_queue_enqueue("req-2", byte_count=50)

    snapshot = stats.snapshot()

    assert snapshot.body_queue.depth == 3
    assert snapshot.body_queue.bytes == 350
    assert snapshot.body_queue.max_depth == 3
    assert snapshot.body_queue.max_bytes == 350
    assert snapshot.body_queue.requests["req-1"].depth == 2
    assert snapshot.body_queue.requests["req-1"].bytes == 300
    assert snapshot.body_queue.requests["req-2"].depth == 1

    stats.record_body_queue_dequeue("req-1", byte_count=100)
    after_dequeue = stats.snapshot()
    assert after_dequeue.body_queue.depth == 2
    assert after_dequeue.body_queue.bytes == 250
    assert after_dequeue.body_queue.requests["req-1"].depth == 1
    # Maxima are all-time, not current values.
    assert after_dequeue.body_queue.max_depth == 3
    assert after_dequeue.body_queue.max_bytes == 350


def test_body_queue_release_drops_request_and_ignores_unknown_ids() -> None:
    """Releasing a request drops its gauge; unknown ids are no-ops."""
    stats = RelayStats()
    stats.record_body_queue_enqueue("req-1", byte_count=10)
    stats.record_body_queue_released("req-1")
    stats.record_body_queue_released("req-1")
    stats.record_body_queue_dequeue("missing", byte_count=5)

    snapshot = stats.snapshot()

    assert snapshot.body_queue.depth == 0
    assert snapshot.body_queue.bytes == 0
    assert snapshot.body_queue.requests == {}


def test_body_queue_map_is_bounded() -> None:
    """Requests beyond the tracking bound are dropped, not tracked unbounded."""
    stats = RelayStats(max_tracked_requests=2)
    stats.record_body_queue_enqueue("req-1", byte_count=1)
    stats.record_body_queue_enqueue("req-2", byte_count=1)
    stats.record_body_queue_enqueue("req-3", byte_count=1)

    snapshot = stats.snapshot()

    assert set(snapshot.body_queue.requests) == {"req-1", "req-2"}
    assert snapshot.body_queue.dropped_requests == 1
    assert snapshot.body_queue.depth == 2


def _assert_aggregate_is_the_tracked_sum(snapshot: BodyQueueSnapshot) -> None:
    """The aggregate gauges must equal the sum over the tracked requests."""
    assert snapshot.depth == sum(request.depth for request in snapshot.requests.values())
    assert snapshot.bytes == sum(request.bytes for request in snapshot.requests.values())
    assert snapshot.depth >= 0
    assert snapshot.bytes >= 0


def test_saturated_request_map_keeps_rejected_requests_untracked() -> None:
    """A request rejected at the bound stays untracked until its stream closes."""
    stats = RelayStats()
    for index in range(512):
        stats.record_body_queue_enqueue(f"req-{index}", byte_count=100)

    # Past the bound the new request is untracked, so its frames count nowhere.
    stats.record_body_queue_enqueue("req-late", byte_count=100)
    stats.record_body_queue_enqueue("req-late", byte_count=100)
    saturated = stats.snapshot().body_queue
    assert saturated.depth == 512
    assert saturated.bytes == 51200
    assert "req-late" not in saturated.requests
    assert saturated.dropped_requests == 2
    _assert_aggregate_is_the_tracked_sum(saturated)

    # A slot frees, yet the rejected request is not re-admitted: its two older
    # uncounted frames may still be queued ahead of any new counted frame.
    stats.record_body_queue_released("req-0")
    stats.record_body_queue_enqueue("req-late", byte_count=10)
    after_slot = stats.snapshot().body_queue
    assert "req-late" not in after_slot.requests
    assert after_slot.depth == 511
    assert after_slot.bytes == 51100
    assert after_slot.dropped_requests == 3
    _assert_aggregate_is_the_tracked_sum(after_slot)

    # Draining the older uncounted frames cannot touch any counted gauge.
    stats.record_body_queue_dequeue("req-late", byte_count=100)
    stats.record_body_queue_dequeue("req-late", byte_count=100)
    drained = stats.snapshot().body_queue
    assert "req-late" not in drained.requests
    assert drained.requests["req-1"].bytes == 100
    assert drained.depth == 511
    assert drained.bytes == 51100
    _assert_aggregate_is_the_tracked_sum(drained)

    # Only after the request closes (its queue has drained) is a new frame
    # counted, and then the gauge equals exactly that counted frame.
    stats.record_body_queue_released("req-late")
    stats.record_body_queue_enqueue("req-late", byte_count=10)
    readmitted = stats.snapshot().body_queue
    assert readmitted.requests["req-late"].depth == 1
    assert readmitted.requests["req-late"].bytes == 10
    assert readmitted.depth == 512
    assert readmitted.bytes == 51110
    _assert_aggregate_is_the_tracked_sum(readmitted)


def test_events_counted_by_source_type_and_session() -> None:
    """Every ingest source contributes to the mix and per-session type counts."""
    stats = RelayStats()
    stats.record_event(SESSION_ID, "response.output_text.delta", source=SOURCE_RELAY_SSE)
    stats.record_event(SESSION_ID, "response.output_text.delta", source=SOURCE_RELAY_SSE)
    stats.record_event(SESSION_ID, "response.completed", source=SOURCE_RELAY_SSE)
    stats.record_event("conv_other", "external_conversation_item", source=SOURCE_EVENT_BATCH)
    stats.record_event("conv_other", "message", source=SOURCE_HTTP_POST)

    snapshot = stats.snapshot()

    assert snapshot.events_by_source == {
        SOURCE_RELAY_SSE: 3,
        SOURCE_EVENT_BATCH: 1,
        SOURCE_HTTP_POST: 1,
    }
    assert snapshot.events_by_type["response.output_text.delta"] == 2
    assert snapshot.events_by_session_by_type[SESSION_ID] == {
        "response.output_text.delta": 2,
        "response.completed": 1,
    }
    assert snapshot.events_by_session_by_type["conv_other"] == {
        "external_conversation_item": 1,
        "message": 1,
    }


def test_event_rates_use_the_snapshot_window() -> None:
    """The snapshot reports per-second rates and then starts a new window."""
    clock = _FakeClock()
    stats = RelayStats(clock=clock)
    for _ in range(3):
        stats.record_event(SESSION_ID, "response.output_text.delta", source=SOURCE_RELAY_SSE)
    clock.now += 2.0

    first = stats.snapshot()

    assert first.window_seconds == pytest.approx(2.0)
    assert first.events_by_type_per_second["response.output_text.delta"] == pytest.approx(1.5)
    assert first.events_per_second_by_session_by_type[SESSION_ID][
        "response.output_text.delta"
    ] == pytest.approx(1.5)

    clock.now += 1.0
    second = stats.snapshot()

    assert second.window_seconds == pytest.approx(1.0)
    assert "response.output_text.delta" not in second.events_by_type_per_second
    # Cumulative counters survive the window rollover.
    assert second.events_by_type["response.output_text.delta"] == 3


def test_per_session_event_maps_are_bounded() -> None:
    """Session and per-session event-type maps evict or fold at their bounds."""
    stats = RelayStats(max_sessions=2, max_event_types_per_session=2)
    stats.record_event("conv_1", "response.output_text.delta", source=SOURCE_RELAY_SSE)
    stats.record_event("conv_2", "response.output_text.delta", source=SOURCE_RELAY_SSE)
    stats.record_event("conv_3", "response.output_text.delta", source=SOURCE_RELAY_SSE)
    stats.record_event("conv_3", "response.completed", source=SOURCE_RELAY_SSE)
    stats.record_event("conv_3", "response.created", source=SOURCE_RELAY_SSE)
    stats.record_event("conv_3", "response.failed", source=SOURCE_RELAY_SSE)

    snapshot = stats.snapshot()

    assert set(snapshot.events_by_session_by_type) == {"conv_2", "conv_3"}
    assert snapshot.dropped_sessions == 1
    # The oldest session was evicted, and the third distinct type folds into
    # the bounded "other" bucket.
    assert snapshot.events_by_session_by_type["conv_3"] == {
        "response.output_text.delta": 1,
        "response.completed": 1,
        "other": 2,
    }


def test_subscriber_serialization_totals() -> None:
    """Per-subscriber serialization time accumulates count, total, and max."""
    stats = RelayStats()
    stats.record_subscriber_serialization(SESSION_ID, 0.002)
    stats.record_subscriber_serialization(SESSION_ID, 0.004)
    stats.record_subscriber_serialization("conv_other", 0.001)

    snapshot = stats.snapshot()

    assert snapshot.subscriber_serialization_count == 3
    assert snapshot.subscriber_serialization_seconds_total == pytest.approx(0.007)
    assert snapshot.subscriber_serialization_seconds_max == pytest.approx(0.004)
    session = snapshot.subscriber_serialization_by_session[SESSION_ID]
    assert session.count == 2
    assert session.seconds_total == pytest.approx(0.006)
    assert session.seconds_max == pytest.approx(0.004)


def test_relay_stretch_totals() -> None:
    """Relay stretches between reader suspensions accumulate count, total, max."""
    stats = RelayStats()
    stats.record_relay_stretch(SESSION_ID, 0.010)
    stats.record_relay_stretch(SESSION_ID, 0.030)

    snapshot = stats.snapshot()

    assert snapshot.relay_stretch_count == 2
    assert snapshot.relay_stretch_seconds_total == pytest.approx(0.040)
    assert snapshot.relay_stretch_seconds_max == pytest.approx(0.030)
    session = snapshot.relay_stretch_by_session[SESSION_ID]
    assert session.count == 2
    assert session.seconds_total == pytest.approx(0.040)


def test_relay_persist_failure_count() -> None:
    """Persist failures count globally, per session, and never raise."""
    stats = RelayStats()
    stats.record_relay_persist_failure(SESSION_ID)
    stats.record_relay_persist_failure(SESSION_ID)
    stats.record_relay_persist_failure("conv_other")

    snapshot = stats.snapshot()

    assert snapshot.relay_persist_failures == 3
    assert snapshot.relay_persist_failures_by_session == {
        SESSION_ID: 2,
        "conv_other": 1,
    }


def test_relay_persist_failure_session_map_is_bounded() -> None:
    """Per-session persist-failure detail stops at the session bound."""
    stats = RelayStats(max_sessions=2)
    stats.record_relay_persist_failure("conv_1")
    stats.record_relay_persist_failure("conv_2")
    stats.record_relay_persist_failure("conv_3")
    stats.record_relay_persist_failure("conv_2")

    snapshot = stats.snapshot()

    assert snapshot.relay_persist_failures == 4
    assert snapshot.relay_persist_failures_by_session == {"conv_1": 1, "conv_2": 2}


@pytest.mark.asyncio
async def test_relay_persist_callsite_counts_failures() -> None:
    """_relay_persist increments the failure counter and swallows the error."""
    failing_store = MagicMock()
    failing_store.append.side_effect = RuntimeError("database unavailable")

    await _relay_persist(failing_store, SESSION_ID, MagicMock())

    snapshot = relay_stats.snapshot()
    assert snapshot.relay_persist_failures == 1
    assert snapshot.relay_persist_failures_by_session == {SESSION_ID: 1}


@pytest.mark.asyncio
async def test_relay_persist_success_does_not_count_failures() -> None:
    """A successful persist leaves the failure counter at zero."""
    ok_store = MagicMock()

    await _relay_persist(ok_store, SESSION_ID, MagicMock())

    assert relay_stats.snapshot().relay_persist_failures == 0


@pytest.mark.asyncio
async def test_registry_body_frames_update_the_queue_gauge() -> None:
    """A tunneled response body frame updates depth/bytes and close releases it."""
    registry = TunnelRegistry()
    registry.register("runner-stats", _NoopWS(), _hello())
    registry.open_request("runner-stats", "req-1")

    registry.route_response_frame(
        "runner-stats",
        ResponseBodyFrame(id="req-1", body="abc"),
    )

    snapshot = relay_stats.snapshot()
    assert snapshot.body_queue.depth == 1
    assert snapshot.body_queue.bytes == 3
    assert snapshot.body_queue.requests["req-1"].bytes == 3

    registry.close_request("runner-stats", "req-1")

    released = relay_stats.snapshot()
    assert released.body_queue.depth == 0
    assert released.body_queue.bytes == 0
    assert "req-1" not in released.body_queue.requests


@pytest.mark.asyncio
async def test_stream_live_events_records_subscriber_serialization() -> None:
    """Each SSE frame the subscriber serializes records its duration."""
    generator = _stream_live_events(
        _ConnectedRequest(),  # type: ignore[arg-type]
        SESSION_ID,
    )
    first = await asyncio.wait_for(generator.__anext__(), timeout=2.0)
    assert "session.heartbeat" in first

    session_stream.publish(
        SESSION_ID,
        {"type": "response.output_text.delta", "delta": "hi"},
    )
    second = await asyncio.wait_for(generator.__anext__(), timeout=2.0)
    assert "response.output_text.delta" in second
    await generator.aclose()

    snapshot = relay_stats.snapshot()
    assert snapshot.subscriber_serialization_count == 2
    session = snapshot.subscriber_serialization_by_session[SESSION_ID]
    assert session.count == 2
    assert session.seconds_total >= 0.0


def test_log_record_is_a_structured_relay_stats_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The periodic record carries every counter as JSON plus log attributes."""
    stats = RelayStats()
    stats.record_body_queue_enqueue("req-1", byte_count=7)
    stats.record_event(SESSION_ID, "response.output_text.delta", source=SOURCE_RELAY_SSE)
    stats.record_subscriber_serialization(SESSION_ID, 0.002)
    stats.record_relay_stretch(SESSION_ID, 0.003)
    stats.record_relay_persist_failure(SESSION_ID)

    with caplog.at_level(logging.INFO, logger="omnigent.server.relay_stats"):
        relay_stats.log_relay_stats_snapshot(stats.snapshot())

    records = [r for r in caplog.records if r.getMessage().startswith(RELAY_STATS_LOG_EVENT)]
    assert len(records) == 1
    record = records[0]
    assert record.__dict__["event_name"] == RELAY_STATS_LOG_EVENT

    payload = json.loads(record.getMessage()[len(RELAY_STATS_LOG_EVENT) :].strip())
    assert payload["body_queue"]["depth"] == 1
    assert payload["body_queue"]["bytes"] == 7
    assert payload["events"]["by_source"][SOURCE_RELAY_SSE] == 1
    assert (
        payload["events"]["sessions"][SESSION_ID]["types"]["response.output_text.delta"]["count"]
        == 1
    )
    assert payload["subscriber_serialization"]["count"] == 1
    assert payload["relay_stretch"]["count"] == 1
    assert payload["relay_persist_failures"]["total"] == 1

    attributes = record.__dict__["attributes"]
    assert attributes["window_seconds"] >= 0.0
    assert attributes["body_queue_depth"] == 1
    assert attributes["events_relay_sse"] == 1
    assert attributes["subscriber_serializations"] == 1
    assert attributes["relay_stretches"] == 1
    assert attributes["relay_persist_failures"] == 1


def test_log_payload_bounds_session_detail() -> None:
    """The log payload keeps only the busiest sessions, so one line stays small."""
    stats = RelayStats()
    for index in range(3):
        for _ in range(index + 1):
            stats.record_event(
                f"conv_{index}",
                "response.output_text.delta",
                source=SOURCE_RELAY_SSE,
            )
        stats.record_subscriber_serialization(f"conv_{index}", 0.001 * (index + 1))
        stats.record_relay_stretch(f"conv_{index}", 0.001 * (index + 1))
    stats.record_relay_persist_failure("conv_2")

    payload = relay_stats.build_relay_stats_log_payload(stats.snapshot(), max_sessions=2)

    assert set(payload["events"]["sessions"]) == {"conv_2", "conv_1"}
    assert set(payload["subscriber_serialization"]["sessions"]) == {"conv_2", "conv_1"}
    assert set(payload["relay_stretch"]["sessions"]) == {"conv_2", "conv_1"}
    assert payload["relay_persist_failures"]["sessions"] == {"conv_2": 1}


@pytest.mark.asyncio
async def test_log_relay_stats_periodically_emits_until_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The periodic task keeps calling the snapshot logger until cancelled."""
    calls: list[None] = []

    def record_call() -> None:
        calls.append(None)

    monkeypatch.setattr(relay_stats, "log_relay_stats_snapshot", record_call)

    task = asyncio.create_task(relay_stats.log_relay_stats_periodically(0.01))
    try:
        for _ in range(100):
            if calls:
                break
            await asyncio.sleep(0.01)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert calls, "periodic relay stats task never emitted a record"


def test_interval_env_var_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The interval env var accepts seconds and disables on zero or negative."""
    monkeypatch.delenv(RELAY_STATS_INTERVAL_ENV_VAR, raising=False)
    assert relay_stats.relay_stats_interval_seconds() == DEFAULT_RELAY_STATS_INTERVAL_S

    monkeypatch.setenv(RELAY_STATS_INTERVAL_ENV_VAR, "2.5")
    assert relay_stats.relay_stats_interval_seconds() == pytest.approx(2.5)

    monkeypatch.setenv(RELAY_STATS_INTERVAL_ENV_VAR, "0")
    assert relay_stats.relay_stats_interval_seconds() is None

    monkeypatch.setenv(RELAY_STATS_INTERVAL_ENV_VAR, "-3")
    assert relay_stats.relay_stats_interval_seconds() is None

    monkeypatch.setenv(RELAY_STATS_INTERVAL_ENV_VAR, "not-a-number")
    assert relay_stats.relay_stats_interval_seconds() == DEFAULT_RELAY_STATS_INTERVAL_S


def test_body_byte_length_counts_utf8_bytes_not_characters() -> None:
    """Body-queue bytes are the body's UTF-8 size, not its character count."""
    assert relay_stats.body_byte_length("") == 0
    assert relay_stats.body_byte_length("abc") == 3
    assert relay_stats.body_byte_length("é") == 2
    assert relay_stats.body_byte_length("😀") == 4


def test_dequeued_body_byte_length_matches_the_enqueue_count() -> None:
    """A body's dequeue count equals its enqueue count for every encoding."""
    for body, encoding in (("😀", "utf-8"), ("héllo", "utf-8"), ("YWJj", "base64")):
        decoded = decode_body(body, encoding)
        assert relay_stats.dequeued_body_byte_length(body, encoding, decoded) == (
            relay_stats.body_byte_length(body)
        )


@pytest.mark.asyncio
async def test_non_ascii_body_counts_utf8_bytes_on_enqueue_and_dequeue() -> None:
    """An emoji body adds four bytes and gives exactly those bytes back."""
    registry = TunnelRegistry()
    registry.register("runner-stats", _NoopWS(), _hello())
    state = registry.open_request("runner-stats", "req-emoji")

    registry.route_response_frame("runner-stats", ResponseBodyFrame(id="req-emoji", body="😀"))

    queued = relay_stats.snapshot().body_queue
    assert queued.bytes == 4
    assert queued.requests["req-emoji"].bytes == 4

    state.body_queue.put_nowait(None)
    stream = _TunneledByteStream(registry, "runner-stats", "req-emoji", state)
    assert [chunk async for chunk in stream] == ["😀".encode()]

    drained = relay_stats.snapshot().body_queue
    assert drained.depth == 0
    assert drained.bytes == 0
    assert "req-emoji" not in drained.requests


def _stretch_timer(
    clock: _FakeClock,
    records: list[float],
    probe: relay_stats.BodyWaitProbe | None = None,
) -> relay_stats.RelayStretchTimer:
    """Build a stretch timer whose samples land in ``records``."""

    def record(session_id: str, seconds: float) -> None:
        del session_id
        records.append(seconds)

    return relay_stats.RelayStretchTimer(
        SESSION_ID, body_wait_probe=probe, clock=clock, record=record
    )


def test_relay_stretch_timer_reports_one_stretch_for_a_queued_drain() -> None:
    """Chunks served without a body wait are one uninterrupted loop stretch."""
    clock = _FakeClock()
    records: list[float] = []
    timer = _stretch_timer(clock, records, relay_stats.BodyWaitProbe())

    for _ in range(3):
        timer.chunk_received()
        timer.chunk_processed()
        clock.now += 0.010

    assert records == []
    timer.finish()
    assert records == [pytest.approx(0.030)]


def test_relay_stretch_timer_ends_the_stretch_where_the_body_wait_started() -> None:
    """A body-queue wait ends the stretch at the suspension, not the arrival."""
    clock = _FakeClock()
    records: list[float] = []
    probe = relay_stats.BodyWaitProbe()
    timer = _stretch_timer(clock, records, probe)

    clock.now += 0.005
    probe.note_wait_started(clock.now)
    clock.now += 0.095  # suspended until the frame arrived
    timer.chunk_received()
    clock.now += 0.002
    timer.finish()

    assert records == [pytest.approx(0.005), pytest.approx(0.002)]


def test_relay_stretch_timer_ends_a_terminal_body_wait_where_it_started() -> None:
    """A body wait that ends the stream is a suspension, not relay time."""
    clock = _FakeClock()
    records: list[float] = []
    probe = relay_stats.BodyWaitProbe()
    timer = _stretch_timer(clock, records, probe)

    clock.now += 0.005
    probe.note_wait_started(clock.now)
    # The pending read ends with EOF, cancellation, timeout, or disconnect.
    clock.now += 0.100
    timer.finish()

    assert records == [pytest.approx(0.005)]


def test_relay_stretch_timer_treats_chunks_as_suspensions_without_a_probe() -> None:
    """A plain HTTP runner stream suspends for every chunk it yields."""
    clock = _FakeClock()
    records: list[float] = []
    timer = _stretch_timer(clock, records)

    timer.chunk_received()
    clock.now += 0.010
    timer.chunk_processed()
    timer.chunk_received()
    clock.now += 0.020
    timer.chunk_processed()
    timer.chunk_received()
    timer.finish()

    assert records == [pytest.approx(0.010), pytest.approx(0.020), pytest.approx(0.0)]


@pytest.mark.asyncio
async def test_relay_stretch_timer_excludes_an_awaited_operation() -> None:
    """An awaited off-loop operation is a suspension boundary, not loop time."""
    clock = _FakeClock()
    records: list[float] = []
    timer = _stretch_timer(clock, records, relay_stats.BodyWaitProbe())

    clock.now += 0.003

    async def offload() -> str:
        clock.now += 5.0
        return "persisted"

    assert await timer.bound(offload()) == "persisted"
    assert records == [pytest.approx(0.003)]

    clock.now += 0.004
    timer.finish()
    assert records == [pytest.approx(0.003), pytest.approx(0.004)]


def test_body_wait_probe_is_read_from_the_response_extensions() -> None:
    """The relay picks up the tunnel's probe and ignores responses without one."""
    probe = relay_stats.BodyWaitProbe()
    response = MagicMock()
    response.extensions = {relay_stats.BODY_WAIT_PROBE_EXTENSION_KEY: probe}

    assert relay_stats.body_wait_probe_for_response(response) is probe
    assert relay_stats.body_wait_probe_for_response(MagicMock(extensions={})) is None


# ── Phase 2: body-queue guardrails ──────────────────────────────────


def test_body_queue_warn_guard_fires_once_per_request() -> None:
    """Crossing the warn threshold reports the request once, never again."""
    stats = RelayStats(warn_threshold_bytes=100, fuse_threshold_bytes=10_000)

    assert stats.record_body_queue_enqueue("req-warn", 60).warn is False
    assert stats.record_body_queue_enqueue("req-warn", 60).warn is True
    assert stats.record_body_queue_enqueue("req-warn", 60).warn is False

    snapshot = stats.snapshot()
    assert snapshot.body_queue.warned_requests == 1
    assert snapshot.body_queue.warn_bytes == 100
    assert snapshot.body_queue.requests["req-warn"].warned is True


def test_body_queue_fuse_guard_trips_only_the_request_over_the_limit() -> None:
    """A request past the fuse reports a trip; its neighbour stays below it."""
    stats = RelayStats(warn_threshold_bytes=100, fuse_threshold_bytes=150)

    assert stats.record_body_queue_enqueue("req-fuse", 60).fuse_tripped is False
    assert stats.record_body_queue_enqueue("req-fuse", 60).fuse_tripped is False
    tripped = stats.record_body_queue_enqueue("req-fuse", 60)
    assert tripped.fuse_tripped is True
    assert tripped.request_bytes == 180
    assert stats.record_body_queue_enqueue("req-other", 60).fuse_tripped is False

    snapshot = stats.snapshot()
    assert snapshot.body_queue.fuse_trips == 1
    assert snapshot.body_queue.fuse_bytes == 150


def test_body_queue_fuse_env_var_overrides_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fuse threshold comes from the environment, defaulting to 256 MiB."""
    monkeypatch.delenv(relay_stats.BODY_QUEUE_FUSE_ENV_VAR, raising=False)
    assert relay_stats.body_queue_fuse_bytes() == relay_stats.DEFAULT_BODY_QUEUE_FUSE_BYTES
    assert relay_stats.DEFAULT_BODY_QUEUE_FUSE_BYTES == 256 * 1024 * 1024

    monkeypatch.setenv(relay_stats.BODY_QUEUE_FUSE_ENV_VAR, "1048576")
    assert relay_stats.body_queue_fuse_bytes() == 1048576

    for invalid in ("0", "-1", "junk", "inf"):
        monkeypatch.setenv(relay_stats.BODY_QUEUE_FUSE_ENV_VAR, invalid)
        assert relay_stats.body_queue_fuse_bytes() == relay_stats.DEFAULT_BODY_QUEUE_FUSE_BYTES


def test_outbound_queue_gauge_tracks_depth_and_release() -> None:
    """The outbound gauge follows the queue depth and forgets closed tunnels."""
    stats = RelayStats()
    stats.record_outbound_queue_depth("runner-out", 1)
    stats.record_outbound_queue_depth("runner-out", 3)
    stats.record_outbound_queue_depth("runner-out", 2)

    snapshot = stats.snapshot()
    assert snapshot.outbound_queue.depth == 2
    assert snapshot.outbound_queue.max_depth == 3
    assert snapshot.outbound_queue.runners == {"runner-out": 2}

    stats.record_outbound_queue_released("runner-out")
    released = stats.snapshot()
    assert released.outbound_queue.depth == 0
    assert released.outbound_queue.runners == {}


@pytest.mark.asyncio
async def test_registry_outbound_queue_gauge_tracks_the_sender_drain() -> None:
    """A frame queued for the sender raises the gauge; the sender lowers it."""
    registry = TunnelRegistry()
    session = registry.register("runner-out", _NoopWS(), _hello())

    await registry.send_text(session, "{}")

    queued = relay_stats.snapshot()
    assert queued.outbound_queue.depth == 1
    assert queued.outbound_queue.max_depth == 1
    assert queued.outbound_queue.runners == {"runner-out": 1}

    sender = asyncio.create_task(_sender_loop(session.ws, session))
    try:
        await asyncio.sleep(0.01)
        drained = relay_stats.snapshot()
        assert drained.outbound_queue.depth == 0
        assert drained.outbound_queue.max_depth == 1
    finally:
        sender.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sender

    registry.deregister("runner-out")
    assert relay_stats.snapshot().outbound_queue.runners == {}


@pytest.mark.asyncio
async def test_registry_warns_once_per_request_above_eight_mib(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Crossing 8 MiB warns once for that request and leaves neighbours alone."""
    registry = TunnelRegistry()
    registry.register("runner-warn", _NoopWS(), _hello())
    registry.open_request("runner-warn", "req-big")
    registry.open_request("runner-warn", "req-small")
    chunk = "x" * (512 * 1024)

    with caplog.at_level(logging.WARNING, logger="omnigent.runner.transports.ws_tunnel.registry"):
        for _ in range(17):  # 8.5 MiB
            registry.route_response_frame(
                "runner-warn", ResponseBodyFrame(id="req-big", body=chunk)
            )
        for _ in range(4):  # 2 MiB stays quiet
            registry.route_response_frame(
                "runner-warn", ResponseBodyFrame(id="req-small", body=chunk)
            )

    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and "req-big" in record.getMessage()
    ]
    assert len(warnings) == 1, "the warn threshold fires once per request"
    snapshot = relay_stats.snapshot()
    assert snapshot.body_queue.warned_requests == 1
    assert snapshot.body_queue.requests["req-small"].warned is False


@pytest.mark.asyncio
async def test_body_queue_fuse_aborts_only_the_offending_request(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One runaway request aborts while its neighbour keeps streaming."""
    monkeypatch.setenv(relay_stats.BODY_QUEUE_FUSE_ENV_VAR, "2048")
    relay_stats.reset_for_tests()

    registry = TunnelRegistry()
    registry.register("runner-fuse", _NoopWS(), _hello())
    victim_state = registry.open_request("runner-fuse", "req-victim")
    other_state = registry.open_request("runner-fuse", "req-other")
    for request_id in ("req-victim", "req-other"):
        registry.route_response_frame("runner-fuse", ResponseHeadFrame(id=request_id, status=200))
    victim = _TunneledByteStream(registry, "runner-fuse", "req-victim", victim_state)
    other = _TunneledByteStream(registry, "runner-fuse", "req-other", other_state)
    other_chunks: list[bytes] = []

    async def drain_other() -> None:
        async for chunk in other:
            other_chunks.append(chunk)

    other_task = asyncio.create_task(drain_other())

    with caplog.at_level(logging.ERROR, logger="omnigent.runner.transports.ws_tunnel.registry"):
        for _ in range(3):  # 3048 bytes > 2048 byte fuse on the third frame
            registry.route_response_frame(
                "runner-fuse", ResponseBodyFrame(id="req-victim", body="v" * 1024)
            )

    assert victim_state.aborted_with is not None
    errors = [
        record
        for record in caplog.records
        if record.levelno == logging.ERROR and "fuse" in record.getMessage().lower()
    ]
    assert len(errors) == 1
    assert "req-victim" in errors[0].getMessage()

    async def drain_victim() -> None:
        async for _ in victim:
            pass

    with pytest.raises(BodyQueueFuseExceeded):
        await asyncio.wait_for(drain_victim(), timeout=2.0)

    registry.route_response_frame(
        "runner-fuse", ResponseBodyFrame(id="req-other", body="still streaming")
    )
    assert list(registry.get("runner-fuse").in_flight) == ["req-other"], (
        "the neighbour request stays in flight while the victim aborted"
    )
    registry.route_response_frame("runner-fuse", ResponseEndFrame(id="req-other"))
    await asyncio.wait_for(other_task, timeout=2.0)

    assert other_chunks == [b"still streaming"]
    assert registry.get("runner-fuse") is not None, "the runner socket stays up"
    snapshot = relay_stats.snapshot()
    assert snapshot.body_queue.fuse_trips == 1
    assert "req-other" not in snapshot.body_queue.requests


@pytest.mark.asyncio
async def test_body_queue_keeps_every_frame_queued_below_the_fuse() -> None:
    """The queue is unbounded by count and drops nothing under the fuse."""
    relay_stats.reset_for_tests()
    registry = TunnelRegistry()
    registry.register("runner-many", _NoopWS(), _hello())
    state = registry.open_request("runner-many", "req-many")
    registry.route_response_frame("runner-many", ResponseHeadFrame(id="req-many", status=200))
    stream = _TunneledByteStream(registry, "runner-many", "req-many", state)

    received: list[bytes] = []

    async def drain() -> None:
        async for chunk in stream:
            received.append(chunk)

    task = asyncio.create_task(drain())
    frame_count = 4096
    for _ in range(frame_count):  # No awaits, so every frame is queued first.
        registry.route_response_frame("runner-many", ResponseBodyFrame(id="req-many", body="y"))

    queued = relay_stats.snapshot().body_queue.requests["req-many"]
    assert queued.depth == frame_count, "the queue keeps every frame below the fuse"

    registry.route_response_frame("runner-many", ResponseEndFrame(id="req-many"))
    await asyncio.wait_for(task, timeout=5.0)

    assert received == [b"y"] * frame_count
    snapshot = relay_stats.snapshot()
    assert snapshot.body_queue.fuse_trips == 0
    assert snapshot.body_queue.warned_requests == 0


def test_outbound_queue_gauge_is_bounded_and_releases_replaced_tunnels() -> None:
    """The gauge keeps a bounded runner map and forgets replaced generations."""
    stats = RelayStats()
    stats.record_outbound_queue_depth("runner-1", 2)
    for index in range(300):
        stats.record_outbound_queue_depth(f"runner-bulk-{index}", 1)

    snapshot = stats.snapshot()
    assert snapshot.outbound_queue.runners_tracked <= 256
    assert snapshot.outbound_queue.dropped_samples > 0
    assert stats.record_outbound_queue_depth("runner-new", 1) is False

    stats.record_outbound_queue_released("runner-1")
    assert "runner-1" not in stats.snapshot().outbound_queue.runners


@pytest.mark.asyncio
async def test_stream_live_events_counts_subscriber_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disconnected slow subscriber increments the overflow counter."""
    monkeypatch.setattr(session_stream, "_SUBSCRIBER_QUEUE_MAX_EVENTS", 2)
    generator = _stream_live_events(
        _ConnectedRequest(),  # type: ignore[arg-type]
        SESSION_ID,
    )

    first = await asyncio.wait_for(generator.__anext__(), timeout=2.0)
    assert "session.heartbeat" in first

    for index in range(4):
        session_stream.publish(
            SESSION_ID, {"type": "response.output_text.delta", "delta": str(index)}
        )

    async def drain() -> None:
        async for _ in generator:
            pass

    # The overflow closes the stream instead of yielding more frames.
    await asyncio.wait_for(drain(), timeout=2.0)

    assert relay_stats.snapshot().subscriber_overflows == 1


def test_log_payload_exposes_the_phase_two_gauges_and_counters() -> None:
    """The structured record carries the guardrail gauges and their counters."""
    stats = RelayStats(warn_threshold_bytes=100, fuse_threshold_bytes=150)
    stats.record_body_queue_enqueue("req-guarded", 180)
    stats.record_outbound_queue_depth("runner-1", 2)
    stats.record_subscriber_overflow()
    stats.snapshot()  # reset the rate window so the record is deterministic

    payload = relay_stats.build_relay_stats_log_payload(stats.snapshot())

    assert payload["body_queue"]["warned_requests"] == 1
    assert payload["body_queue"]["warn_bytes"] == 100
    assert payload["body_queue"]["fuse"] == {"trips": 1, "limit_bytes": 150}
    assert payload["body_queue"]["requests"] == {
        "req-guarded": {"depth": 1, "bytes": 180, "warned": True}
    }
    assert payload["outbound_queue"]["depth"] == 2
    assert payload["outbound_queue"]["max_depth"] == 2
    assert payload["outbound_queue"]["runners"] == {"runner-1": 2}
    assert payload["session_stream"] == {"subscriber_overflows": 1}

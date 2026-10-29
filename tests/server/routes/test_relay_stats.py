"""Unit tests for the Phase 0 relay counters.

Each test pins one observable counter, its bound, or the structured
``relay_stats`` log record that the load harness reads.
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
    decode_body,
)
from omnigent.runner.transports.ws_tunnel.registry import TunnelRegistry
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


def test_saturated_request_map_keeps_aggregates_equal_to_tracked_requests() -> None:
    """A request admitted after saturation cannot subtract older uncounted frames."""
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
    _assert_aggregate_is_the_tracked_sum(saturated)

    # A slot frees and the request is admitted with one small counted frame
    # while its two older uncounted frames are still ahead in its queue.
    stats.record_body_queue_released("req-0")
    stats.record_body_queue_enqueue("req-late", byte_count=10)
    admitted = stats.snapshot().body_queue
    assert admitted.requests["req-late"].bytes == 10
    _assert_aggregate_is_the_tracked_sum(admitted)

    # Draining the uncounted frames may exhaust its own gauge, never others'.
    stats.record_body_queue_dequeue("req-late", byte_count=100)
    after_first = stats.snapshot().body_queue
    _assert_aggregate_is_the_tracked_sum(after_first)
    assert after_first.requests["req-1"].bytes == 100

    stats.record_body_queue_dequeue("req-late", byte_count=100)
    drained = stats.snapshot().body_queue
    _assert_aggregate_is_the_tracked_sum(drained)
    assert drained.requests["req-late"].bytes == 0
    assert drained.requests["req-1"].bytes == 100
    assert drained.bytes == 511 * 100
    assert drained.depth == 511


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

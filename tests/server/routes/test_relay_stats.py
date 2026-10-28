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
)
from omnigent.runner.transports.ws_tunnel.registry import TunnelRegistry
from omnigent.runtime import session_stream
from omnigent.server import presence, relay_stats, shutdown_state
from omnigent.server.relay_stats import (
    DEFAULT_RELAY_STATS_INTERVAL_S,
    RELAY_STATS_INTERVAL_ENV_VAR,
    RELAY_STATS_LOG_EVENT,
    SOURCE_EVENT_BATCH,
    SOURCE_HTTP_POST,
    SOURCE_RELAY_SSE,
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
